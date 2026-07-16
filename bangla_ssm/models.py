"""Build parameter-matched Transformer and Mamba-3 causal LMs.

Both models expose: model(input_ids).logits with shape (B, L, vocab).

- Transformer: Llama-style decoder (RoPE, SwiGLU, RMSNorm) via HF transformers.
- Mamba-3: official implementation via mamba_ssm.MambaLMHeadModel with
  ssm_cfg={"layer": "Mamba3"} (SISO/Triton path by default; MIMO needs TileLang).
"""

from __future__ import annotations

from .config import ExpConfig


def build_model(cfg: ExpConfig, device=None, dtype=None):
    if cfg.arch == "transformer":
        return _build_transformer(cfg, device=device, dtype=dtype)
    if cfg.arch == "mamba3":
        return _build_mamba_family(cfg, set(), device=device, dtype=dtype)
    if cfg.arch == "hybrid":
        return _build_mamba_family(cfg, set(cfg.attn_layer_idx), device=device, dtype=dtype)
    raise ValueError(cfg.arch)


def _build_transformer(cfg: ExpConfig, device=None, dtype=None):
    from transformers import LlamaConfig, LlamaForCausalLM

    hf_cfg = LlamaConfig(
        vocab_size=cfg.vocab_size,
        hidden_size=cfg.d_model,
        num_hidden_layers=cfg.n_layer,
        num_attention_heads=cfg.n_head,
        num_key_value_heads=cfg.n_head,  # full MHA; no GQA at this scale
        intermediate_size=cfg.intermediate_size,
        max_position_embeddings=cfg.seq_len,
        rope_theta=cfg.rope_theta,
        tie_word_embeddings=cfg.tie_embeddings,
        attention_dropout=0.0,
        use_cache=False,
    )
    hf_cfg._attn_implementation = "sdpa"
    model = LlamaForCausalLM(hf_cfg)
    if device is not None or dtype is not None:
        model = model.to(device=device, dtype=dtype)
    return model


def _build_mamba_family(cfg: ExpConfig, attn_idx: set, device=None, dtype=None):
    """Mamba-3 (attn_idx empty) or a hybrid tower (attn_idx = attention layers).

    A uniform pre-norm RMSNorm residual tower where each layer's token mixer is
    either the official Mamba3 block or a causal RoPE self-attention block. This
    keeps the architecture comparison clean — only the mixer type varies per
    layer — and mirrors the Samba/Jamba recipe of sprinkling a few attention
    layers into an SSM backbone to recover local precision.

    We build the tower ourselves because the released mamba-ssm wheel's
    MambaLMHeadModel.create_block only wires Mamba1/Mamba2 (Mamba3 lives only on
    git main, not in any prebuilt wheel).
    """
    import math
    from types import SimpleNamespace

    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from mamba_ssm.modules.mamba3 import Mamba3

    fk = {"device": device, "dtype": dtype}

    def _init(module):
        # GPT-NeoX / Mamba recipe: embeddings ~ N(0, 0.02), zero linear biases.
        # Mamba3's internal special params (dt_bias, B/C_bias, D, B/C_norm) are
        # not Linear/Embedding, so they keep their designed initialization.
        if isinstance(module, nn.Linear):
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, std=0.02)

    def rmsnorm(dim):
        if hasattr(nn, "RMSNorm"):  # torch >= 2.4
            return nn.RMSNorm(dim, eps=1e-5, **fk)
        return _FallbackRMSNorm(dim, **fk)

    class Attention(nn.Module):
        """Causal multi-head self-attention with RoPE (matches the baseline)."""

        def __init__(self):
            super().__init__()
            self.n_head = cfg.n_head
            self.head_dim = cfg.d_model // cfg.n_head
            assert self.head_dim % 2 == 0, "head_dim must be even for RoPE"
            self.qkv = nn.Linear(cfg.d_model, 3 * cfg.d_model, bias=False, **fk)
            self.out_proj = nn.Linear(cfg.d_model, cfg.d_model, bias=False, **fk)
            inv = 1.0 / (cfg.rope_theta ** (
                torch.arange(0, self.head_dim, 2, device=device).float() / self.head_dim))
            t = torch.arange(cfg.seq_len, device=device).float()
            freqs = torch.outer(t, inv)
            self.register_buffer("cos", freqs.cos()[None, None], persistent=False)
            self.register_buffer("sin", freqs.sin()[None, None], persistent=False)

        def _rope(self, x):  # x: (B, H, L, D)
            L = x.shape[2]
            cos, sin = self.cos[..., :L, :], self.sin[..., :L, :]
            x1, x2 = x[..., ::2], x[..., 1::2]
            xr1 = x1 * cos - x2 * sin
            xr2 = x1 * sin + x2 * cos
            return torch.stack([xr1, xr2], dim=-1).flatten(-2)

        def forward(self, x):
            B, L, _ = x.shape
            qkv = self.qkv(x).view(B, L, 3, self.n_head, self.head_dim)
            q, k, v = qkv.unbind(2)                       # each (B, L, H, D)
            q, k, v = (t.transpose(1, 2) for t in (q, k, v))  # (B, H, L, D)
            q, k = self._rope(q.float()).type_as(v), self._rope(k.float()).type_as(v)
            o = F.scaled_dot_product_attention(q, k, v, is_causal=True)
            return self.out_proj(o.transpose(1, 2).reshape(B, L, cfg.d_model))

    class Block(nn.Module):
        def __init__(self, layer_idx: int):
            super().__init__()
            self.norm = rmsnorm(cfg.d_model)
            if layer_idx in attn_idx:
                self.mixer = Attention()
            else:
                self.mixer = Mamba3(
                    d_model=cfg.d_model, d_state=cfg.d_state, expand=cfg.expand,
                    headdim=cfg.headdim, is_mimo=cfg.is_mimo, mimo_rank=cfg.mimo_rank,
                    layer_idx=layer_idx, **fk,
                )

        def forward(self, x):
            return x + self.mixer(self.norm(x))

    class MambaFamilyLM(nn.Module):
        def __init__(self):
            super().__init__()
            self.embedding = nn.Embedding(cfg.vocab_size, cfg.d_model, **fk)
            self.layers = nn.ModuleList(Block(i) for i in range(cfg.n_layer))
            self.norm_f = rmsnorm(cfg.d_model)
            self.lm_head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False, **fk)

            # standard init (fixes default N(0,1) embedding -> huge tied-head logits)
            self.apply(_init)
            # scale down residual out-projections by 1/sqrt(2 * n_layer) for depth
            for name, p in self.named_parameters():
                if name.endswith("out_proj.weight"):
                    with torch.no_grad():
                        p.div_(math.sqrt(2 * cfg.n_layer))

            if cfg.tie_embeddings:
                self.lm_head.weight = self.embedding.weight

        def forward(self, input_ids, **kw):
            x = self.embedding(input_ids)
            for layer in self.layers:
                x = layer(x)
            x = self.norm_f(x)
            return SimpleNamespace(logits=self.lm_head(x))

    return MambaFamilyLM()


class _FallbackRMSNorm:  # only used if torch < 2.4 (Colab is far newer)
    def __new__(cls, dim, device=None, dtype=None, eps=1e-5):
        import torch
        import torch.nn as nn

        class RMSNorm(nn.Module):
            def __init__(self):
                super().__init__()
                self.weight = nn.Parameter(torch.ones(dim, device=device, dtype=dtype))
                self.eps = eps

            def forward(self, x):
                v = x.float()
                v = v * torch.rsqrt(v.pow(2).mean(-1, keepdim=True) + self.eps)
                return (v.type_as(x)) * self.weight

        return RMSNorm()


def count_params(model) -> dict:
    """Total and non-embedding parameter counts.

    Embedding/LM-head params are excluded from the matching criterion because
    they are identical across architectures (same vocab, same d_model, tied).
    """
    total = sum(p.numel() for p in model.parameters())
    emb = 0
    seen = set()
    for name, p in model.named_parameters():
        if ("embed" in name or "lm_head" in name) and id(p) not in seen:
            emb += p.numel()
            seen.add(id(p))
    return {"total": total, "embedding": emb, "non_embedding": total - emb}
