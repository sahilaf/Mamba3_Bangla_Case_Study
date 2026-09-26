"""Aggregate per-seed eval JSONs into paper/results.json (any number of seeds).

Reads, from --runs_dir (default: the released results/ directory):
  results_<tag>.json            (minimal-pairs eval; produced by eval_minimal_pairs.py --out)
  ppl_<tag>.json                (perplexity;         produced by eval_ppl.py --out)
where <tag> is like tf_s1, m3_s2, hybrid_s3419, ... (s1 = seed 1337, s2 = seed 2024).

  python paper/aggregate_seeds.py
  python paper/aggregate_seeds.py --runs_dir /content/drive/MyDrive/mamba3_bn/runs

Writes paper/results.json in the multi-seed list format used by figures.py / stats.py,
and prints a mean±sd summary. Run `python paper/figures.py` afterwards to refresh figures.
"""

from __future__ import annotations

import argparse
import glob
import json
import re
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
MODELS = {"tf": "transformer", "m3": "mamba3", "hybrid": "hybrid"}
DIST_ORDER = ["none", "short", "medium", "long"]


def _dumps(obj, level: int = 0) -> str:
    """JSON with nested dicts indented and every list kept on one line."""
    if not isinstance(obj, dict):
        return json.dumps(obj, ensure_ascii=False)
    pad = "  " * (level + 1)
    items = [f"{pad}{json.dumps(k)}: {_dumps(v, level + 1)}" for k, v in obj.items()]
    return "{\n" + ",\n".join(items) + "\n" + "  " * level + "}"


def parse_tag(tag: str):
    """tf_s1 -> ('transformer', '1'); hybrid_s3419 -> ('hybrid', '3419')."""
    m = re.match(r"(tf|m3|hybrid)_s(\w+)", tag)
    if not m:
        return None, None
    return MODELS.get(m.group(1)), m.group(2)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs_dir", default=str(HERE.parent / "results"))
    args = ap.parse_args()
    runs = Path(args.runs_dir)

    # model -> seed -> {probe accuracies, distance arrays, ppl}
    data = defaultdict(dict)
    for path in sorted(glob.glob(str(runs / "results_*.json"))):
        tag = Path(path).stem[len("results_"):]
        model, seed = parse_tag(tag)
        if not model:
            continue
        res = json.loads(Path(path).read_text(encoding="utf-8"))["results"]
        rec = data[model].setdefault(seed, {})
        for probe in ("sva", "attraction", "honorific", "discourse"):
            key = f"{probe}.tsv"
            if key in res:
                rec[f"{probe}_acc"] = res[key]["accuracy"]
        if "sva.tsv" in res:
            dist = res["sva.tsv"]["breakdown"].get("distance", {})
            rec["sva_dist"] = [dist.get(b, {}).get("acc", None) for b in DIST_ORDER]
        # p2int cell
        sp = res.get("sva.tsv", {}).get("breakdown", {}).get("subj_person", {})
        if "p2_int" in sp:
            rec["p2int"] = sp["p2_int"]["acc"]

    for path in sorted(glob.glob(str(runs / "ppl_*.json"))):
        tag = Path(path).stem[len("ppl_"):]
        model, seed = parse_tag(tag)
        if model and seed in data[model]:
            data[model][seed]["ppl"] = json.loads(Path(path).read_text())["ppl"]

    def col(model, field):
        return [data[model][s][field] for s in sorted(data[model]) if field in data[model][s]]

    models = [m for m in MODELS.values() if m in data]
    out = {
        "_note": "Final results, 5 seeds (1337, 2024, 3419, 5150, 8888). 3 architectures, "
                 "~24.5M non-embedding params, 1B tokens Bangla (FineWeb-2 ben_Beng). "
                 "Each list = per-seed values. Aggregated by paper/aggregate_seeds.py.",
        "seeds_per_model": {m: sorted(data[m]) for m in models},
        "perplexity": {m: col(m, "ppl") for m in models},
        "sva_overall": {m: col(m, "sva_acc") for m in models},
        "sva_by_distance": {
            "_order": DIST_ORDER,
            "_comment": "each model -> list of per-seed [none, short, medium, long] arrays",
            **{m: col(m, "sva_dist") for m in models},
        },
        "attraction_overall": {m: col(m, "attraction_acc") for m in models},
        "honorific_overall": {m: col(m, "honorific_acc") for m in models},
        "discourse_overall": {m: col(m, "discourse_acc") for m in models},
        "p2_int_sva": {
            "_note": "2nd-person intimate (তুই); rare register -> frequency artifact, "
                     "wildly seed-variable, reported separately.",
            **{m: col(m, "p2int") for m in models},
        },
    }
    # atomic write (robust to file-sync locks on some filesystems)
    import os
    tmp = HERE / ".results.json.tmp"
    tmp.write_bytes((_dumps(out) + "\n").encode("utf-8"))
    os.replace(tmp, HERE / "results.json")

    print("wrote paper/results.json")
    for m in ("transformer", "mamba3", "hybrid"):
        if m not in data:
            continue
        seeds = out["seeds_per_model"][m]
        print(f"\n{m}  ({len(seeds)} seeds: {', '.join(seeds)})")
        for label, key in [("ppl", "perplexity"), ("SVA", "sva_overall"),
                           ("attraction", "attraction_overall"),
                           ("honorific", "honorific_overall"),
                           ("discourse", "discourse_overall")]:
            vals = out[key].get(m, [])
            if not vals:
                continue
            mean = sum(vals) / len(vals)
            sd = (sum((v - mean) ** 2 for v in vals) / len(vals)) ** 0.5
            unit = "" if key == "perplexity" else "%"
            scale = 1 if key == "perplexity" else 100
            print(f"  {label:11s} {mean*scale:6.2f}{unit} ± {sd*scale:.2f}  (n={len(vals)})")


if __name__ == "__main__":
    main()
