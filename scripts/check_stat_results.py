#!/usr/bin/env python
"""Verify the statistical-rigor smoke-test outputs. Prints a PASS/FAIL checklist.

Usage: python scripts/check_stat_results.py <OUT_dir>
"""
import glob
import json
import sys


def load(out_sub):
    fs = sorted(glob.glob(f"{out_sub}/**/results_*.json", recursive=True))
    if not fs:
        return None
    return json.load(open(fs[-1], encoding="utf-8"))["results"]


def task_block(results):
    # return the single task's metric dict (skip group wrappers)
    for name, d in results.items():
        if isinstance(d, dict) and any("," in k for k in d):
            return name, d
    name = next(iter(results))
    return name, results[name]


PASS, FAIL = "PASS", "FAIL"
checks = []


def check(desc, ok, evidence=""):
    checks.append((PASS if ok else FAIL, desc, evidence))


def fam_present(d, base, fk):
    return all(
        f"{base}{suf},{fk}" in d
        for suf in ("", "_stderr", "_ci_low", "_ci_high", "_n", "_clusters", "_stat_warnings")
    )


def ci_width(d, base, fk):
    return d[f"{base}_ci_high,{fk}"] - d[f"{base}_ci_low,{fk}"]


base_out = sys.argv[1] if len(sys.argv) > 1 else "./stat_smoke_out"

# ---- A: arc_easy IID family ----
r = load(f"{base_out}/arc95")
if r is None:
    check("[A] arc_easy ran", False, "no results json")
else:
    _, d = task_block(r)
    check("[A] arc acc rigorous SE family present", fam_present(d, "acc", "none"))
    check("[A] arc acc_norm family present", fam_present(d, "acc_norm", "none"))
    n, c = d.get("acc_n,none"), d.get("acc_clusters,none")
    check("[A] arc IID: _clusters == _n", n == c, f"_n={n} _clusters={c}")

# ---- B: confidence_level narrows CI ----
r90 = load(f"{base_out}/arc90")
if r is not None and r90 is not None:
    _, d95 = task_block(r)
    _, d90 = task_block(r90)
    w95, w90 = ci_width(d95, "acc", "none"), ci_width(d90, "acc", "none")
    check("[B] CI(0.90) narrower than CI(0.95)", w90 < w95, f"w90={w90:.4f} < w95={w95:.4f}")
else:
    check("[B] confidence_level run", False, "missing arc90 results")

# ---- C: clustered SE on xnli (premise) ----
r = load(f"{base_out}/xnli")
if r is None:
    check("[C] xnli ran", False, "no results json (try xquad_en instead)")
else:
    _, d = task_block(r)
    check("[C] xnli acc family present", fam_present(d, "acc", "none"))
    n, c = d.get("acc_n,none"), d.get("acc_clusters,none")
    check("[C] clustered: _clusters < _n", (c is not None and n is not None and c < n),
          f"_n={n} _clusters={c}")

# ---- D: avg@k + pass@k ----
r = load(f"{base_out}/gsm8k_stat")
if r is None:
    check("[D] gsm8k_stat ran", False, "no results json")
else:
    _, d = task_block(r)
    fk = "avg@K"
    check("[D] avg@k column + family", fam_present(d, "exact_match", fk))
    check("[D] pass@1 column + family", fam_present(d, "pass@1", fk))
    check("[D] pass@2 column + family", fam_present(d, "pass@2", fk))
    p1, p2 = d.get(f"pass@1,{fk}"), d.get(f"pass@2,{fk}")
    check("[D] pass@k monotonic: pass@2 >= pass@1",
          (p1 is not None and p2 is not None and p2 >= p1 - 1e-9),
          f"pass@1={p1} pass@2={p2}")
    n, c = d.get(f"pass@1_n,{fk}"), d.get(f"pass@1_clusters,{fk}")
    check("[D] gsm8k_stat IID: _clusters == _n", n == c, f"_n={n} _clusters={c}")
    # reserved keys must NOT leak as columns
    leaked = [k for k in d if "__" in k]
    check("[D] reserved __*__ keys quarantined", not leaked, f"leaked={leaked}")

# ---- report ----
print("\n================ STAT SMOKE-TEST CHECKLIST ================")
npass = sum(1 for s, *_ in checks if s == PASS)
for status, desc, ev in checks:
    line = f"  [{status}] {desc}"
    if ev:
        line += f"   ({ev})"
    print(line)
print(f"----------------------------------------------------------")
print(f"  {npass}/{len(checks)} checks passed")
print("==========================================================")
sys.exit(0 if npass == len(checks) else 1)
