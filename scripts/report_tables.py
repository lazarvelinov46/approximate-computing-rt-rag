# scripts/report_tables.py — NEW FILE (complete)
#!/usr/bin/env python3
"""Markdown result tables for the report, straight from results/*_summary.*.

Reuses the McNemar parsing and verdict logic from plot_phase2.py so the
tables and the figures cannot disagree.

Usage (from repo root):
    python scripts/report_tables.py > results/report_tables.md
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from plot_phase2 import KNOB1_MC, mc_index, verdict  # noqa: E402

SHORT_VERDICT = {"SIGNIFICANT": "sig", "NULL": "null",
                 "NULL (zero-discordant carve-out)": "null (d=0)",
                 "UNDERPOWERED": "null, underpowered",
                 "UNINFORMATIVE": "uninformative"}


def load(name, cols=()):
    d = pd.read_csv(os.path.join(ARGS.results_dir, name))
    for c in cols:
        if c in d.columns:
            d[c] = pd.to_numeric(d[c], errors="coerce")
    return d


def mc_all(name):
    """Comparison records keyed by name AND by b_setting, so a row can be
    looked up by whichever handle its knob used."""
    with open(os.path.join(ARGS.results_dir, name)) as fh:
        mc = mc_index(json.load(fh))
    out = dict(mc)
    for rec in mc.values():
        if "b_setting" in rec:
            out.setdefault(str(rec["b_setting"]), rec)
    return out


def cell(mc, *candidates):
    for key in candidates:
        rec = mc.get(str(key))
        if rec:
            d = int(rec["b"]) + int(rec["c"])
            v = SHORT_VERDICT[verdict(int(rec["b"]), int(rec["c"]),
                                      float(rec["p"]))]
            return f"d={d}, p={float(rec['p']):.3f} ({v})"
    return "—"


def to_md(rows, headers):
    out = ["| " + " | ".join(headers) + " |",
           "|" + "|".join(["---"] * len(headers)) + "|"]
    for r in rows:
        out.append("| " + " | ".join(str(x) for x in r) + " |")
    return "\n".join(out)


def f(x, n=3):
    return "—" if pd.isna(x) else f"{float(x):.{n}f}"


def knob1():
    d = load("knob1_summary.csv", ("ndis", "ann_recall", "complete_frac",
                                   "em", "f1"))
    mc = mc_all("knob1_summary.json")
    order = {"flat": 0, "hnsw": 1, "ivf": 2}
    d = d.assign(_o=d["index"].map(order)).sort_values(["_o", "ndis"],
                                                       ascending=[True, False])
    rows = []
    for _, r in d.iterrows():
        label = "exact (Flat)" if r["index"] == "flat" else \
            f"{r['index'].upper()} {'ef' if r['index']=='hnsw' else 'nprobe'}={int(r.value)}"
        rows.append([label, f"{int(r.ndis):,}", f(r.ann_recall),
                     f(r.complete_frac), f(r.em), f(r.f1),
                     "reference" if r["index"] == "flat"
                     else cell(mc, KNOB1_MC.get(r.setting, ""))])
    return to_md(rows, ["setting", "dist. comps/query", "ANN recall",
                        "both gold", "EM", "F1", "vs exact"])


def knob2():
    d = load("knob2_summary.csv", ("compression", "corpus_mib", "ann_recall",
                                   "exact_match_topk", "complete_frac",
                                   "em", "f1")).sort_values("compression")
    mc = mc_all("knob2_summary.json")
    base = d.iloc[0].setting
    rows = []
    for _, r in d.iterrows():
        rows.append([r.setting.replace("knob2_", "").replace(
            "regime2_a1_baseline", "exact fp32"),
            f"{float(r.compression):.0f}x", f"{float(r.corpus_mib):.1f}",
            f(r.ann_recall), f(r.exact_match_topk), f(r.complete_frac),
            f(r.em), f(r.f1),
            "reference" if r.setting == base
            else cell(mc, f"exact_vs_{r.setting}")])
    return to_md(rows, ["setting", "compression", "corpus MiB", "ANN recall",
                        "top-5 identical", "both gold", "EM", "F1",
                        "vs exact"])


def knob3():
    s = load("knob3_summary.csv", ("nbits", "q_group_size", "residual_length",
                                   "effective_bits", "em", "f1",
                                   "abstain_rate"))
    e = load("knob3_explain_summary.csv", ("effective_bits", "em"))
    mcs, mce = mc_all("knob3_summary.json"), mc_all("knob3_explain_summary.json")
    base = s.loc[s.effective_bits.idxmax()].setting
    s = s.sort_values("effective_bits", ascending=False)
    ex = dict(zip(e.setting, e.em))
    rows = []
    for _, r in s.iterrows():
        rows.append([r.setting.replace("kv_hqq_", "").replace("_r512", ""),
                     "—" if pd.isna(r.nbits) else int(r.nbits),
                     "—" if pd.isna(r.q_group_size) else int(r.q_group_size),
                     f(r.effective_bits, 2), f(r.em), f(r.f1),
                     f(ex.get(r.setting, float("nan"))),
                     "reference" if r.setting == base
                     else cell(mcs, f"fp16_vs_{r.setting}"),
                     "reference" if r.setting == base
                     else cell(mce, f"fp16ex_vs_{r.setting}")])
    return to_md(rows, ["setting", "nbits", "group", "eff. bits",
                        "EM short", "F1 short", "EM explain",
                        "short vs fp16", "explain vs fp16"])


def knob4():
    d = load("knob4_all_settings.csv", ("keep_ratio", "em", "f1",
                                        "parse_given_stopped"))
    mcs = mc_all("knob4_short_summary.json")
    mce = mc_all("knob4_explain_summary.json")
    sh = d[d["mode"] == "short"].set_index("setting")
    ex = d[d["mode"] == "explain"].set_index("setting")
    rows = []
    for setting, r in sh.sort_values(["policy", "keep_ratio"],
                                     ascending=[True, False]).iterrows():
        xr = ex.loc[setting] if setting in ex.index else None
        rows.append([setting, r.policy,
                     "—" if pd.isna(r.keep_ratio) else f(r.keep_ratio, 2),
                     f(r.em), f(r.f1),
                     "—" if xr is None else f(xr.em),
                     "—" if xr is None else f(xr.parse_given_stopped),
                     "reference" if r.policy == "none"
                     else cell(mcs, f"ref_vs_{setting}"),
                     "reference" if r.policy == "none"
                     else cell(mce, f"ref_vs_{setting}")])
    return to_md(rows, ["setting", "policy", "keep", "EM short", "F1 short",
                        "EM explain", "parse rate | stopped",
                        "short vs ref", "explain vs ref"])


def knob5():
    r1 = load("knob5_summary.csv", ("k", "complete_frac", "em", "f1",
                                    "prompt_tokens_mean"))
    r2 = load("knob5_r2_summary.csv", ("k", "batch", "complete_frac", "em",
                                       "f1", "prompt_tokens_mean"))
    m1, m2 = mc_all("knob5_summary.json"), mc_all("knob5_r2_summary.json")
    rows = []
    for _, r in r1.sort_values("k").iterrows():
        rows.append(["1", int(r.k), 16, f(r.complete_frac), f(r.em), f(r.f1),
                     f"{float(r.prompt_tokens_mean):.0f}",
                     "reference" if int(r.k) == 5
                     else cell(m1, f"5_vs_{int(r.k)}", int(r.k))])
    for _, r in r2.sort_values(["batch", "k"], ascending=[False, True]).iterrows():
        rows.append(["2", int(r.k), int(r.batch), f(r.complete_frac), f(r.em),
                     f(r.f1), f"{float(r.prompt_tokens_mean):.0f}",
                     "reference" if int(r.k) == 5 and int(r.batch) == 16
                     else cell(m2, r.setting, f"5_vs_{int(r.k)}")])
    return to_md(rows, ["regime", "k", "batch", "both gold", "EM", "F1",
                        "prompt tokens", "vs k=5"])


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-dir", default="results")
    ARGS = ap.parse_args()
    for name, fn in (("Knob 1 — search effort", knob1),
                     ("Knob 2 — embedding precision", knob2),
                     ("Knob 3 — KV precision", knob3),
                     ("Knob 4 — KV eviction", knob4),
                     ("Knob 5 — top-k", knob5)):
        print(f"\n### {name}\n")
        try:
            print(fn())
        except Exception as exc:
            print(f"FAILED: {type(exc).__name__}: {exc}")
