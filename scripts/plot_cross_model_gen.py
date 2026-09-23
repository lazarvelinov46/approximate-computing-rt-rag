#!/usr/bin/env python3
# scripts/plot_cross_model_gen.py — NEW FILE
"""Knobs 3, 4 and 5 across generators: per-model report figures, comparison
figures and tables. Regime 1, short mode, n=1000 (plus each model's explain
arm where it exists).

Per model  results/figures/knob{3,4,5}_curves_<tag>.png — the layout of the
           report figures, with explain panels replaced by short-mode ones
           for models with no explain arm.
Across     results/figures/knob{3,4,5}_cross_model.png
           (a) EM vs the knob axis, (b) retained EM or the mechanism view,
           (c) dEM(model) - dEM(reference), 95% CI.
Tables     results/cross_model/knob345_tables.md + one CSV per table.

These three knobs act inside the generator: retrieval is untouched, so
complete_frac is constant within a model for knobs 3 and 4 and varies only
with k for knob 5. Quantizer specs (effective bits, compression) are
properties of the configuration, not of the model, and are read once from
the reference model's knob3_summary.csv.

References are per setting and per model: knob 3 and knob 5 (k != 10) against
that model's own baseline; knob 4 position policies against evict_none_b1;
knob-4 attention against evict_none_b1_eager where it exists (Llama, whose
eager and sdpa outputs differ at batch 1); knob 5 at k=10 against
topk_05_b8 where the run used batch 8.

Bootstrap, colours and markdown helpers are shared with plot_cross_model.py.

Pure CPU. From the repo root:
    python scripts/plot_cross_model_gen.py --self-test
    python scripts/plot_cross_model_gen.py
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts"))

from src.config import results_dir                                     # noqa: E402
import src.metrics.quality as Q                                        # noqa: E402
import src.pipeline.rag as R                                           # noqa: E402
from plot_cross_model import (                                         # noqa: E402
    COLOUR, DISPLAY, FIG_DIR, N, N_BOOT, REF, SEED, SHORT, TAB_DIR, TAG2GEN,
    TOP_K, _ci, _ci_str, _hits, _load, _mcnemar, _md, delta_boot)

MNT = {"short": 32, "explain": 256}
BASE = {"short": ("baseline_1k_results.csv", "baseline"),
        "explain": ("baseline_explain_1k_results.csv", "baseline_explain")}
FILES = {3: ("knob3_short_results.csv", "knob3_explain_results.csv"),
         4: ("knob4_short_results.csv", "knob4_explain_results.csv"),
         5: ("knob5_results.csv", None)}
KNOB_NAME = {3: "KV-cache precision", 4: "KV-cache eviction", 5: "top-k"}
GROUP_STYLE = {32: ("group 32", "C0", "o"), 16: ("group 16", "C1", "s")}
POLICY = {"evict_recency": ("recency", "tab:red", "o"),
          "evict_sink4": ("sink + recent", "tab:orange", "s"),
          "evict_random": ("random (control)", "tab:gray", "^"),
          "evict_attn": ("attention (H2O/SnapKV)", "tab:blue", "D")}

# Published Qwen numbers (notebooks 12, 27, 16) — the self-test recomputes them.
QWEN_EM = {"kv_hqq_n8_g32_r512": 0.391, "kv_hqq_n4_g32_r512": 0.385,
           "kv_hqq_n3_g32_r512": 0.339, "kv_hqq_n2_g32_r512": 0.046,
           "kv_hqq_n2_g16_r512": 0.370, "evict_recency_r050": 0.161,
           "evict_sink4_r050": 0.282, "evict_random_r050_s42": 0.320,
           "evict_attn_r050": 0.337, "evict_random_r025_s42": 0.087,
           "evict_attn_r025": 0.299, "topk_02": 0.366, "topk_03": 0.379,
           "topk_10": 0.428}
QWEN_P = {"kv_hqq_n4_g32_r512": 0.2101, "kv_hqq_n2_g16_r512": 0.0025, "topk_10": 0.0014}
# Notebook 16's regime-1 projected curve, read off the published figure.
QWEN_K5_PROJ = {2: 0.292, 3: 0.342, 5: 0.390, 7: 0.412, 10: 0.441}


# --- setting metadata -------------------------------------------------------

def kv_label(s):
    return s.replace("kv_hqq_", "").replace("kv_quanto_", "quanto ").replace("_r512", "")


def knob3_spec():
    """Quantizer specs from the reference summary — configuration, not model."""
    df = pd.read_csv(REPO / "results" / "knob3_summary.csv")
    return {r["setting"]: r for _, r in df.iterrows()}


def parse_evict(s):
    m = re.match(r"(evict_(?:recency|sink4|random|attn))_r(\d{3})(?:_s\d+)?(_naive)?$", s)
    if not m:
        return None
    return {"policy": m.group(1), "policy_label": POLICY[m.group(1)][0],
            "keep_ratio": int(m.group(2)) / 100, "naive": bool(m.group(3))}


def parse_topk(s):
    m = re.match(r"topk_(\d+)(?:_b(\d+))?$", s)
    if not m:
        return None
    return {"k": int(m.group(1)), "batch": int(m.group(2) or 16)}


def describe(knob, setting, spec3):
    if knob == 3:
        r = spec3.get(setting)
        if r is None:
            return None
        return {"nbits": r["nbits"], "q_group_size": r["q_group_size"],
                "residual_length": r["residual_length"], "backend": r["backend"],
                "effective_bits": float(r["effective_bits"]),
                "compression": float(r["compression"]), "label": kv_label(setting)}
    if knob == 4:
        d = parse_evict(setting)
        if d is None:
            return None
        variant = " (no RoPE shift)" if d["naive"] else ""
        return {**d, "label": f"{d['policy_label']} r{d['keep_ratio']:.2f}{variant}"}
    d = parse_topk(setting)
    return None if d is None else {**d, "label": f"k={d['k']}"}


def reference_of(knob, setting, available):
    """(file, setting) of the reference this comparison uses, for this model."""
    if knob == 4:
        eager = "evict_none_b1_eager"
        ref = eager if setting.startswith("evict_attn") and eager in available else "evict_none_b1"
        return FILES[4][0], ref
    if knob == 5 and setting.endswith("_b8") and setting != "topk_05_b8":
        return FILES[5][0], "topk_05_b8"
    return BASE["short"]


def sort_key(knob):
    return {3: "effective_bits", 4: "keep_ratio", 5: "k"}[knob]


def match_key(knob, row):
    """What identifies the same comparison across models."""
    return row["k"] if knob == 5 else row["setting"]


# --- loading ----------------------------------------------------------------

def settings_of(tag, fname):
    path = results_dir(TAG2GEN[tag]) / fname
    if not path.exists():
        return []
    return sorted(R.load_results(str(path)).setting.unique())


def _stats(df, mode, k=TOP_K):
    # k must be the setting's own top-k: aggregate() truncates retrieved_ids to
    # k, so complete_frac / em_complete / em_incomplete are wrong for any knob-5
    # setting scored at the frozen 5.
    a = Q.aggregate(df.to_dict("records"), k=k)
    dec = pd.to_numeric(df["decode_tokens"], errors="coerce").astype(float)
    out = {k: a[k] for k in ("em", "f1", "complete_frac", "em_complete",
                             "em_incomplete", "abstain_rate")}
    out["decode_mean"] = round(float(dec.mean()), 2)
    out["trunc_frac"] = round(float((dec >= MNT[mode] - 1).mean()), 4)
    if mode == "explain":
        # parsed_ok is written by newer runs; older explain CSVs predate the
        # column, so fall back to re-parsing the stored generation.
        if "parsed_ok" in df.columns:
            ok = pd.to_numeric(df["parsed_ok"], errors="coerce")
        else:
            import src.generator.model as G
            ok = pd.Series([G.parse_answer(t)[1] for t in df["raw_generation"]], dtype=float)
        out["parse_rate"] = round(float(ok.mean()), 4)
    return out


def _fetch(m, tag, fname, setting, mode, k=TOP_K):
    """Load once per (mode, setting); return the cache key.

    Explain and short runs share setting labels, so a setting-only key made
    every explain lookup return the cached SHORT row — identical short/explain
    bars in knob 3 panel (d), and no parse_rate in knob 4 panel (d).
    """
    key = setting if mode == "short" else f"explain:{setting}"
    if key not in m["stats"]:
        df = _load(tag, fname, setting)
        assert df.qid.tolist() == m["qids"], f"{tag}:{setting} ({mode}): question set differs"
        m["hits"][key] = _hits(df)
        m["stats"][key] = _stats(df, mode, k)
        m["pred"][key] = dict(zip(df.qid, df.prediction))
    return key


def build_table(m, tag, knob, mode, spec3):
    fname = FILES[knob][0 if mode == "short" else 1]
    if fname is None or not (results_dir(TAG2GEN[tag]) / fname).exists():
        return None
    available = set(settings_of(tag, fname))
    rows = []
    for s in sorted(available):
        if s.startswith("evict_none") or s == "topk_05_b8":
            continue
        meta = describe(knob, s, spec3)
        if meta is None:
            continue
        k_s = _fetch(m, tag, fname, s, mode, meta.get("k", TOP_K))
        if mode == "short":
            rfile, rset = reference_of(knob, s, available)
        else:
            rfile, rset = BASE["explain"]
        k_r = _fetch(m, tag, rfile, rset, mode)
        ident = float(np.mean([m["pred"][k_r][q] == p
                               for q, p in m["pred"][k_s].items()]))
        st, rst = m["stats"][k_s], m["stats"][k_r]
        rows.append({"setting": s, **meta, "reference": rset, "ref_key": k_r, **st,
                     "identity_vs_ref": round(ident, 4),
                     "dEM": round(st["em"] - rst["em"], 4),
                     **_mcnemar(m["hits"][k_r], m["hits"][k_s])})
    if not rows:
        return None
    return pd.DataFrame(rows).sort_values(sort_key(knob)).reset_index(drop=True)


def load_model(tag, spec3):
    base = _load(tag, *BASE["short"])
    m = {"tag": tag, "qids": base.qid.tolist(), "hits": {}, "stats": {}, "pred": {},
         "tables": {}, "explain": {}}
    m["hits"]["baseline"] = _hits(base)
    m["stats"]["baseline"] = _stats(base, "short")
    m["pred"]["baseline"] = dict(zip(base.qid, base.prediction))
    for knob in FILES:
        t = build_table(m, tag, knob, "short", spec3)
        if t is not None:
            m["tables"][knob] = t
        e = build_table(m, tag, knob, "explain", spec3)
        if e is not None:
            m["explain"][knob] = e
    # regime-2 knob 5, if that arm exists (reference model only so far)
    r2 = results_dir(TAG2GEN[tag]) / "knob5_r2_results.csv"
    if r2.exists():
        rows = []
        b2 = _load(tag, "regime2_a1_baseline_results.csv", "regime2_a1_baseline")
        rows.append({"k": TOP_K, **_stats(b2, "short")})
        for s in settings_of(tag, "knob5_r2_results.csv"):
            d = parse_topk(s.replace("r2_", ""))
            if d:
                rows.append({"k": d["k"],
                             **_stats(_load(tag, "knob5_r2_results.csv", s), "short", d["k"])})
        m["regime2"] = pd.DataFrame(rows).drop_duplicates("k").sort_values("k")
    return m


# --- per-model figures ------------------------------------------------------

def _hqq_series(t):
    """(group size -> rows) for the main chains, and the flush probe separately."""
    main = t[(t["backend"] == "hqq") & (t["residual_length"] != 0)]
    return ({g: d.sort_values("effective_bits") for g, d in main.groupby("q_group_size")},
            t[t["residual_length"] == 0])


def fig_knob3(m, path):
    t, b = m["tables"][3], m["stats"]["baseline"]
    series, flush = _hqq_series(t)
    ex = m["explain"].get(3)
    panels = [("em", "exact match", "(a) EM vs storage cost"),
              ("f1", "F1", "(b) F1 vs storage cost"),
              ("abstain_rate", "abstain rate", "(c) abstention as the cache degrades")]
    fig, axes = plt.subplots(2, 2, figsize=(13.5, 9))
    ax = axes.ravel()

    for j, (col, ylab, title) in enumerate(panels):
        for g, d in sorted(series.items(), reverse=True):
            lab, colour, mk = GROUP_STYLE.get(int(g), (f"group {g}", "C2", "^"))
            ax[j].plot(d["effective_bits"], d[col], "-", marker=mk, ms=7, color=colour, label=lab)
            if j == 0:
                for _, r in d.iterrows():
                    ax[j].annotate(r["label"], (r["effective_bits"], r[col]), fontsize=7,
                                   color=colour, ha="center", textcoords="offset points",
                                   xytext=(0, -14))
        if not flush.empty:
            ax[j].plot(flush["effective_bits"], flush[col], "^", ms=9, color="tab:gray",
                       label="residual_length = 0 (flush probe)")
        ax[j].axhline(b[col], ls="--", c="k", lw=1, label=f"fp16 — {b[col]:.3f}")
        ax[j].scatter([16], [b[col]], marker="*", s=200, c="k", zorder=6)
        ax[j].set_xlabel("effective bits per KV element (measured)\nleft = more aggressive")
        ax[j].set_ylabel(ylab)
        ax[j].set_title(title)
        ax[j].grid(alpha=.3); ax[j].legend(fontsize=8, loc="best")

    if ex is not None:                      # (d) explain vs short, as in the report
        shared = [s for s in ex["setting"] if s in set(t["setting"])]
        labels = ["kv_fp16"] + [kv_label(s) for s in shared]
        xs = np.arange(len(labels))
        sh = [b["em"]] + [float(t.loc[t["setting"] == s, "em"].iloc[0]) for s in shared]
        exv = [m["stats"]["explain:baseline_explain"]["em"]] + \
              [float(ex.loc[ex["setting"] == s, "em"].iloc[0]) for s in shared]
        ax[3].bar(xs - 0.2, sh, 0.4, color="tab:gray", label="short")
        ax[3].bar(xs + 0.2, exv, 0.4, color="tab:blue", label="explain")
        ax[3].axhline(b["em"], ls="--", c="k", lw=1, label=f"fp16 short — {b['em']:.3f}")
        ax[3].axhline(exv[0], ls="--", c="tab:blue", lw=1, label=f"fp16 explain — {exv[0]:.3f}")
        ax[3].set_xticks(xs); ax[3].set_xticklabels(labels, fontsize=8)
        ax[3].set_ylabel("exact match")
        ax[3].set_title("(d) explain vs short, same settings")
    else:                                   # no explain arm: generator damage instead
        for g, d in sorted(series.items(), reverse=True):
            lab, colour, mk = GROUP_STYLE.get(int(g), (f"group {g}", "C2", "^"))
            ax[3].plot(d["effective_bits"], d["identity_vs_ref"], "-", marker=mk, ms=7,
                       color=colour, label=lab)
        ax[3].axhline(1.0, ls="--", c="k", lw=1, label="fp16 = 1.000 by definition")
        ax[3].scatter([16], [1.0], marker="*", s=200, c="k", zorder=6)
        ax[3].set_xlabel("effective bits per KV element (measured)\nleft = more aggressive")
        ax[3].set_ylabel("predictions identical to fp16")
        ax[3].set_title("(d) how much of the output changes at all")
    ax[3].grid(alpha=.3); ax[3].legend(fontsize=8, loc="best")

    fig.suptitle(f"Knob 3 — KV-cache precision · regime 1 · n=1000 · retrieval untouched "
                 f"(complete_frac {b['complete_frac']:.3f} at every setting) · {DISPLAY[m['tag']]}")
    fig.tight_layout()
    _require_drawn(fig, path)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _evict_panel(ax, t, col, ylab, title, ref_val, ref_label, ylim_zero=False):
    assert col in t.columns, f"{title}: no '{col}' column; have {list(t.columns)}"
    drawn = 0
    for pol, (lab, colour, mk) in POLICY.items():
        d = t[(t["policy"] == pol) & (~t["naive"].astype(bool))].sort_values("keep_ratio")
        if d.empty:
            continue
        ax.plot(d["keep_ratio"], d[col], "-", marker=mk, ms=7, color=colour, label=lab)
        drawn += 1
    assert drawn, (
        f"{title}: no policy matched.\n"
        f"  rows={len(t)}  columns={list(t.columns)}\n"
        f"  POLICY keys={list(POLICY)}\n"
        f"  policy values={t['policy'].value_counts().to_dict()}\n"
        f"  policy dtype={t['policy'].dtype}  naive dtype={t['naive'].dtype}  "
        f"naive values={t['naive'].tolist()}\n"
        f"  keep_ratio={t['keep_ratio'].tolist()}  {col}={t[col].tolist()}")
    nv = t[t["naive"]]
    if not nv.empty:
        ax.plot(nv["keep_ratio"], nv[col], "o", ms=14, mfc="none", mec="k",
                label="sink+recent, no RoPE shift (control)")
    ax.axhline(ref_val, ls="--", c="k", lw=1, label=f"{ref_label} — {ref_val:.3f}")
    ax.scatter([1.0], [ref_val], marker="*", s=200, c="k", zorder=6)
    ax.set_xlabel("keep_ratio (fraction of cache kept)\nleft = more aggressive")
    ax.set_ylabel(ylab)
    ax.set_title(title)
    if ylim_zero:
        ax.set_ylim(bottom=0)
    ax.grid(alpha=.3); ax.legend(fontsize=8, loc="best")


def fig_knob4(m, path):
    t = m["tables"][4]
    ex = m["explain"].get(4)
    ref = m["stats"].get("evict_none_b1", m["stats"]["baseline"])
    fig, axes = plt.subplots(2, 2, figsize=(13.5, 9))
    ax = axes.ravel()

    _evict_panel(ax[0], t, "em", "exact match", "(a) short: EM vs keep_ratio",
                 ref["em"], "no eviction")
    if ex is not None:                      # report layout
        exref = m["stats"]["explain:baseline_explain"]
        _evict_panel(ax[1], ex, "em", "exact match", "(b) explain: EM vs keep_ratio",
                     exref["em"], "no eviction")
        _evict_panel(ax[2], t, "f1", "F1", "(c) short: F1 vs keep_ratio", ref["f1"], "no eviction")
        _evict_panel(ax[3], ex, "parse_rate", "parse rate | stopped cleanly",
                     "(d) explain: format adherence", 1.0, "no eviction")
    else:                                   # short-only: is the answer wrong, or unformatted?
        _evict_panel(ax[1], t, "f1", "F1", "(b) short: F1 vs keep_ratio", ref["f1"], "no eviction")
        _evict_panel(ax[2], t, "trunc_frac", "fraction hitting the token cap",
                     "(c) short: answers that never stop", ref["trunc_frac"],
                     "no eviction", ylim_zero=True)
        _evict_panel(ax[3], t, "identity_vs_ref", "predictions identical to reference",
                     "(d) short: how much of the output changes", 1.0, "no eviction")

    fig.suptitle(f"Knob 4 — KV-cache eviction · regime 1 · batch_size 1 · n=1000 · "
                 f"retrieval untouched · {DISPLAY[m['tag']]}")
    fig.tight_layout()
    _require_drawn(fig, path)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _knob5_curve(m):
    """Regime-1 points including the k=5 baseline, with the projection."""
    t, b = m["tables"][5], m["stats"]["baseline"]
    rows = [{"k": TOP_K, "setting": "baseline", "batch": 16, **b}]
    rows += [{c: r[c] for c in r.index} for _, r in t.iterrows()]
    d = pd.DataFrame(rows).drop_duplicates("k").sort_values("k")
    d["projected"] = (d["complete_frac"] * b["em_complete"]
                      + (1 - d["complete_frac"]) * b["em_incomplete"])
    return d


def fig_knob5(m, path):
    d, b = _knob5_curve(m), m["stats"]["baseline"]
    fig, ax = plt.subplots(1, 3, figsize=(15, 4.2))

    ax[0].plot(d["k"], d["em"], "o-", label="regime 1")
    ax[0].plot(d["k"], d["projected"], "o--", alpha=.6, label="regime 1 projected")
    if "regime2" in m:
        r2 = m["regime2"]
        ax[0].plot(r2["k"], r2["em"], "s-", color="tab:green", label="regime 2")
    ax[0].axvline(TOP_K, ls=":", c="k", lw=1)
    ax[0].scatter([TOP_K], [b["em"]], marker="*", s=180, c="k", zorder=5)
    b8 = m["stats"].get("topk_05_b8")
    if b8:
        ax[0].plot([TOP_K], [b8["em"]], "o", ms=9, mfc="none", mec="k",
                   label="k=5, batch 8 (control)")
    ax[0].set_xlabel("top-k"); ax[0].set_ylabel("EM")
    ax[0].set_title("EM vs top-k")
    ax[0].grid(alpha=.3); ax[0].legend(fontsize=8)

    ax[1].plot(d["k"], d["em_complete"], "o-", label="evidence complete")
    ax[1].plot(d["k"], d["em_incomplete"], "s-", label="evidence incomplete")
    ax[1].set_xlabel("top-k"); ax[1].set_ylabel("EM within group")
    ax[1].set_title("Conditional EM: distractor load")
    ax[1].grid(alpha=.3); ax[1].legend(fontsize=8)

    ax[2].plot(d["k"], d["complete_frac"], "o-", color="tab:green")
    ax[2].set_xlabel("top-k"); ax[2].set_ylabel("complete_frac")
    ax[2].set_title("Evidence delivered")
    ax[2].grid(alpha=.3)

    fig.suptitle(f"Knob 5 — top-k · regime 1 · short mode · n=1000 · {DISPLAY[m['tag']]}")
    fig.tight_layout()
    _require_drawn(fig, path)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


# --- cross-model ------------------------------------------------------------

def cross_table(knob, models):
    ref = models[REF]
    if knob not in ref["tables"]:
        return None
    keyed = {t: {match_key(knob, r): r for _, r in m["tables"][knob].iterrows()}
             for t, m in models.items()}
    shared = [match_key(knob, r) for _, r in ref["tables"][knob].iterrows()
              if all(match_key(knob, r) in keyed[t] for t in models)]
    rows = []
    for key in shared:
        rr = keyed[REF][key]
        dR, rhoR = delta_boot(ref["hits"][rr["setting"]], ref["hits"][rr["ref_key"]])
        row = {"key": str(key), "label": rr["label"]}
        for tag, m in models.items():
            r = keyed[tag][key]
            d, rho = delta_boot(m["hits"][r["setting"]], m["hits"][r["ref_key"]])
            row[f"{tag}_setting"] = r["setting"]
            row[f"{tag}_reference"] = r["reference"]
            row[f"{tag}_em"] = r["em"]
            row[f"{tag}_dEM"] = r["dEM"]
            row[f"{tag}_dEM_lo"], row[f"{tag}_dEM_hi"] = _ci(d)
            row[f"{tag}_retained"] = round(r["em"] / m["stats"][r["ref_key"]]["em"], 4)
            row[f"{tag}_verdict"] = r["verdict"]
            if tag != REF:
                row[f"{tag}_ddEM"] = round(r["dEM"] - rr["dEM"], 4)
                row[f"{tag}_ddEM_lo"], row[f"{tag}_ddEM_hi"] = _ci(d - dR)
                row[f"{tag}_ddrel"] = round(
                    r["em"] / m["stats"][r["ref_key"]]["em"]
                    - rr["em"] / ref["stats"][rr["ref_key"]]["em"], 4)
                row[f"{tag}_ddrel_lo"], row[f"{tag}_ddrel_hi"] = _ci(rho - rhoR)
        rows.append(row)
    return pd.DataFrame(rows)


def _panel_dd(ax, cross, models, xlabel):
    others = [t for t in models if t != REF]
    xpos = np.arange(len(cross))
    for j, tag in enumerate(others):
        off = (j - (len(others) - 1) / 2) * 0.15
        y = cross[f"{tag}_ddEM"].to_numpy()
        err = np.clip([y - cross[f"{tag}_ddEM_lo"].to_numpy(),
                       cross[f"{tag}_ddEM_hi"].to_numpy() - y], 0, None)
        ax.errorbar(xpos + off, y, yerr=err, fmt="o", ms=6, color=COLOUR[tag],
                    capsize=4, lw=1, label=f"{SHORT[tag]} − {SHORT[REF]}")
    ax.axhline(0, c="k", lw=.8)
    ax.set_xticks(xpos)
    ax.set_xticklabels(list(cross["label"]), fontsize=8, rotation=20, ha="right")
    ax.set_xlabel(xlabel)
    ax.set_ylabel("ΔEM(model) − ΔEM(reference)")
    ax.set_title("(c) difference in degradation  (95% CI)")
    ax.grid(alpha=.3); ax.legend(fontsize=8)

def _require_drawn(fig, path):
    """An empty panel is a bug, not an output: never write one to disk.
    Legend proxies (plot([], [])) carry no data and must not count."""
    for i, a in enumerate(fig.axes):
        drawn = sum(1 for ln in a.lines if len(ln.get_xdata()))
        drawn += len(a.patches) + len(a.collections)
        if drawn == 0:
            raise RuntimeError(f"{path.name}: panel {i} ({a.get_title()!r}) drew "
                               f"nothing with data")

def fig_cross3(models, cross, path):
    fig, ax = plt.subplots(1, 3, figsize=(15, 4.2))
    for tag, m in models.items():
        col, t = COLOUR[tag], m["tables"][3]
        t = t[t["setting"].isin(cross[f"{tag}_setting"])]
        b = m["stats"]["baseline"]["em"]
        first = True
        for g, d in sorted(t.groupby("q_group_size"), reverse=True):
            # One line across group sizes crosses the G16 point between the G32
            # chain's points and draws a dip that is an artefact of the x-tie
            # (n3_g32 and n4_g32 both sit at 5.00 effective bits), not data.
            d = d.sort_values(["effective_bits", "nbits"])
            mk = "o" if int(g) == 32 else "s"
            ax[0].plot(d["effective_bits"], d["em"], "-", marker=mk, ms=6, color=col,
                       label=SHORT[tag] if first else None)
            ax[1].plot(d["effective_bits"], d["em"] / b, "-", marker=mk, ms=6, color=col,
                       label=SHORT[tag] if first else None)
            first = False
        ax[0].axhline(b, ls="--", c=col, lw=1)
        ax[0].scatter([16], [b], marker="*", s=180, color=col, zorder=5)
    ax[0].set_xlabel("effective bits per KV element\nleft = more aggressive")
    ax[0].set_ylabel("EM"); ax[0].set_title("(a) EM vs storage cost  (star: fp16)")
    ax[1].axhline(1.0, ls="--", c="k", lw=1)
    ax[1].set_xlabel("effective bits per KV element\nleft = more aggressive")
    ax[1].set_ylabel("EM retained (setting / own fp16)")
    ax[1].set_title("(b) retained quality, baselines normalized")
    for a in ax[:2]:
        a.grid(alpha=.3)
        models_leg = a.legend(fontsize=8, loc="lower right")
        a.add_artist(models_leg)
        a.legend(handles=[plt.Line2D([], [], color="k", marker=mk, ls="-", ms=6, label=lab)
                          for lab, mk in (("group 32", "o"), ("group 16", "s"))],
                 fontsize=8, loc="center right")
    _panel_dd(ax[2], cross, models, "setting (effective bits ascending)")
    fig.suptitle("Knob 3 — KV-cache precision across generators · regime 1 · short mode · n=1000")
    fig.tight_layout();
    _require_drawn(fig, path)
    fig.savefig(path, dpi=150, bbox_inches="tight"); plt.close(fig)


def fig_cross4(models, cross, path):
    fig, ax = plt.subplots(1, 3, figsize=(15, 4.2))
    for tag, m in models.items():
        t = m["tables"][4]
        t = t[t["setting"].isin(cross[f"{tag}_setting"])]
        ls = "-" if tag == REF else "--"
        b = m["stats"].get("evict_none_b1", m["stats"]["baseline"])["em"]
        fill = "full" if tag == REF else "none"
        for pol, (lab, colour, mk) in POLICY.items():
            d = t[t["policy"] == pol].sort_values("keep_ratio")
            if d.empty:
                continue
            ax[0].plot(d["keep_ratio"], d["em"], ls, marker=mk, ms=7, color=colour,
                       fillstyle=fill, mew=1.4)
            ax[1].plot(d["keep_ratio"], d["em"] / b, ls, marker=mk, ms=7, color=colour,
                       fillstyle=fill, mew=1.4)
        ax[0].plot([], [], ls, color="k", marker="o", fillstyle=fill, mew=1.4,
                   label=f"{SHORT[tag]} ({'filled' if tag == REF else 'hollow'})")
        ax[0].axhline(b, ls=":", c="k", lw=.8)
    for pol, (lab, colour, mk) in POLICY.items():
        ax[0].plot([], [], "-", marker=mk, color=colour, label=lab)
    ax[0].set_xlabel("keep_ratio\nleft = more aggressive"); ax[0].set_ylabel("EM")
    ax[0].set_title("(a) EM vs keep_ratio  (solid: reference model)")
    ax[1].axhline(1.0, ls="--", c="k", lw=1)
    ax[1].set_xlabel("keep_ratio\nleft = more aggressive")
    ax[1].set_ylabel("EM retained (setting / own no-eviction)")
    ax[1].set_title("(b) retained quality, baselines normalized")
    for a in ax[:2]:
        a.grid(alpha=.3)
    ax[0].legend(fontsize=7)
    _panel_dd(ax[2], cross, models, "policy and keep_ratio")
    fig.suptitle("Knob 4 — KV-cache eviction across generators · regime 1 · batch_size 1 · n=1000")
    fig.tight_layout();
    _require_drawn(fig, path)
    fig.savefig(path, dpi=150, bbox_inches="tight"); plt.close(fig)


def fig_cross5(models, cross, path):
    fig, ax = plt.subplots(1, 3, figsize=(15, 4.2))
    cf0 = models[REF]["stats"]["baseline"]["complete_frac"]
    for tag, m in models.items():
        col, d = COLOUR[tag], _knob5_curve(m)
        b = m["stats"]["baseline"]
        ax[0].plot(d["k"], d["em"], "-o", ms=6, color=col, label=SHORT[tag])
        ax[0].plot(d["k"], d["projected"], "--", lw=1, color=col,
                   label=f"{SHORT[tag]} · projection")
        ax[0].scatter([TOP_K], [b["em"]], marker="*", s=180, color=col, zorder=5)
        gap = b["em_complete"] - b["em_incomplete"]
        sel = d["k"] != TOP_K
        ax[1].plot(d.loc[sel, "complete_frac"], d.loc[sel, "em"] - b["em"], "o", ms=6, color=col,
                   label=SHORT[tag])
        xs = np.linspace(d["complete_frac"].min() - 0.02, d["complete_frac"].max() + 0.02, 50)
        ax[1].plot(xs, (xs - cf0) * gap, "--", lw=1, color=col,
                   label=f"{SHORT[tag]} · projection")
    ax[0].axvline(TOP_K, ls=":", c="k", lw=1)
    ax[0].set_xlabel("top-k"); ax[0].set_ylabel("EM")
    ax[0].set_title("(a) EM vs top-k  (star: frozen k=5)")
    ax[1].axhline(0, c="k", lw=.8)
    ax[1].scatter([cf0], [0], marker="*", s=180, c="k", zorder=5)
    ax[1].set_xlabel("complete_frac (evidence delivered)")
    ax[1].set_ylabel("ΔEM vs own k=5 baseline")
    ax[1].set_title("(b) change vs evidence  (dashed: projection)")
    for a in ax[:2]:
        a.grid(alpha=.3); a.legend(fontsize=7)
    _panel_dd(ax[2], cross, models, "top-k")
    fig.suptitle("Knob 5 — top-k across generators · regime 1 · short mode · n=1000")
    fig.tight_layout();
    _require_drawn(fig, path)
    fig.savefig(path, dpi=150, bbox_inches="tight"); plt.close(fig)


# --- tables -----------------------------------------------------------------

COLS = {
    3: (["label", "effective_bits", "compression", "identity_vs_ref", "em", "f1",
         "abstain_rate", "dEM", "b", "c", "p", "verdict"],
        ["setting", "eff. bits", "compression", "identical to fp16", "EM", "F1",
         "abstain", "ΔEM", "b", "c", "p", "verdict"]),
    4: (["policy_label", "keep_ratio", "em", "f1", "identity_vs_ref", "trunc_frac",
         "dEM", "b", "c", "p", "verdict", "reference"],
        ["policy", "keep_ratio", "EM", "F1", "identical to ref", "hits token cap",
         "ΔEM", "b", "c", "p", "verdict", "reference"]),
    5: (["k", "complete_frac", "em", "f1", "em_complete", "em_incomplete",
         "dEM", "b", "c", "p", "verdict", "reference"],
        ["top-k", "complete_frac", "EM", "F1", "EM complete", "EM incomplete",
         "ΔEM", "b", "c", "p", "verdict", "reference"]),
}
FMT = {"effective_bits": "{:.2f}", "compression": "{:.2f}×", "keep_ratio": "{:.2f}",
       "identity_vs_ref": "{:.3f}", "trunc_frac": "{:.3f}", "complete_frac": "{:.3f}",
       "em": "{:.3f}", "f1": "{:.3f}", "em_complete": "{:.3f}", "em_incomplete": "{:.3f}",
       "abstain_rate": "{:.3f}", "dEM": "{:+.3f}", "p": "{:.4f}"}


def model_md(knob, m):
    cols, head = COLS[knob]
    rows = []
    for _, r in m["tables"][knob].iterrows():
        rows.append(["—" if pd.isna(r[c]) else (FMT[c].format(r[c]) if c in FMT else r[c])
                     for c in cols])
    return _md(rows, head)


def cross_md(cross, models):
    others = [t for t in models if t != REF]
    head = ["setting"]
    for t in models:
        head += [f"EM {SHORT[t]}", f"ΔEM {SHORT[t]} [95% CI]", f"retained {SHORT[t]}"]
    for t in others:
        head += [f"ΔΔ abs {SHORT[t]} − {SHORT[REF]} [95% CI]", "ΔΔ rel [95% CI]"]
    head += [f"verdict {SHORT[t]}" for t in models]
    rows = []
    for _, r in cross.iterrows():
        row = [r["label"]]
        for t in models:
            row += [f"{r[f'{t}_em']:.3f}",
                    _ci_str(r[f"{t}_dEM"], r[f"{t}_dEM_lo"], r[f"{t}_dEM_hi"]),
                    f"{r[f'{t}_retained']:.3f}"]
        for t in others:
            row += [_ci_str(r[f"{t}_ddEM"], r[f"{t}_ddEM_lo"], r[f"{t}_ddEM_hi"]),
                    _ci_str(r[f"{t}_ddrel"], r[f"{t}_ddrel_lo"], r[f"{t}_ddrel_hi"])]
        row += [r[f"{t}_verdict"] for t in models]
        rows.append(row)
    return _md(rows, head)


def write_tables(models, crosses):
    md = ["# Knobs 3, 4 and 5 across generators", "",
          "Regime 1, short mode, n=1000. Retrieval is untouched by these knobs. ΔEM is "
          "against each setting's own reference, which is listed per row: the model's "
          "baseline for knobs 3 and 5, evict_none_b1 for knob-4 position policies, "
          "evict_none_b1_eager for knob-4 attention where that run exists, and "
          "topk_05_b8 for k=10 where the run used batch 8. McNemar b = setting-only "
          f"correct, c = reference-only correct. CIs: paired bootstrap, B={N_BOOT:,}, "
          f"seed {SEED}.", ""]
    for knob in sorted(FILES):
        for tag, m in models.items():
            if knob not in m["tables"]:
                continue
            md += [f"## Knob {knob} — {KNOB_NAME[knob]} — {DISPLAY[tag]}", "",
                   model_md(knob, m), ""]
            m["tables"][knob].to_csv(TAB_DIR / f"knob{knob}_{tag}.csv", index=False)
        if crosses.get(knob) is not None:
            md += [f"## Knob {knob} — compared with {DISPLAY[REF]}", "",
                   cross_md(crosses[knob], models), ""]
            crosses[knob].to_csv(TAB_DIR / f"knob{knob}_cross.csv", index=False)
    path = TAB_DIR / "knob345_tables.md"
    path.write_text("\n".join(md) + "\n", encoding="utf-8")
    return path


# --- self-test --------------------------------------------------------------

def self_test() -> int:
    fails, n = [], 0
    spec3 = knob3_spec()
    q = load_model(REF, spec3)
    for knob, t in q["tables"].items():
        for _, r in t.iterrows():
            s = r["setting"]
            if s in QWEN_EM:
                n += 1
                if abs(r["em"] - QWEN_EM[s]) > 5e-4:
                    fails.append(f"{s}: EM {r['em']} vs published {QWEN_EM[s]}")
            if s in QWEN_P:
                n += 1
                if abs(r["p"] - QWEN_P[s]) > 5e-5:
                    fails.append(f"{s}: p {r['p']} vs published {QWEN_P[s]}")
            n += 1
            if r["reference"] not in q["hits"]:
                fails.append(f"{s}: reference {r['reference']} never loaded")
    # every knob-4 setting must parse into a policy and keep_ratio
    n += 1
    raw = [s for s in settings_of(REF, FILES[4][0]) if not s.startswith("evict_none")]
    if len(raw) != len(q["tables"][4]):
        missed = set(raw) - set(q["tables"][4]["setting"])
        fails.append(f"knob-4 settings not parsed: {sorted(missed)}")
    # the reference rule must pick eager only where that run exists
    n += 1
    if any(r["reference"] != "evict_none_b1" for _, r in q["tables"][4].iterrows()):
        fails.append(f"{REF} has no eager reference run, but one was selected")

    d = _knob5_curve(q).set_index("k")
    for k, want in QWEN_K5_PROJ.items():
        if k not in d.index:
            continue
        n += 1
        got = float(d.loc[k, "projected"])
        if abs(got - want) > 0.002:
            fails.append(f"knob-5 projection at k={k}: {got:.3f}, notebook 16 has {want:.3f}")

    n += 1
    t4 = q["tables"][4]
    per_pol = {p: int((t4["policy"] == p).sum()) for p in POLICY}
    if min(per_pol.values()) == 0 or not np.isfinite(t4["em"]).all():
        fails.append(f"knob-4 table not plottable: rows per policy {per_pol}")

    for f in fails:
        print(f"FAIL  {f}")
    print(f"self-test: {n - len(fails)}/{n} checks passed")
    return 1 if fails else 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--models", nargs="+")
    ap.add_argument("--self-test", action="store_true")
    ap.add_argument("--per-model", action="store_true",
                    help="also regenerate the reference model's own figures")
    ap.add_argument("--dump", action="store_true",
                    help="print every per-model table and exit")
    args = ap.parse_args()

    rc = self_test()
    if rc or args.self_test:
        return rc

    spec3 = knob3_spec()
    have = lambda t: (results_dir(TAG2GEN[t]) / FILES[3][0]).exists()
    tags = args.models or [t for t in TAG2GEN if have(t)]
    tags = [REF] + [t for t in tags if t != REF]
    models = {t: load_model(t, spec3) for t in tags}

    if args.dump:
        pd.set_option("display.width", 200, "display.max_columns", 40)
        for tag, m in models.items():
            for knob, t in sorted(m["tables"].items()):
                print(f"\n=== {tag} knob {knob} — short ({len(t)} rows) ===\n{t}")
            for knob, t in sorted(m["explain"].items()):
                print(f"\n=== {tag} knob {knob} — explain ({len(t)} rows) ===\n{t}")
        return 0

    FIG_DIR.mkdir(parents=True, exist_ok=True)
    TAB_DIR.mkdir(parents=True, exist_ok=True)
    for tag, m in ((t, mm) for t, mm in models.items() if t != REF or args.per_model):
        for knob, fn in ((3, fig_knob3), (4, fig_knob4), (5, fig_knob5)):
            if knob in m["tables"]:
                path = FIG_DIR / f"knob{knob}_curves_{tag}.png"
                fn(m, path)
                print(f"wrote {path.relative_to(REPO)}")

    crosses = {}
    if len(models) > 1:
        for knob, fn in ((3, fig_cross3), (4, fig_cross4), (5, fig_cross5)):
            crosses[knob] = cross_table(knob, models)
            if crosses[knob] is not None and len(crosses[knob]):
                path = FIG_DIR / f"knob{knob}_cross_model.png"
                fn(models, crosses[knob], path)
                print(f"wrote {path.relative_to(REPO)}")

    path = write_tables(models, crosses)
    print(f"wrote {path.relative_to(REPO)} (+ CSVs in {TAB_DIR.relative_to(REPO)}/)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
