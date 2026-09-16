# scripts/plot_phase2.py — NEW FILE (complete, step 1 of 2: preflight + verdict)
#!/usr/bin/env python3
"""Phase 2 figures: quality vs aggressiveness per knob, plus a cross-knob panel.

Pure CPU. Reads results/*_summary.{json,csv}; no model, no GPU, no Kaggle.

Step 1 (this version) validates inputs and fixes the significance verdict
logic. Figure generation is added on top once --check reports clean.

Usage:
    python scripts/plot_phase2.py --check      # inventory + schemas
    python scripts/plot_phase2.py --self-test  # verdict logic only
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import dataclass

import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SIGNIFICANT = "SIGNIFICANT"
NULL = "NULL"
NULL_ZERO = "NULL (zero-discordant carve-out)"
UNDERPOWERED = "UNDERPOWERED"
UNINFORMATIVE = "UNINFORMATIVE"


def verdict(b: int, c: int, p: float) -> str:
    """Frozen acceptance criterion, in the order the criterion is written.

    p decides first. Then the zero-discordant carve-out (knob 5 amendment,
    inherited from knob 3's 8-bit case) — it MUST be tested before the
    d >= 30 rule, because d == 0 fails that rule and would otherwise print
    UNINFORMATIVE for a case the study calls a NULL. That ordering bug is
    recorded in knob 4's corrections; the self-test below pins it.

    UNDERPOWERED (10 <= d < 30, p > 0.05) is NOT in the frozen criteria —
    the criteria define d >= 30 and d < 10 and leave the band undefined.
    Notebooks 12 and 28 resolved it differently. Named explicitly here
    rather than folded into either neighbour.
    """
    d = b + c
    if p <= 0.05:
        return SIGNIFICANT
    if d == 0 and b == c:
        return NULL_ZERO
    if d >= 30:
        return NULL
    if d < 10:
        return UNINFORMATIVE
    return UNDERPOWERED


def _verdict_no_carveout(b: int, c: int, p: float) -> str:
    """The real regression, kept as a fixture.

    Knob 4's first verdict helper had NO carve-out branch: d == 0 failed the
    d >= 30 test and fell through to UNINFORMATIVE. Note that placing the
    carve-out after d >= 30 (the first version of this fixture) is harmless —
    d == 0 fails d >= 30 and reaches the carve-out anyway. That fixture was
    not a mutant, and the disagreement check below correctly rejected it.
    """
    d = b + c
    if p <= 0.05:
        return SIGNIFICANT
    if d >= 30:
        return NULL
    return UNINFORMATIVE

# --- paired-test plumbing, shared by every figure ---------------------------

def mc_index(obj: dict) -> dict:
    """Every paired test in a summary JSON, keyed by comparison name.

    Knobs 1-4 store a dict under "mcnemar". Knob 5 stores LISTS under
    "mcnemar_vs_k5", "mcnemar_batch16_vs_k5", "mcnemar_batch_control" and
    "mcnemar_k20_vs_matched_k5"; those records name themselves through
    "a"/"b_setting", so they key as "<a>_vs_<b_setting>" (regime 1 uses ints,
    regime 2 uses setting strings).

    Records without b/c/p are DROPPED. knob 3's "hqq_vs_quanto" is a status
    note about the sm_75 failure, not a test, and would otherwise be read as
    a comparison with missing fields.
    """
    out = {}
    for key, block in obj.items():
        if not key.startswith("mcnemar"):
            continue
        if isinstance(block, dict):
            items = list(block.items())
        elif isinstance(block, list):
            items = [(f"{r.get('a')}_vs_{r.get('b_setting')}", r)
                     for r in block if isinstance(r, dict)]
        else:
            continue
        for name, rec in items:
            if isinstance(rec, dict) and {"b", "c", "p"} <= set(rec):
                out[name] = rec
    return out


def verdict_of(rec) -> str | None:
    if not rec:
        return None
    return verdict(int(rec["b"]), int(rec["c"]), float(rec["p"]))


def tag_of(rec) -> str:
    """Short ASCII label drawn next to a tested point. No unicode: the glyphs
    must exist in whatever font matplotlib falls back to on this machine."""
    if not rec:
        return ""
    d = int(rec["b"]) + int(rec["c"])
    v = verdict_of(rec)
    suffix = {SIGNIFICANT: "*", NULL: " ns", NULL_ZERO: " ns",
              UNDERPOWERED: " ns?", UNINFORMATIVE: " ?"}[v]
    return f"d={d}{suffix}"


#                      fillstyle,  markersize, alpha
POINT_STYLE = {
    SIGNIFICANT:   ("full",   9, 1.00),
    NULL:          ("none",   9, 1.00),
    NULL_ZERO:     ("none",   9, 1.00),
    UNDERPOWERED:  ("bottom", 9, 1.00),
    UNINFORMATIVE: ("none",   9, 1.00),
    None:          ("none",   9, 0.45),   # no paired test exists
}

MARKER_LEGEND = ("markers:  d=N* significant (p<=0.05)  |  d=N ns null (d>=30)  "
                 "|  d=N ns? underpowered (10<=d<30)  |  d=N ? uninformative "
                 "(d<10)  |  small unlabelled point = no paired test exists")

C_A, C_B = "tab:blue", "tab:orange"


def draw_point(ax, x, y, marker, colour, rec, manifest_row=None,
               manifest=None, tag_offset=(0, 12), tag_ha="center",
               show_tag=True) -> None:
    fill, ms, alpha = POINT_STYLE[verdict_of(rec)]
    ax.plot([x], [y], marker=marker, color=colour, linestyle="none",
            fillstyle=fill, markersize=ms, alpha=alpha,
            markeredgewidth=1.2, zorder=5)
    if rec and show_tag:
        ax.annotate(tag_of(rec), (x, y), textcoords="offset points",
                    xytext=tag_offset, ha=tag_ha, fontsize=7.5, color=colour,
                    zorder=7,
                    bbox=dict(boxstyle="round,pad=0.18", fc="white",
                              ec="none", alpha=0.8))
    if manifest is not None and manifest_row is not None:
        manifest.append({**manifest_row,
                         "verdict": verdict_of(rec) or "no paired test",
                         "discordant": (int(rec["b"]) + int(rec["c"]))
                                       if rec else None,
                         "p": float(rec["p"]) if rec else None})

# --- knob 1 ----------------------------------------------------------------

# setting -> comparison key. Knob 1's keys use short index names, not setting
# names, so this cannot be derived by prefix-stripping the way knobs 2-4 can.
# Only three of the seven approximate settings were ever tested against exact;
# ef16, ef8, nprobe48 and nprobe6 have NO paired test and are drawn as such.
KNOB1_MC = {
    "knob1_hnsw_ef32":    "exact_vs_hnsw_ef32",
    "knob1_hnsw_ef5":     "exact_vs_hnsw_ef5",
    "knob1_ivf_nprobe1":  "exact_vs_ivf_nprobe1",
}


def fig_knob1(results_dir: str, out_path: str, manifest: list) -> None:
    """Regime 1 ... no: regime 2, short mode. Quality vs search effort.

    2x2: quality on top (EM, F1), retrieval below (complete_frac, ann_recall).
    F1 gets its own panel rather than sharing EM's, because its range is
    roughly twice EM's and sharing flattens both.

    No significance encoding: McNemar is reported in the text.
    """
    df = pd.read_csv(os.path.join(results_dir, "knob1_summary.csv"))
    base = df[df["index"] == "flat"].iloc[0]
    fam = (("hnsw", "HNSW (ef)", C_A, "o"),
           ("ivf", "IVF (nprobe)", C_B, "s"))

    panels = (("em", "exact match", "(a) EM vs search effort"),
              ("f1", "F1", "(b) F1 vs search effort"),
              ("complete_frac", "complete_frac",
               "(c) both gold passages retrieved"),
              ("ann_recall", "ANN recall",
               "(d) agreement with exact top-5"))

    fig, axes = plt.subplots(2, 2, figsize=(11.5, 8.2))
    ax = axes.ravel()

    for family, label, colour, marker in fam:
        d = df[df["index"] == family].sort_values("ndis")
        for j, (col, _, _) in enumerate(panels):
            ax[j].plot(d.ndis, d[col], "-", marker=marker, ms=7,
                       color=colour, lw=1.4, label=label)
        for _, r in d.iterrows():
            manifest.append({"figure": "knob1_search_effort", "knob": 1,
                             "regime": 2, "setting": r.setting,
                             "x_axis": "ndis", "x": float(r.ndis),
                             "em": float(r.em), "f1": float(r.f1),
                             "complete_frac": float(r.complete_frac),
                             "ann_recall": float(r.ann_recall)})

    for j, (col, ylab, title) in enumerate(panels):
        # ann_recall is agreement with exact search, so exact is 1.0 by
        # construction — a baseline line there would be a tautology, not a
        # measurement. Every other panel gets the measured fp32 baseline.
        if col == "ann_recall":
            ax[j].axhline(1.0, ls="--", c="k", lw=1,
                          label="exact (Flat) = 1.0 by definition")
        else:
            yb = float(base[col])
            ax[j].axhline(yb, ls="--", c="k", lw=1,
                          label=f"exact (Flat) — {yb:.3f}")
            ax[j].scatter([float(base.ndis)], [yb], marker="*", s=200, c="k",
                          zorder=6)
        ax[j].set_xscale("log")
        ax[j].set_xlim(55, 1.8e5)
        ax[j].set_xlabel("distance computations per query (log)\n"
                         "left = more aggressive")
        ax[j].set_ylabel(ylab)
        ax[j].set_title(title)
        ax[j].grid(alpha=.3)
        ax[j].legend(fontsize=8, loc="lower right")

    fig.suptitle("Knob 1 — retrieval search effort · regime 2 (A1, 66,581 "
                 "passages) · short mode · n=1000", fontsize=11)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

def fig_knob1_params(results_dir: str, out_path: str, manifest: list) -> None:
    """The same data on the axes you configure.

    ef and nprobe get separate panels: they are not commensurable, so a
    shared x-axis would invite a comparison the numbers do not support.
    """
    df = pd.read_csv(os.path.join(results_dir, "knob1_summary.csv"))
    base = df[df["index"] == "flat"].iloc[0]
    fam = (("hnsw", "HNSW", "ef", C_A, "o"),
           ("ivf", "IVF", "nprobe", C_B, "s"))
    fig, ax = plt.subplots(1, 3, figsize=(14, 4.4))

    for j, (family, label, pname, colour, marker) in enumerate(fam):
        d = df[df["index"] == family].sort_values("value")
        ax[j].plot(d.value, d.em, "-", marker=marker, ms=7, color=colour,
                   lw=1.4, label=label)
        ax[j].axhline(base.em, ls="--", c="k", lw=1,
                      label=f"exact (Flat) — EM {base.em:.3f}")
        ax[j].set_xscale("log")
        ax[j].set_xticks(list(d.value))
        ax[j].get_xaxis().set_major_formatter(
            matplotlib.ticker.ScalarFormatter())
        ax[j].minorticks_off()
        ax[j].set_xlim(d.value.min() * 0.7, d.value.max() * 1.45)
        ax[j].set_ylim(df.em.min() - 0.02, df.em.max() + 0.03)
        ax[j].set_xlabel(f"{pname} (log) — left = more aggressive")
        ax[j].set_ylabel("exact match")
        ax[j].set_title(f"({'ab'[j]}) {label}: quality vs {pname}")

        ax[2].plot(d.value, d.ndis, "-", marker=marker, ms=7, color=colour,
                   lw=1.4, label=f"{label} ({pname})")
        for _, r in d.iterrows():
            manifest.append({"figure": "knob1_parameters", "knob": 1,
                             "regime": 2, "setting": r.setting,
                             "x_axis": pname, "x": float(r.value),
                             "em": float(r.em), "ndis": float(r.ndis)})

    ax[2].axhline(base.ndis, ls="--", c="k", lw=1,
                  label=f"exact = {int(base.ndis):,} per query")
    ax[2].set_xscale("log")
    ax[2].set_yscale("log")
    ax[2].set_xlabel("parameter value (log) — not comparable across families")
    ax[2].set_ylabel("distance computations per query (log)")
    ax[2].set_title("(c) what each parameter costs")

    for a in ax:
        a.grid(alpha=.3)
        a.legend(fontsize=8, loc="best")

    fig.suptitle("Knob 1 — quality vs the configured parameter · regime 2 · "
                 "short mode · n=1000", fontsize=11)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

# --- knob 2 ----------------------------------------------------------------

def fig_knob2(results_dir: str, out_path: str, manifest: list) -> None:
    """Regime 2, short mode. 2x2: quality on top, retrieval below.

    Aggressiveness increases to the RIGHT (compression ratio), the opposite
    of knob 1's ndis axis. The 17-point CPU sweep goes behind the retrieval
    panels in grey: it is retrieval-only (no generation), so it has no EM,
    but it shows the GPU points sampling a curve rather than defining it.
    """
    df = pd.read_csv(os.path.join(results_dir, "knob2_summary.csv"))
    base = df.loc[df.compression.astype(float).idxmin()]

    panels = (("em", "exact match", "(a) EM vs compression"),
              ("f1", "F1", "(b) F1 vs compression"),
              ("complete_frac", "complete_frac",
               "(c) both gold passages retrieved"),
              ("ann_recall", "ANN recall",
               "(d) agreement with exact top-5"))

    fig, axes = plt.subplots(2, 2, figsize=(11.5, 8.2))
    ax = axes.ravel()


    for key, label, colour, marker in (("pq", "PQ (product quant.)", C_A, "o"),
                                       ("sq", "SQ (scalar quant.)", C_B, "s")):
        d = df[df.setting.str.contains(key, case=False)
               & (df.setting != base.setting)].sort_values("compression")
        if d.empty:
            continue
        for j, (col, _, _) in enumerate(panels):
            ax[j].plot(d.compression, d[col], "-", marker=marker, ms=7,
                       color=colour, lw=1.4, zorder=3, label=label)
        for _, r in d.iterrows():
            manifest.append({"figure": "knob2_embedding_precision", "knob": 2,
                             "regime": 2, "setting": r.setting,
                             "x_axis": "compression",
                             "x": float(r.compression), "em": float(r.em),
                             "f1": float(r.f1),
                             "complete_frac": float(r.complete_frac),
                             "ann_recall": float(r.ann_recall)})

    for j, (col, ylab, title) in enumerate(panels):
        if col == "ann_recall":
            ax[j].axhline(1.0, ls="--", c="k", lw=1,
                          label="exact fp32 = 1.0 by definition")
        else:
            yb = float(base[col])
            ax[j].axhline(yb, ls="--", c="k", lw=1,
                          label=f"exact fp32 — {yb:.3f}")
            ax[j].scatter([float(base.compression)], [yb], marker="*", s=200,
                          c="k", zorder=6)
        ax[j].set_xscale("log")
        ax[j].set_xlabel("compression vs fp32 (log)\nright = more aggressive")
        ax[j].set_ylabel(ylab)
        ax[j].set_title(title)
        ax[j].grid(alpha=.3)
        ax[j].legend(fontsize=8, loc="best")

    fig.suptitle("Knob 2 — embedding precision · regime 2 (A1, 66,581 "
                 "passages) · short mode · n=1000", fontsize=11)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

def fig_knob2_params(results_dir: str, out_path: str, manifest: list) -> None:
    """The same data on the axes you configure.

    PQ is set by m (subquantizers per vector), SQ by bits per component.
    Separate panels: m=16 and 8 bits are not the same kind of number.

    knob2_summary.csv stores m as TEXT ("192", and "-" for SQ rows) because
    notebook 08 casts it to str to join against the dense sweep. Left as
    strings, matplotlib plots them categorically at positions 0..N and the
    log axis then drops position 0 — so coerce to numeric here and let the
    "-" rows become NaN.

    No parameter-to-compression panel: unlike knob 1's measured ndis,
    compression here is arithmetic (m bytes per vector against fp32's 1,536),
    so plotting it would present a definition as a result. It is annotated
    on the points instead.
    """
    df = pd.read_csv(os.path.join(results_dir, "knob2_summary.csv"))
    base = df.loc[df.compression.astype(float).idxmin()]
    fam = (("pq", "PQ", "m (subquantizers)", C_A, "o"),
           ("sq", "SQ", "bits per component", C_B, "s"))

    fig, ax = plt.subplots(1, 2, figsize=(10.5, 4.4))
    for j, (key, label, pname, colour, marker) in enumerate(fam):
        d = df[df.setting.str.contains(key, case=False)
               & (df.setting != base.setting)].copy()
        if d.empty:
            continue
        xcol = "m" if key == "pq" else "nbits"
        d[xcol] = pd.to_numeric(d[xcol], errors="coerce")
        d = d.dropna(subset=[xcol]).sort_values(xcol)
        if d.empty:
            continue

        ax[j].plot(d[xcol], d.em, "-", marker=marker, ms=7, color=colour,
                   lw=1.4, label=label)
        ax[j].axhline(base.em, ls="--", c="k", lw=1,
                      label=f"exact fp32 — EM {base.em:.3f}")
        ax[j].set_xscale("log")
        ax[j].set_xticks(list(d[xcol]))
        ax[j].get_xaxis().set_major_formatter(
            matplotlib.ticker.ScalarFormatter())
        ax[j].minorticks_off()
        ax[j].set_xlim(float(d[xcol].min()) * 0.7, float(d[xcol].max()) * 1.45)
        ax[j].set_ylim(df.em.min() - 0.03, df.em.max() + 0.03)
        ax[j].set_xlabel(f"{pname} (log) — left = more aggressive")
        ax[j].set_ylabel("exact match")
        ax[j].set_title(f"({'ab'[j]}) {label}: quality vs {pname}")

        for _, r in d.iterrows():
            ax[j].annotate(f"{float(r.compression):.0f}x",
                           (float(r[xcol]), float(r.em)),
                           textcoords="offset points", xytext=(0, -15),
                           ha="center", fontsize=7.5, color=colour)
            manifest.append({"figure": "knob2_parameters", "knob": 2,
                             "regime": 2, "setting": r.setting,
                             "x_axis": xcol, "x": float(r[xcol]),
                             "em": float(r.em),
                             "compression": float(r.compression),
                             "corpus_mib": float(r.corpus_mib)})

        ax[j].grid(alpha=.3)
        ax[j].legend(fontsize=8, loc="best")

    fig.suptitle("Knob 2 — quality vs the configured parameter · regime 2 · "
                 "short mode · n=1000", fontsize=11)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)



# --- knob 3 ----------------------------------------------------------------

def _k3_label(s: str) -> str:
    return str(s).replace("kv_hqq_", "").replace("_r512", "")


def fig_knob3(results_dir: str, out_path: str, manifest: list) -> None:
    """Regime 1, hqq backend. Quality vs KV storage cost.

    x is MEASURED effective bits (nbits plus per-group scale/zero metadata).
    It is NOT injective: 4b/g32 and 3b/g32 both land on 5.00, because 3-bit
    uses a packing whose payload does not follow nbits + 32/G. Series are
    therefore ordered by nbits, not by x, and the vertical segment at 5.00 is
    the group-size finding made visible rather than a glitch.

    residual_length=0 is drawn detached: it varies flush behaviour, not
    precision, and re-quantizes an already-lossy store on alternating decode
    steps.

    No retrieval panels: complete_frac and recall@5 are constant across every
    setting here. That constancy is the isolation check, stated in the title.
    """
    short = pd.read_csv(os.path.join(results_dir, "knob3_summary.csv"))
    expl = pd.read_csv(os.path.join(results_dir, "knob3_explain_summary.csv"))
    num = ("nbits", "q_group_size", "residual_length", "effective_bits",
           "em", "f1", "abstain_rate", "complete_frac")
    for d_ in (short, expl):
        for c in num:
            if c in d_.columns:
                d_[c] = pd.to_numeric(d_[c], errors="coerce")

    base = short.loc[short.effective_bits.idxmax()]
    ebase = expl.loc[expl.effective_bits.idxmax()]
    curves = short[(short.residual_length == 512)
                   & (short.setting != base.setting)]

    panels = (("em", "exact match", "(a) EM vs storage cost"),
              ("f1", "F1", "(b) F1 vs storage cost"),
              ("abstain_rate", "abstain rate",
               "(c) abstention falls as the cache degrades"))

    fig, axes = plt.subplots(2, 2, figsize=(11.5, 8.4))
    ax = axes.ravel()

    for g, colour, marker in ((32, C_A, "o"), (16, C_B, "s")):
        d = curves[curves.q_group_size == g].sort_values("nbits",
                                                         ascending=False)
        if d.empty:
            continue
        for j, (col, _, _) in enumerate(panels):
            ax[j].plot(d.effective_bits, d[col], "-", marker=marker, ms=7,
                       color=colour, lw=1.4, label=f"group {g}")
        for _, r in d.iterrows():
            ax[0].annotate(_k3_label(r.setting),
                           (float(r.effective_bits), float(r.em)),
                           textcoords="offset points", xytext=(0, -15),
                           ha="center", fontsize=7, color=colour)
            manifest.append({"figure": "knob3_kv_precision", "knob": 3,
                             "regime": 1, "mode": "short",
                             "setting": r.setting, "x_axis": "effective_bits",
                             "x": float(r.effective_bits), "em": float(r.em),
                             "f1": float(r.f1),
                             "abstain_rate": float(r.abstain_rate)})

    r0 = short[short.residual_length == 0]
    if not r0.empty:
        ax[0].plot(r0.effective_bits, r0.em, "^", ms=10, color="0.35",
                   ls="none", label="residual_length = 0 (flush probe)")

    for j, (col, ylab, title) in enumerate(panels):
        yb = float(base[col])
        ax[j].axhline(yb, ls="--", c="k", lw=1, label=f"fp16 — {yb:.3f}")
        ax[j].scatter([float(base.effective_bits)], [yb], marker="*", s=200,
                      c="k", zorder=6)
        ax[j].set_xlabel("effective bits per KV element (measured)\n"
                         "left = more aggressive")
        ax[j].set_ylabel(ylab)
        ax[j].set_title(title)

    # (d) the same settings under both answer formats.
    m = (expl[["setting", "effective_bits", "em"]]
         .rename(columns={"em": "em_explain"})
         .merge(short[["setting", "em"]].rename(columns={"em": "em_short"}),
                on="setting", how="inner")
         .sort_values("effective_bits", ascending=False))
    xs = list(range(len(m)))
    ax[3].bar([x - 0.19 for x in xs], m.em_short, width=0.38, color="0.6",
              label="short")
    ax[3].bar([x + 0.19 for x in xs], m.em_explain, width=0.38, color=C_A,
              label="explain")
    ax[3].axhline(float(base.em), ls="--", c="0.4", lw=1,
                  label=f"fp16 short — {float(base.em):.3f}")
    ax[3].axhline(float(ebase.em), ls="--", c=C_A, lw=1,
                  label=f"fp16 explain — {float(ebase.em):.3f}")
    ax[3].set_xticks(xs)
    ax[3].set_xticklabels([f"{_k3_label(s)}\n{b:.2f} bits"
                           for s, b in zip(m.setting, m.effective_bits)],
                          fontsize=8)
    ax[3].set_ylabel("exact match")
    ax[3].set_title("(d) explain vs short, same settings")
    for _, r in m.iterrows():
        manifest.append({"figure": "knob3_kv_precision", "knob": 3,
                         "regime": 1, "mode": "short+explain",
                         "setting": r.setting,
                         "effective_bits": float(r.effective_bits),
                         "em_short": float(r.em_short),
                         "em_explain": float(r.em_explain)})

    for a in ax:
        a.grid(alpha=.3)
        a.legend(fontsize=8, loc="best")

    fig.suptitle("Knob 3 — KV-cache precision · regime 1 · n=1000 · "
                 f"retrieval untouched (complete_frac "
                 f"{float(base.complete_frac):.3f} at every setting)",
                 fontsize=11)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

def fig_knob3_params(results_dir: str, out_path: str, manifest: list) -> None:
    """The same data on the axis you configure.

    nbits IS injective within a group size, which effective bits is not, so
    this is the axis the curve actually belongs on. Panel (b) is the
    translation and it earns its place: unlike knob 2's arithmetic
    compression, effective bits here is measured, and 3-bit provably breaks
    the nbits + 32/G formula that the other widths follow.
    """
    short = pd.read_csv(os.path.join(results_dir, "knob3_summary.csv"))
    for c in ("nbits", "q_group_size", "residual_length", "effective_bits",
              "em"):
        if c in short.columns:
            short[c] = pd.to_numeric(short[c], errors="coerce")

    base = short.loc[short.effective_bits.idxmax()]
    curves = short[(short.residual_length == 512)
                   & (short.setting != base.setting)]

    fig, ax = plt.subplots(1, 2, figsize=(10.5, 4.4))
    ticks = sorted(set(curves.nbits.dropna().astype(int)))

    for g, colour, marker in ((32, C_A, "o"), (16, C_B, "s")):
        d = curves[curves.q_group_size == g].sort_values("nbits")
        if d.empty:
            continue
        ax[0].plot(d.nbits, d.em, "-", marker=marker, ms=7, color=colour,
                   lw=1.4, label=f"group {g}")
        ax[1].plot(d.nbits, d.effective_bits, "-", marker=marker, ms=7,
                   color=colour, lw=1.4, label=f"group {g} (measured)")
        ax[1].plot(ticks, [t + 32.0 / g for t in ticks], ":", color=colour,
                   lw=1.1, alpha=.7, label=f"group {g} (nbits + 32/G)")
        for _, r in d.iterrows():
            manifest.append({"figure": "knob3_parameters", "knob": 3,
                             "regime": 1, "mode": "short",
                             "setting": r.setting, "x_axis": "nbits",
                             "x": float(r.nbits), "em": float(r.em),
                             "effective_bits": float(r.effective_bits)})

    ax[0].axhline(float(base.em), ls="--", c="k", lw=1,
                  label=f"fp16 — EM {float(base.em):.3f}")
    ax[0].set_ylabel("exact match")
    ax[0].set_title("(a) quality vs nominal bit width")
    ax[1].set_ylabel("effective bits per element")
    ax[1].set_title("(b) what each width actually costs")

    for a in ax:
        a.set_xticks(ticks)
        a.set_xlabel("nbits (hqq) — left = more aggressive")
        a.grid(alpha=.3)
        a.legend(fontsize=8, loc="best")

    fig.suptitle("Knob 3 — quality vs the configured parameter · regime 1 · "
                 "short mode · n=1000", fontsize=11)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

# --- knob 4 ----------------------------------------------------------------

POLICY_STYLE = {
    "recency":     ("tab:red",    "o", "recency"),
    "sink_recent": ("tab:orange", "s", "sink + recent"),
    "random":      ("0.45",       "^", "random (control)"),
    "attention":   ("tab:blue",   "D", "attention (H2O/SnapKV)"),
}


def fig_knob4(results_dir: str, out_path: str, manifest: list) -> None:
    """Regime 1, batch 1. Quality vs KV-cache eviction aggressiveness.

    x is keep_ratio, inverted so more aggressive reads right.

    The explain arm has keep_ratio 0.50 and 0.25 only; short also has 0.75.
    That asymmetry is left visible — no interpolation across the missing
    ratio. sink4_r050_naive is the RoPE control, not a policy setting, so it
    is drawn detached.
    """
    df = pd.read_csv(os.path.join(results_dir, "knob4_all_settings.csv"))
    for c in ("keep_ratio", "em", "f1", "parse_given_stopped"):
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    sh = df[df["mode"] == "short"]
    ex = df[df["mode"] == "explain"]
    ref_s = sh[sh.policy == "none"].iloc[0]
    ref_e = ex[ex.policy == "none"].iloc[0]

    panels = ((sh, "em", "exact match", "(a) short: EM vs keep_ratio", ref_s),
              (ex, "em", "exact match", "(b) explain: EM vs keep_ratio",
               ref_e),
              (sh, "f1", "F1", "(c) short: F1 vs keep_ratio", ref_s),
              (ex, "parse_given_stopped", "parse rate | stopped cleanly",
               "(d) explain: format adherence", ref_e))

    fig, axes = plt.subplots(2, 2, figsize=(11.5, 8.4))
    ax = axes.ravel()

    for j, (src, col, ylab, title, ref) in enumerate(panels):
        for pol, (colour, marker, label) in POLICY_STYLE.items():
            d = src[(src.policy == pol)].dropna(subset=[col])
            d = d.sort_values("keep_ratio", ascending=False)
            if d.empty:
                continue
            ax[j].plot(d.keep_ratio, d[col], "-", marker=marker, ms=7,
                       color=colour, lw=1.4, label=label)
            if j in (0, 1):
                for _, r in d.iterrows():
                    manifest.append({"figure": "knob4_eviction", "knob": 4,
                                     "regime": 1, "mode": r["mode"],
                                     "setting": r.setting, "policy": pol,
                                     "x_axis": "keep_ratio",
                                     "x": float(r.keep_ratio),
                                     "em": float(r.em)})

        naive = src[src.policy == "sink_recent_naive"].dropna(subset=[col])
        if not naive.empty:
            ax[j].plot(naive.keep_ratio, naive[col], "o", ms=17, color="0.25",
                       fillstyle="none", markeredgewidth=1.6, ls="none",
                       zorder=2,
                       label="sink+recent, no RoPE shift (control)")

        yb = float(ref[col]) if pd.notna(ref[col]) else None
        if yb is not None:
            ax[j].axhline(yb, ls="--", c="k", lw=1,
                          label=f"no eviction — {yb:.3f}")
            ax[j].scatter([float(ref.keep_ratio)], [yb], marker="*", s=200,
                          c="k", zorder=6)

        ax[j].set_xlabel("keep_ratio (fraction of cache kept)\n"
                         "left = more aggressive")
        ax[j].set_ylabel(ylab)
        ax[j].set_title(title)
        ax[j].grid(alpha=.3)
        ax[j].legend(fontsize=7.5, loc="best")

    fig.suptitle("Knob 4 — KV-cache eviction · regime 1 · batch_size 1 · "
                 "n=1000 · retrieval untouched", fontsize=11)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


# --- knob 5 ----------------------------------------------------------------

def fig_knob5(results_dir: str, out_path: str, manifest: list) -> None:
    """Both regimes, short mode. Quality vs top-k.

    k=20 in regime 2 ran at batch 8 (it OOMs at the frozen batch 16), so it
    is NOT joined to the batch-16 line — it is drawn as its own pair with the
    matched k=5 batch-8 control, which is the only setting it can legally be
    compared against.
    """
    r1 = pd.read_csv(os.path.join(results_dir, "knob5_summary.csv"))
    r2 = pd.read_csv(os.path.join(results_dir, "knob5_r2_summary.csv"))
    cols = ("k", "batch", "em", "f1", "complete_frac", "prompt_tokens_mean")
    for d_ in (r1, r2):
        for c in cols:
            if c in d_.columns:
                d_[c] = pd.to_numeric(d_[c], errors="coerce")

    r2_16 = r2[r2.batch == 16].sort_values("k")
    r2_8 = r2[r2.batch == 8].sort_values("k")
    b1 = r1[r1.k == 5].iloc[0]
    b2 = r2_16[r2_16.k == 5].iloc[0]

    panels = (("em", "exact match", "(a) EM vs k", "{:.3f}"),
              ("f1", "F1", "(b) F1 vs k", "{:.3f}"),
              ("complete_frac", "complete_frac",
               "(c) both gold passages in the prompt", "{:.3f}"),
              ("prompt_tokens_mean", "prompt tokens (mean)",
               "(d) what k costs the generator", "{:.0f}"))

    fig, axes = plt.subplots(2, 2, figsize=(11.5, 8.4))
    ax = axes.ravel()

    for j, (col, ylab, title, fmt) in enumerate(panels):
        ax[j].plot(r1.sort_values("k").k, r1.sort_values("k")[col], "-",
                   marker="o", ms=7, color=C_A, lw=1.4,
                   label="regime 1 (per-question)")
        ax[j].plot(r2_16.k, r2_16[col], "-", marker="s", ms=7, color=C_B,
                   lw=1.4, label="regime 2 (pooled A1)")
        if not r2_8.empty:
            ax[j].plot(r2_8.k, r2_8[col], ":", marker="s", ms=8, color=C_B,
                       lw=1.2, fillstyle="none", markeredgewidth=1.4,
                       label="regime 2, batch 8 (k=20 + matched control)")
        ax[j].axhline(float(b1[col]), ls="--", c=C_A, lw=1,
                      label=f"regime 1, k=5 — {fmt.format(float(b1[col]))}")
        ax[j].axhline(float(b2[col]), ls="--", c=C_B, lw=1,
                      label=f"regime 2, k=5 — {fmt.format(float(b2[col]))}")
        ax[j].set_xscale("log")
        ticks = sorted(set(r1.k.dropna().astype(int))
                       | set(r2.k.dropna().astype(int)))
        ax[j].set_xticks(ticks)
        ax[j].get_xaxis().set_major_formatter(
            matplotlib.ticker.ScalarFormatter())
        ax[j].minorticks_off()
        ax[j].set_xlabel("top-k passages retrieved (log)")
        ax[j].set_ylabel(ylab)
        ax[j].set_title(title)
        ax[j].grid(alpha=.3)
        ax[j].legend(fontsize=7.5, loc="best")
    ax[3].set_yscale("log")

    for src, regime, batch in ((r1, 1, 16), (r2_16, 2, 16), (r2_8, 2, 8)):
        for _, r in src.iterrows():
            manifest.append({"figure": "knob5_topk", "knob": 5,
                             "regime": regime, "batch": batch,
                             "setting": r.get("setting", f"r1_topk_{int(r.k)}"),
                             "x_axis": "k", "x": float(r.k),
                             "em": float(r.em), "f1": float(r.f1),
                             "complete_frac": float(r.complete_frac),
                             "prompt_tokens_mean": float(r.prompt_tokens_mean)})

    fig.suptitle("Knob 5 — top-k data sampling · regimes 1 and 2 · short "
                 "mode · n=1000", fontsize=11)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

# --- cross-knob -------------------------------------------------------------

def fig_crossknob(results_dir: str, out_path: str, manifest: list) -> None:
    """Qualitative panel: five knobs, shared EM axis, own x-axis each.

    NOT one plot. The knobs run in different regimes, at different modes,
    with aggressiveness axes that share no units — ndis, compression ratio,
    effective bits, keep_ratio, k. Forcing them onto a common x would require
    inventing a normalisation nobody measured. The shared y-axis is the only
    honest thing they have in common, so that is the only thing shared, and
    each panel carries its own reference line and its own coverage note.
    """
    def load(name, cols):
        d = pd.read_csv(os.path.join(results_dir, name))
        for c in cols:
            if c in d.columns:
                d[c] = pd.to_numeric(d[c], errors="coerce")
        return d

    k1 = load("knob1_summary.csv", ("ndis", "em"))
    k2 = load("knob2_summary.csv", ("compression", "em"))
    k3 = load("knob3_summary.csv",
              ("effective_bits", "em", "q_group_size", "residual_length",
               "nbits"))
    k4 = load("knob4_all_settings.csv", ("keep_ratio", "em"))
    k5a = load("knob5_summary.csv", ("k", "em"))
    k5b = load("knob5_r2_summary.csv", ("k", "em", "batch"))

    fig, axes = plt.subplots(1, 5, figsize=(19, 4.9), sharey=True)
    ax = axes.ravel()
    ems, notes = [], []

    # knob 1
    b1 = k1[k1["index"] == "flat"].iloc[0]
    for fam, lab, col, mk in (("hnsw", "HNSW", C_A, "o"),
                              ("ivf", "IVF", C_B, "s")):
        d = k1[k1["index"] == fam].sort_values("ndis")
        ax[0].plot(d.ndis, d.em, "-", marker=mk, ms=6, color=col, lw=1.3,
                   label=lab)
        ems += list(d.em)
    ax[0].axhline(float(b1.em), ls="--", c="k", lw=1, label="exact")
    ax[0].set_xscale("log")
    ax[0].set_xlabel("ndis (log) — left = aggressive")
    ax[0].set_title("1 · search effort\nregime 2 · short")
    notes.append(("1 · search effort", "regime 2 · short", "ndis",
                  float(b1.em)))

    # knob 2
    b2 = k2.loc[k2.compression.idxmin()]
    for key, lab, col, mk in (("pq", "PQ", C_A, "o"), ("sq", "SQ", C_B, "s")):
        d = k2[k2.setting.str.contains(key, case=False)
               & (k2.setting != b2.setting)].sort_values("compression")
        ax[1].plot(d.compression, d.em, "-", marker=mk, ms=6, color=col,
                   lw=1.3, label=lab)
        ems += list(d.em)
    ax[1].axhline(float(b2.em), ls="--", c="k", lw=1, label="exact fp32")
    ax[1].set_xscale("log")
    ax[1].set_xlabel("compression (log) — right = aggressive")
    ax[1].set_title("2 · embedding precision\nregime 2 · short")
    notes.append(("2 · embedding precision", "regime 2 · short",
                  "compression", float(b2.em)))

    # knob 3
    b3 = k3.loc[k3.effective_bits.idxmax()]
    cur3 = k3[(k3.residual_length == 512) & (k3.setting != b3.setting)]
    for g, col, mk in ((32, C_A, "o"), (16, C_B, "s")):
        d = cur3[cur3.q_group_size == g].sort_values("nbits", ascending=False)
        if d.empty:
            continue
        ax[2].plot(d.effective_bits, d.em, "-", marker=mk, ms=6, color=col,
                   lw=1.3, label=f"group {g}")
        ems += list(d.em)
    ax[2].axhline(float(b3.em), ls="--", c="k", lw=1, label="fp16")
    ax[2].set_xlabel("effective bits — left = aggressive")
    ax[2].set_title("3 · KV precision\nregime 1 · short")
    notes.append(("3 · KV precision", "regime 1 · short", "effective bits",
                  float(b3.em)))

    # knob 4
    sh = k4[k4["mode"] == "short"]
    b4 = sh[sh.policy == "none"].iloc[0]
    for pol, (col, mk, lab) in POLICY_STYLE.items():
        d = sh[sh.policy == pol].sort_values("keep_ratio")
        if d.empty:
            continue
        ax[3].plot(d.keep_ratio, d.em, "-", marker=mk, ms=6, color=col,
                   lw=1.3, label=lab)
        ems += list(d.em)
    ax[3].axhline(float(b4.em), ls="--", c="k", lw=1, label="no eviction")
    ax[3].set_xlabel("keep_ratio — left = aggressive")
    ax[3].set_title("4 · KV eviction\nregime 1 · short · batch 1")
    notes.append(("4 · KV eviction", "regime 1 · short · batch 1",
                  "keep_ratio", float(b4.em)))

    # knob 5
    r2_16 = k5b[k5b.batch == 16].sort_values("k")
    d1 = k5a.sort_values("k")
    ax[4].plot(d1.k, d1.em, "-", marker="o", ms=6, color=C_A, lw=1.3,
               label="regime 1")
    ax[4].plot(r2_16.k, r2_16.em, "-", marker="s", ms=6, color=C_B, lw=1.3,
               label="regime 2")
    ems += list(d1.em) + list(r2_16.em)
    ax[4].axhline(float(d1[d1.k == 5].iloc[0].em), ls="--", c=C_A, lw=1,
                  label="regime 1, k=5")
    ax[4].axhline(float(r2_16[r2_16.k == 5].iloc[0].em), ls="--", c=C_B,
                  lw=1, label="regime 2, k=5")
    ax[4].set_xscale("log")
    ax[4].set_xticks(sorted(set(d1.k.astype(int)) | set(r2_16.k.astype(int))))
    ax[4].get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    ax[4].minorticks_off()
    ax[4].set_xlabel("k (log) — left = aggressive")
    ax[4].set_title("5 · top-k\nregimes 1 and 2 · short")
    notes.append(("5 · top-k", "regimes 1 and 2 · short", "k",
                  float(d1[d1.k == 5].iloc[0].em)))

    for a in ax[:5]:
        a.grid(alpha=.3)
        a.legend(fontsize=6.5, loc="lower right", framealpha=.9)
    ax[0].set_ylabel("exact match")
    lo, hi = min(ems), max(ems)
    ax[0].set_ylim(lo - 0.04, hi + 0.04)

    # coverage cell — the asymmetries, stated rather than hidden
    cov = (
        "Only the EM axis is shared — the five x-axes share no units, so this "
        "is a qualitative panel, not one plot. Regime 1's baseline is EM "
        "0.391 and regime 2's is 0.306, so vertical\nposition is comparable "
        "WITHIN a panel, not across panels. Coverage: knobs 1-2 ran regime 2 "
        "only · knobs 3-4 regime 1 only · knob 5 both · knob 4's explain arm "
        "covers 2 keep_ratios where short covers 3."
    )
    fig.text(0.5, -0.02, cov, ha="center", va="top", fontsize=8,
             linespacing=1.6)

    for n, cov, axis, ref in notes:
        manifest.append({"figure": "crossknob_panel", "knob": n.split(" ")[0],
                         "coverage": cov, "x_axis": axis,
                         "reference_em": ref})

    fig.suptitle("Phase 2 cross-knob panel — five approximation knobs on a "
                 "shared EM axis (qualitative comparison)", fontsize=12)
    fig.subplots_adjust(left=0.045, right=0.995, top=0.80, bottom=0.20,
                        wspace=0.18)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

@dataclass
class Input:
    path: str
    required: bool
    note: str = ""


INPUTS = [
    Input("knob1_summary.csv", True),
    Input("knob1_summary.json", True, "mcnemar: 5 comparisons, 8 settings"),
    Input("knob2_summary.csv", True),
    Input("knob2_summary.json", True),
    Input("knob2_dense_sweep.csv", True, "16 free points for the mechanism panel"),
    Input("knob3_summary.csv", True, "short arm"),
    Input("knob3_summary.json", True),
    Input("knob3_explain_summary.csv", True),
    Input("knob3_explain_summary.json", True),
    Input("knob4_short_summary.csv", True),
    Input("knob4_short_summary.json", True),
    Input("knob4_explain_summary.csv", True),
    Input("knob4_explain_summary.json", True),
    Input("knob4_all_settings.csv", False, "preferred; falls back to the two arms"),
    Input("knob4_summary.json", False, "caveats block for the figure footnote"),
    Input("knob5_summary.csv", True, "regime 1"),
    Input("knob5_summary.json", True),
    Input("knob5_r2_summary.csv", True, "regime 2"),
    Input("knob5_r2_summary.json", True),
]


def describe_csv(path: str) -> None:
    df = pd.read_csv(path)
    print(f"    rows={len(df)}  cols={list(df.columns)}")


def describe_json(path: str) -> None:
    with open(path) as fh:
        obj = json.load(fh)
    print(f"    top-level keys: {list(obj)}")
    settings = obj.get("settings")
    if isinstance(settings, list) and settings:
        print(f"    settings: {len(settings)} records, "
              f"keys={list(settings[0])}")
    for key in (k for k in obj if k.startswith("mcnemar")):
        mc = obj[key]
        if isinstance(mc, dict):
            shown = list(mc)[:8]
            more = "" if len(mc) <= 8 else f"  (+{len(mc) - 8} more)"
            print(f"    {key}: dict-of-{len(mc)}, keys={shown}{more}")
            for rec in mc.values():
                if isinstance(rec, dict):
                    print(f"      record keys={list(rec)}")
                    break
        elif isinstance(mc, list):
            print(f"    {key}: list-of-{len(mc)}")
            if mc and isinstance(mc[0], dict):
                print(f"      record keys={list(mc[0])}")
        else:
            print(f"    {key}: {type(mc).__name__} = {mc!r}")


def check(results_dir: str) -> int:
    missing_required, missing_optional = [], []
    print(f"results dir: {os.path.abspath(results_dir)}\n")
    for inp in INPUTS:
        path = os.path.join(results_dir, inp.path)
        tag = "" if inp.required else "  [optional]"
        if not os.path.exists(path):
            (missing_required if inp.required else missing_optional).append(inp.path)
            print(f"  MISSING  {inp.path}{tag}"
                  + (f"  — {inp.note}" if inp.note else ""))
            continue
        print(f"  ok       {inp.path}{tag}"
              + (f"  — {inp.note}" if inp.note else ""))
        try:
            if path.endswith(".csv"):
                describe_csv(path)
            else:
                describe_json(path)
        except Exception as exc:
            print(f"    UNREADABLE: {type(exc).__name__}: {exc}")
            if inp.required:
                missing_required.append(inp.path)

    print()
    if missing_optional:
        print(f"optional absent: {', '.join(missing_optional)}")
    if missing_required:
        print(f"REQUIRED ABSENT: {', '.join(missing_required)}")
        print("Everything under results/ is gitignored by default — check the "
              "whitelist in .gitignore and `git ls-files results/`.")
        return 1
    print("all required inputs present")
    return 0


def self_test() -> int:
    """Every case below is a real number from the study, and every case can fail."""
    cases = [
        # (b, c, p, expected, provenance)
        (0, 0, 1.0000, NULL_ZERO, "knob4 rope shifted vs naive"),
        (2, 11, 0.0225, SIGNIFICANT, "knob1 exact vs hnsw_ef32, d=13"),
        (67, 75, 0.5571, NULL, "knob3 explain fp16 vs n4_g32"),
        (3, 1, 0.6250, UNINFORMATIVE, "knob2 exact vs sq_8bit, d=4"),
        (20, 13, 0.2962, NULL, "knob1 hnsw_ef32 vs ivf_nprobe48"),
        (0, 424, 0.0000, SIGNIFICANT, "knob3 explain fp16 vs n2_g32"),
        (4, 8, 0.3877, UNDERPOWERED, "knob4 explain ref vs frozen, d=12"),
    ]
    failures = []
    for b, c, p, want, why in cases:
        got = verdict(b, c, p)
        if got != want:
            failures.append(f"verdict({b},{c},{p}) = {got!r}, want {want!r}  [{why}]")

    # Negative case: the fixture MUST disagree on the carve-out, otherwise the
    # ordering this function exists to protect is not actually being tested.
    if _verdict_no_carveout(0, 0, 1.0) == verdict(0, 0, 1.0):
        failures.append(
            "no-carve-out fixture agrees with verdict() on d=0 — the carve-out "
            "test is vacuous; it would pass with the bug reinstated")

    # Negative case: p must dominate d. A low-discordant significant result
    # is SIGNIFICANT, not UNINFORMATIVE.
    if verdict(2, 11, 0.0225) == UNINFORMATIVE:
        failures.append("d<10 branch is shadowing a significant p")

    for f in failures:
        print(f"FAIL  {f}")
    print(f"\n{len(cases) + 2 - len(failures)}/{len(cases) + 2} checks passed")
    return 1 if failures else 0


FIGURES = [
    ("knob1_search_effort", fig_knob1),
    ("knob1_parameters", fig_knob1_params),
    ("knob2_embedding_precision", fig_knob2),
    ("knob2_parameters", fig_knob2_params),
    ("knob3_kv_precision", fig_knob3),
    ("knob3_parameters", fig_knob3_params),
    ("knob4_eviction", fig_knob4),
    ("knob5_topk", fig_knob5),
    ("crossknob_panel", fig_crossknob),
]


def build(results_dir: str, out_dir: str) -> int:
    os.makedirs(out_dir, exist_ok=True)
    manifest: list = []
    for name, fn in FIGURES:
        path = os.path.join(out_dir, f"{name}.png")
        fn(results_dir, path, manifest)
        print(f"wrote {path}")
    mpath = os.path.join(out_dir, "phase2_plot_manifest.json")
    with open(mpath, "w") as fh:
        json.dump({"figures": [n for n, _ in FIGURES],
                   "points": manifest}, fh, indent=2)
    print(f"wrote {mpath}  ({len(manifest)} plotted points)")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--results-dir", default="results")
    ap.add_argument("--out-dir", default="results/figures")
    ap.add_argument("--check", action="store_true",
                    help="inventory and print input schemas, then exit")
    ap.add_argument("--self-test", action="store_true",
                    help="verdict logic only, no files touched")
    args = ap.parse_args()

    if args.self_test:
        return self_test()
    if args.check:
        return check(args.results_dir)

    rc = self_test()
    if rc:
        return rc
    rc = check(args.results_dir)
    if rc:
        return rc
    print()
    return build(args.results_dir, args.out_dir)


if __name__ == "__main__":
    sys.exit(main())
