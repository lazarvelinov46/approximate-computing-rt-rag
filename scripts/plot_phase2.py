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
    UNINFORMATIVE: ("none",   9, 0.55),
    None:          ("none",   5, 0.55),   # no paired test exists
}

MARKER_LEGEND = ("markers:  d=N* significant (p<=0.05)  |  d=N ns null (d>=30)  "
                 "|  d=N ns? underpowered (10<=d<30)  |  d=N ? uninformative "
                 "(d<10)  |  small unlabelled point = no paired test exists")

C_A, C_B = "tab:blue", "tab:orange"


def draw_point(ax, x, y, marker, colour, rec, manifest_row=None,
               manifest=None) -> None:
    fill, ms, alpha = POINT_STYLE[verdict_of(rec)]
    ax.plot([x], [y], marker=marker, color=colour, linestyle="none",
            fillstyle=fill, markersize=ms, alpha=alpha,
            markeredgewidth=1.2, zorder=5)
    if rec:
        ax.annotate(tag_of(rec), (x, y), textcoords="offset points",
                    xytext=(0, 10), ha="center", fontsize=7, color=colour)
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
    """Regime 2, short mode. x = distance computations per query (log).

    ndis is the aggressiveness axis rather than ef/nprobe because ef and
    nprobe are not comparable numbers across index families; ndis is the work
    both families actually do. Decreasing ndis = more aggressive.
    """
    df = pd.read_csv(os.path.join(results_dir, "knob1_summary.csv"))
    with open(os.path.join(results_dir, "knob1_summary.json")) as fh:
        mc = mc_index(json.load(fh))

    base = df[df["index"] == "flat"].iloc[0]
    fig, ax = plt.subplots(1, 2, figsize=(11.5, 4.4))

    for family, label, colour, marker in (("hnsw", "HNSW (ef)", C_A, "o"),
                                          ("ivf", "IVF (nprobe)", C_B, "s")):
        d = df[df["index"] == family].sort_values("ndis")
        ax[0].plot(d.ndis, d.em, "-", color=colour, lw=1.2, label=f"{label} EM")
        ax[0].plot(d.ndis, d.f1, "--", color=colour, lw=1.0, alpha=.65,
                   label=f"{label} F1")
        for _, r in d.iterrows():
            rec = mc.get(KNOB1_MC.get(r.setting, ""))
            draw_point(ax[0], r.ndis, r.em, marker, colour, rec,
                       {"figure": "knob1_search_effort", "knob": 1,
                        "regime": 2, "setting": r.setting,
                        "x_axis": "ndis", "x": float(r.ndis),
                        "em": float(r.em), "f1": float(r.f1)}, manifest)
        ax[1].plot(d.ndis, d.complete_frac, "-", marker=marker, ms=5,
                   color=colour, lw=1.2, label=f"{label} complete_frac")
        ax[1].plot(d.ndis, d.ann_recall, ":", marker=marker, ms=4, alpha=.6,
                   color=colour, lw=1.0, label=f"{label} ANN recall")

    for a, yb, lab in ((ax[0], base.em, f"exact EM {base.em:.3f}"),
                       (ax[1], base.complete_frac, "exact complete_frac")):
        a.axhline(yb, ls="--", c="k", lw=1, label=lab)
        a.scatter([base.ndis], [yb], marker="*", s=170, c="k", zorder=6)
        a.set_xscale("log")
        a.set_xlabel("distance computations per query (ndis, log) "
                     "— left is more aggressive")
        a.grid(alpha=.3)
        a.legend(fontsize=7.5)

    ax[0].axhline(base.f1, ls=":", c="k", lw=.8)
    ax[0].set_ylabel("quality")
    ax[0].set_title("(a) EM and F1 vs search effort")
    ax[1].set_ylabel("fraction")
    ax[1].set_title("(b) mechanism: evidence delivered to the generator")

    fig.suptitle("Knob 1 — retrieval search effort · regime 2 (A1, 66,581 "
                 "passages) · short mode · n=1000", fontsize=11)
    fig.text(0.5, 0.005, MARKER_LEGEND, ha="center", fontsize=7)
    fig.tight_layout(rect=(0, 0.035, 1, 1))
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
    untested = sum(1 for r in manifest if r["verdict"] == "no paired test")
    print(f"wrote {mpath}  ({len(manifest)} points, {untested} untested)")
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
