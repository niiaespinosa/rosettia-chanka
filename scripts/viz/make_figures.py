"""Model-card figures for rosettia-quy (Spanish -> Chanka/Ayacucho Quechua).

Clean, conservative, big-lab style. ALL numbers are ChrF (sacrebleu word_order=0),
AmericasNLP 2021 spa->quy test (1003 sentences, single reference) unless noted.
No boasting: we show exactly what we measured, with caveats in the footnotes.
"""
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator

OUT = "docs/cards/figures"
os.makedirs(OUT, exist_ok=True)

# ---- restrained palette -------------------------------------------------
INK   = "#22272e"   # near-black text
GRID  = "#e6e8ec"
PRIOR = "#b8c0cc"   # published / prior work (muted gray-blue)
OURS  = "#5b8fb9"   # our intermediate systems
BEST  = "#256d6b"   # our best system (calm teal, not a loud color)
NLLB  = "#9bbcd6"
plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 11,
    "text.color": INK, "axes.labelcolor": INK,
    "xtick.color": INK, "ytick.color": INK,
    "axes.edgecolor": "#c3c8d0", "axes.linewidth": 0.8,
    "figure.dpi": 200, "savefig.dpi": 200,
    "savefig.bbox": "tight", "savefig.facecolor": "white",
})


def _clean(ax):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.tick_params(length=0)


def _titles(ax, title, subtitle):
    ax.text(0, 1.13, title, transform=ax.transAxes, fontsize=14.5, fontweight="bold", va="top")
    ax.text(0, 1.055, subtitle, transform=ax.transAxes, fontsize=10, color="#5b6675", va="top")


def fig_comparison():
    # (label, value, color, is_ours)
    rows = [
        ("Sheffield 2023  (NLLB-3.3B ensemble)", 34.01, PRIOR),
        ("Helsinki 2021  (prior task winner)",   39.40, PRIOR),
        ("Qwen-9B (ours), greedy",               40.55, OURS),
        ("NLLB-1.3B (ours), supervised",         42.95, OURS),
        ("+ GSPO reinforcement learning",        45.53, OURS),
        ("+ MBR decoding (single model)",        46.43, OURS),
        ("Our best system",                      46.71, BEST),
    ]
    labels = [r[0] for r in rows]; vals = [r[1] for r in rows]; cols = [r[2] for r in rows]
    fig, ax = plt.subplots(figsize=(8.6, 4.5))
    y = range(len(rows))
    ax.barh(y, vals, color=cols, height=0.62, zorder=3)
    ax.set_yticks(y); ax.set_yticklabels(labels, fontsize=10.3)
    ax.invert_yaxis()
    ax.set_xlim(30, 49)
    ax.xaxis.set_major_locator(MultipleLocator(2))
    ax.xaxis.grid(True, color=GRID, zorder=0)
    ax.set_axisbelow(True)
    for yi, v, c in zip(y, vals, cols):
        ax.text(v + 0.18, yi, f"{v:.2f}", va="center", ha="left", fontsize=10,
                fontweight="bold" if c == BEST else "normal", color=INK)
    _clean(ax)
    _titles(ax, "Spanish → Chanka (Ayacucho) Quechua",
            "ChrF (sacrebleu, word_order=0) · AmericasNLP 2021 test · 1003 sentences · single reference")
    ax.set_xlabel("ChrF (higher is better)", fontsize=10)
    fig.text(0.0, -0.02,
             "Prior work in gray. Our build-up in blue; best system in teal. "
             "BSC-2024 (2024 task winner) reported 38.21 ChrF++ (word_order=2), not directly\n"
             "comparable to the word_order=0 scores shown here. Single benchmark, single reference, no human evaluation.",
             fontsize=7.6, color="#7a828f")
    fig.savefig(f"{OUT}/fig1_comparison.png"); plt.close(fig)


def fig_gspo_curve():
    steps = [0, 200, 400, 600, 800, 1000]
    val   = [46.98, 50.70, 51.98, 52.99, 52.94, 52.05]
    fig, ax = plt.subplots(figsize=(7.4, 4.3))
    ax.plot(steps, val, "-o", color=BEST, lw=2, ms=6, zorder=3)
    # mark the val-selected peak honestly
    pk = val.index(max(val))
    ax.scatter([steps[pk]], [val[pk]], s=150, facecolors="none", edgecolors=BEST, lw=1.8, zorder=4)
    ax.annotate("val-selected\ncheckpoint", (steps[pk], val[pk]), textcoords="offset points",
                xytext=(6, -38), fontsize=9, color="#5b6675")
    ax.axhline(46.98, color=PRIOR, ls="--", lw=1.2, zorder=1)
    ax.text(1000, 46.98 + 0.12, "no-RL baseline", ha="right", va="bottom", fontsize=9, color="#7a828f")
    ax.set_xlim(-30, 1060); ax.set_ylim(46, 54)
    ax.yaxis.grid(True, color=GRID); ax.set_axisbelow(True)
    _clean(ax)
    _titles(ax, "GSPO reinforcement learning — held-out validation",
            "Validation ChrF (w0) vs RL step. Reward = sentence ChrF on unseen Ayacucho data.")
    ax.set_xlabel("GSPO step", fontsize=10); ax.set_ylabel("Validation ChrF", fontsize=10)
    fig.text(0.0, -0.02,
             "Validation peaks then declines (over-optimization) — we select the peak checkpoint and run a single\n"
             "test evaluation. Validation set is held out from RL training; the test set is never used for selection.",
             fontsize=7.6, color="#7a828f")
    fig.savefig(f"{OUT}/fig2_gspo_curve.png"); plt.close(fig)


def fig_scorecard():
    # higher = better on all axes (transforms noted in caption)
    axes_lbl = ["ChrF\n(surface)", "Adequacy\n(round-trip)", "Clean output\n(100 − ES-leak%)", "Length match\n(100·(1−|err|))"]
    nllb = [43.17, 48.28, 96.61, 79.9]
    gspo = [45.53, 52.86, 97.31, 82.6]
    import numpy as np
    x = np.arange(len(axes_lbl)); w = 0.38
    fig, ax = plt.subplots(figsize=(7.8, 4.4))
    b1 = ax.bar(x - w/2, nllb, w, label="NLLB-1.3B (supervised, pre-RL)", color=NLLB, zorder=3)
    b2 = ax.bar(x + w/2, gspo, w, label="+ GSPO RL", color=BEST, zorder=3)
    ax.set_xticks(x); ax.set_xticklabels(axes_lbl, fontsize=9.3)
    ax.set_ylim(0, 105); ax.yaxis.grid(True, color=GRID); ax.set_axisbelow(True)
    for b in (b1, b2):
        for r in b:
            ax.text(r.get_x() + r.get_width()/2, r.get_height() + 1.2, f"{r.get_height():.1f}",
                    ha="center", fontsize=8.6, color=INK)
    _clean(ax)
    ax.legend(frameon=False, fontsize=9.3, loc="lower center", ncol=2, bbox_to_anchor=(0.5, -0.0))
    _titles(ax, "Quality beyond the surface metric",
            "GSPO improved multiple automatic quality axes, not only ChrF — evidence the RL gains are real.")
    fig.text(0.0, -0.04,
             "All axes oriented higher = better. Adequacy = ChrF between the back-translated output (quy→spa) and the\n"
             "original Spanish source. ES-leak% = sentences with an untranslated Spanish function word. Length match from\n"
             "mean |pred/ref length − 1|. Automatic proxies (no human judgments exist for this language) — directional, not absolute.",
             fontsize=7.4, color="#7a828f")
    fig.savefig(f"{OUT}/fig3_scorecard.png"); plt.close(fig)


if __name__ == "__main__":
    fig_comparison(); fig_gspo_curve(); fig_scorecard()
    print("wrote figures to", OUT)
