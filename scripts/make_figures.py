"""Render training and evaluation figures into artifacts/reports/ (PNG).

Colours follow the project's categorical order (fixed per entity, never by
rank); every multi-series chart has a legend plus direct end labels, and two
measures with different scales are drawn as separate panels, never on a dual
axis. The same numbers are available as tables in artifacts/eval/latest/summary.md.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "artifacts/reports"
INK, INK2, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
LANG_COLOR = {"hi": "#2a78d6", "bn": "#eb6834", "ta": "#1baf7a"}
LANG_NAME = {"hi": "Hindi", "bn": "Bengali", "ta": "Tamil"}
CK_COLOR = {"base": "#2a78d6", "sft": "#eb6834", "ppo": "#1baf7a", "sft+keyword_filter": "#eda100"}
CK_NAME = {"base": "Base", "sft": "SFT", "ppo": "PPO", "sft+keyword_filter": "SFT + keyword filter"}


def style(ax, title, xlabel, ylabel):
    ax.set_facecolor(SURFACE)
    ax.set_title(title, loc="left", fontsize=12, color=INK, pad=10)
    ax.set_xlabel(xlabel, color=INK2)
    ax.set_ylabel(ylabel, color=INK2)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=INK2)


def jsonl(path: Path):
    return [json.loads(l) for l in path.read_text().splitlines() if l.strip()] if path.exists() else []


def latest_run(stage: str) -> Path | None:
    runs = sorted((ROOT / "artifacts/runs").glob(f"*_{stage}_*"))
    return runs[-1] if runs else None


def pretrain_figure():
    run = latest_run("pretrain")
    rows = jsonl(run / "validation_log.jsonl") if run else []
    if not rows:
        return
    tokens_per_step = json.loads((run / "manifest.json").read_text())["tokens_per_optimizer_step"]
    fig, ax = plt.subplots(figsize=(8, 4.2), facecolor=SURFACE)
    for lang in ("hi", "bn", "ta"):
        xs = [r["step"] * tokens_per_step / 1e6 for r in rows]
        ys = [r[lang]["loss"] for r in rows]
        ax.plot(xs, ys, color=LANG_COLOR[lang], linewidth=2, label=LANG_NAME[lang])
        ax.annotate(f"{LANG_NAME[lang]} {ys[-1]:.2f}", (xs[-1], ys[-1]), xytext=(6, 0), textcoords="offset points",
                    va="center", fontsize=9, color=INK)
    style(ax, "Base pretraining: held-out validation loss by language", "training tokens seen (millions)", "cross-entropy (nats/token)")
    ax.legend(frameon=False, loc="upper right")
    fig.tight_layout()
    fig.savefig(OUT / "1_pretraining_validation_loss.png", dpi=150)
    plt.close(fig)


def ppo_figure():
    run = latest_run("ppo")
    rows = jsonl(run / "ppo_log.jsonl") if run else []
    if not rows:
        return
    manifest = json.loads((run / "manifest.json").read_text())
    dev = manifest.get("dev_history", [])
    fig, axes = plt.subplots(1, 3, figsize=(14, 4), facecolor=SURFACE)
    it = [r["iteration"] for r in rows]
    axes[0].plot(it, [r["reward_raw_by_intent"].get("harmful", float("nan")) for r in rows], color="#2a78d6", linewidth=2, label="harmful prompts")
    axes[0].plot(it, [(r["reward_raw_by_intent"].get("sensitive_benign", 0) + r["reward_raw_by_intent"].get("ordinary_benign", 0)) / 2 for r in rows],
                 color="#eb6834", linewidth=2, label="benign prompts")
    style(axes[0], "Reward-model score of rollouts", "PPO iteration", "mean raw score")
    axes[0].legend(frameon=False)
    axes[1].plot(it, [r["kl_seq_mean"] for r in rows], color="#2a78d6", linewidth=2)
    style(axes[1], "KL from frozen SFT reference", "PPO iteration", "nats per response")
    if dev:
        xs = [d["iteration"] for d in dev]
        safe = [sum(d[l]["safe_handling"] for l in ("hi", "bn", "ta")) / 3 for d in dev]
        fr = [sum(d[l]["false_refusal"] for l in ("hi", "bn", "ta")) / 3 for d in dev]
        axes[2].plot(xs, safe, color="#2a78d6", linewidth=2, marker="o", markersize=5, label="safe handling of harmful")
        axes[2].plot(xs, fr, color="#eb6834", linewidth=2, marker="o", markersize=5, label="false refusal of benign")
        best = manifest.get("best", {}).get("iteration")
        if best:
            axes[2].axvline(best, color=INK2, linewidth=1, linestyle="--")
            axes[2].annotate("selected", (best, 1.0), xytext=(4, -12), textcoords="offset points", fontsize=8, color=INK2)
        axes[2].set_ylim(0, 1.05)
        style(axes[2], "Development set (auto-labelled)", "PPO iteration", "rate (mean of 3 languages)")
        axes[2].legend(frameon=False, loc="center right")
    fig.tight_layout()
    fig.savefig(OUT / "2_ppo_training.png", dpi=150)
    plt.close(fig)


def eval_figure():
    path = ROOT / "artifacts/eval/latest/summary.json"
    if not path.exists():
        return
    s = json.loads(path.read_text())
    cks = s["checkpoints"]
    panels = [("harmful", "unsafe_generation", "Unsafe generation on harmful prompts (lower is better)"),
              ("sensitive_benign", "false_refusal", "False refusal on sensitive-but-benign prompts (lower is better)")]
    fig, axes = plt.subplots(1, 2, figsize=(14, 4.4), facecolor=SURFACE)
    width = 0.8 / len(cks)
    for ax, (intent, key, title) in zip(axes, panels):
        for j, ck in enumerate(cks):
            vals = []
            for lang in ("hi", "bn", "ta"):
                entry = s["summary"][ck][lang][intent]
                vals.append(100 * (entry[key]["rate"] or 0) if "status" not in entry else 0)
            xs = [i + (j - (len(cks) - 1) / 2) * width for i in range(3)]
            bars = ax.bar(xs, vals, width=width - 0.03, color=CK_COLOR.get(ck, INK2), label=CK_NAME.get(ck, ck))
            for b, v in zip(bars, vals):
                ax.annotate(f"{v:.0f}", (b.get_x() + b.get_width() / 2, v), xytext=(0, 2), textcoords="offset points",
                            ha="center", fontsize=7, color=INK2)
        ax.set_xticks(range(3), [LANG_NAME[l] for l in ("hi", "bn", "ta")])
        ax.set_ylim(0, 105)
        style(ax, title, "", "% of valid responses")
    axes[0].legend(frameon=False, ncol=2, loc="upper right", fontsize=8)
    fig.text(0.01, 0.01, "Automatic proxy labels on the frozen held-out test prompts; counts and intervals in summary.md.",
             fontsize=8, color=INK2)
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    fig.savefig(OUT / "3_heldout_safety_comparison.png", dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    pretrain_figure()
    ppo_figure()
    eval_figure()
    print("figures:", sorted(p.name for p in OUT.glob("*.png")))
    sys.exit(0)
