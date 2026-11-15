"""Aggregate benchmark JSONL files into markdown tables (with uncertainty)."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from kinetixrt.bench.stats import mcnemar_exact, paired_ratio_ci, spearman, wilson


def load(paths: list[str]) -> list[dict]:
    rows = []
    for p in paths:
        for line in Path(p).read_text().splitlines():
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return rows


def closed_loop_table(rows: list[dict], baseline: str) -> str:
    eps = [r for r in rows if r.get("kind") == "episode"]
    latest: dict[tuple, dict] = {}
    for r in eps:
        latest[(r["controller"], r["task"], r["init"])] = r
    by: dict[str, dict[tuple, dict]] = defaultdict(dict)
    for (c, t, i), r in latest.items():
        by[c][(t, i)] = r
    if not by:
        return "_no episodes_\n"
    out = [
        "| controller | episodes | success (95% CI) | inferences/ep | denoise steps/ep | mean steps | model s/ep (wall) | steps to success | deadline-miss ticks/ep | mean action age (ticks) | smoothness |",
        "|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    order = sorted(by, key=lambda c: (c != baseline, c))
    for c in order:
        rs = list(by[c].values())
        n, k = len(rs), sum(r["success"] for r in rs)
        lo, hi = wilson(k, n)
        succ_steps = [r["steps"] for r in rs if r["success"]]
        ages = [r["mean_action_age_ticks"] for r in rs if r["mean_action_age_ticks"] is not None]
        sm = [r["smoothness"] for r in rs if r["smoothness"] is not None]
        out.append(
            f"| `{c}` | {n} | {100 * k / n:.1f}% ({100 * lo:.0f}-{100 * hi:.0f}) | {np.mean([r['inferences'] for r in rs]):.1f} | "
            f"{np.mean([r['denoise_steps_total'] for r in rs]):.0f} | {np.mean([r['mean_steps'] for r in rs if r['mean_steps']]):.1f} | "
            f"{np.mean([r['model_wall_s_total'] for r in rs]):.1f} | {np.mean(succ_steps) if succ_steps else float('nan'):.0f} | "
            f"{np.mean([r['deadline_miss_ticks'] for r in rs]):.1f} | {np.mean(ages) if ages else float('nan'):.1f} | {np.mean(sm) if sm else float('nan'):.3f} |"
        )
    if baseline in by:
        out += ["", f"Paired comparison against `{baseline}` (same task, initial state and noise seed):", ""]
        out += [
            "| controller | pairs | success diff (pp) | wins / losses | McNemar p | inference ratio (95% CI) | denoise-step ratio (95% CI) | model-time ratio (95% CI) |",
            "|---|---|---|---|---|---|---|---|",
        ]
        for c in order:
            if c == baseline:
                continue
            keys = sorted(set(by[c]) & set(by[baseline]))
            if not keys:
                continue
            a = [by[c][k] for k in keys]
            b = [by[baseline][k] for k in keys]
            wins = sum(x["success"] and not y["success"] for x, y in zip(a, b))
            losses = sum(y["success"] and not x["success"] for x, y in zip(a, b))
            diff = 100 * (sum(x["success"] for x in a) - sum(y["success"] for y in b)) / len(keys)

            def ratio(field: str) -> str:
                r, lo, hi = paired_ratio_ci(np.array([x[field] for x in a]), np.array([y[field] for y in b]))
                return f"{r:.2f} ({lo:.2f}-{hi:.2f})"

            out.append(
                f"| `{c}` | {len(keys)} | {diff:+.1f} | {wins} / {losses} | {mcnemar_exact(wins, losses):.2f} | "
                f"{ratio('inferences')} | {ratio('denoise_steps_total')} | {ratio('model_wall_s_total')} |"
            )
    return "\n".join(out) + "\n"


def signals_report(rows: list[dict]) -> str:
    s = [r for r in rows if r.get("kind") == "sample"]
    if not s:
        return "_no samples_\n"
    eps = {(r["task"], r["init"]): r["episode_success"] for r in s}
    out = [f"{len(s)} samples from {len(eps)} episodes ({sum(eps.values())} successful).", ""]

    def col(name: str) -> np.ndarray:
        return np.array([r.get(name, np.nan) if r.get(name) is not None else np.nan for r in s], dtype=float)

    def sig(name: str, sub: str | None = None) -> np.ndarray:
        vals = []
        for r in s:
            v = r["signals"].get(name)
            if isinstance(v, dict):
                v = v.get(sub)
            vals.append(np.nan if v is None else v)
        return np.array(vals, dtype=float)

    tte = col("ticks_to_event_new")
    near = np.where(np.isnan(tte), 0.0, np.clip(1.0 - tte / 25.0, 0.0, 1.0))
    tse = sig("ticks_since_event")
    preds = {
        "event proximity (1 - ticks_to_event/25)": near,
        "ticks since last event (neg.)": -np.where(np.isnan(tse), 999, tse),
        "planned speed": col("speed_new"),
        "state step (proprio speed)": sig("state_step"),
        "image step: scene cam": sig("img_step", "image"),
        "image step: wrist cam": sig("img_step", "image2"),
        "sample disagreement (2 noise draws)": col("sample_dis"),
    }
    drift_keys = sorted({k for r in s for k in r if k.startswith("drift_") and not k.startswith("drift_grip")}, key=lambda k: int(k.split("_")[1]))
    targets = {"sens3 (3 vs 10 steps)": col("sens3"), "sens5 (5 vs 10 steps)": col("sens5")}
    for k in drift_keys[:4]:
        targets[f"{k} (old plan vs fresh)"] = col(k)
    out += ["Spearman rank correlation between runtime signals (rows) and measured error proxies (columns):", ""]
    out += ["| signal | " + " | ".join(targets) + " |", "|---|" + "---|" * len(targets)]
    for pn, pv in preds.items():
        cells = []
        for tv in targets.values():
            m = ~np.isnan(pv) & ~np.isnan(tv)
            cells.append(f"{spearman(pv[m], tv[m]):+.2f}" if m.sum() >= 10 else "n/a")
        out.append(f"| {pn} | " + " | ".join(cells) + " |")
    out += ["", "Error proxies by plan phase (mean translational action error, action units; n in brackets):", ""]
    post = ~np.isnan(tse) & (tse <= 8)
    phases = {
        "event within 12 ticks": (~np.isnan(tte)) & (tte <= 12),
        "event 13-25 ticks away": (~np.isnan(tte)) & (tte > 12) & (tte <= 25),
        "no event within 25 ticks": np.isnan(tte) | (tte > 25),
        "within 8 ticks after an event": post,
    }
    names = ["sens3", "sens5", "sample_dis"] + drift_keys[:4]
    out += ["| phase | n | " + " | ".join(names) + " | grip flip mismatch 3-vs-10 |", "|---|---|" + "---|" * (len(names) + 1)]
    for pn, m in phases.items():
        cells = []
        for nm in names:
            v = col(nm)[m]
            v = v[~np.isnan(v)]
            cells.append(f"{v.mean():.3f}" if len(v) else "n/a")
        g = col("sens3_grip")[m]
        out.append(f"| {pn} | {int(m.sum())} | " + " | ".join(cells) + f" | {np.nanmean(g) if m.sum() else float('nan'):.3f} |")
    return "\n".join(out) + "\n"


def run(args: argparse.Namespace) -> None:
    rows = load(args.inputs)
    text = signals_report(rows) if args.kind == "signals" else closed_loop_table(rows, args.baseline)
    if args.out:
        Path(args.out).write_text(text)
    print(text)


def add_parser(sub: argparse._SubParsersAction) -> None:
    p = sub.add_parser("report", help="aggregate result JSONL files into markdown tables")
    p.add_argument("inputs", nargs="+")
    p.add_argument("--kind", choices=["closed-loop", "signals"], default="closed-loop")
    p.add_argument("--baseline", default="fixed_high")
    p.add_argument("--out", default=None)
    p.set_defaults(func=run)
