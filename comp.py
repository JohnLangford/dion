import re
from pathlib import Path

import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

RESULTS = Path("results")

# Recursively load every per-run metrics JSON. Layout is
# results/<size>/<condition>/active_step_metrics.json, so the first dir level is
# the model size and the second is the condition -- both become columns.
frames = []
for p in sorted(RESULTS.rglob("*.json")):
    rel = p.relative_to(RESULTS)
    if len(rel.parts) != 3:  # expect exactly size/condition/<file>.json
        continue
    size, condition = rel.parts[0], rel.parts[1]
    frames.append(pd.read_json(p).assign(size=size, condition=condition))

if not frames:
    raise SystemExit("No metrics JSONs found under results/<size>/<condition>/")

df = pd.concat(frames, ignore_index=True)

metrics = ["gpu_opt_ms", "cpu_opt_ms", "dt_ms"]

# Ordering: sizes by their leading number (1b, 3b, ... 14b), conditions sorted.
def size_key(s):
    m = re.match(r"[\d.]+", s)
    return float(m.group()) if m else float("inf")

sizes = sorted(df["size"].unique(), key=size_key)
conditions = sorted(df["condition"].unique())

# Median over steps, grouped by BOTH size and condition. Drop the per-step index
# column, whose median is meaningless.
median = df.drop(columns="step").groupby(["size", "condition"]).median(numeric_only=True)
print(median)
median.to_csv("results/median.csv", index=True)

# Within each size, normalize by that size's own baseline (its first available
# condition in sorted order) -- sizes don't all share the same conditions.
baseline_rows = (
    median.reset_index()
    .sort_values("condition")
    .groupby("size")[list(median.columns)]
    .first()
)
median_relative = median.div(baseline_rows, level="size")
print(median_relative)
median_relative.to_csv("results/median_relative.csv", index=True)

# One violin figure and one line figure per size that we have data for.
sns.set_theme(style="whitegrid")

for size in sizes:
    sub = df[df["size"] == size]
    conds = [c for c in conditions if c in sub["condition"].unique()]

    # Violins: per-step distribution of each metric across conditions.
    fig, axes = plt.subplots(len(metrics), 1, figsize=(10, 11), sharex=True)
    for ax, metric in zip(axes, metrics):
        sns.violinplot(data=sub, x="condition", y=metric, order=conds, ax=ax)
        ax.set_xlabel("")
    axes[-1].set_xlabel("condition")
    axes[-1].tick_params(axis="x", rotation=30)
    fig.suptitle(f"Per-step timing distributions — {size}")
    fig.tight_layout()
    fig.savefig(f"results/violins_{size}.png", dpi=150)
    plt.close(fig)

    # Lines: metric vs step index, one line per condition (check for a flat
    # steady-state plateau vs a warmup transient).
    fig, axes = plt.subplots(len(metrics), 1, figsize=(10, 11), sharex=True)
    for i, (ax, metric) in enumerate(zip(axes, metrics)):
        sns.lineplot(
            data=sub, x="step", y=metric, hue="condition", hue_order=conds,
            marker=".", ax=ax, legend=(i == 0),
        )
    axes[-1].set_xlabel("active step index")
    fig.suptitle(f"Per-step timing vs step — {size}")
    fig.tight_layout()
    fig.savefig(f"results/lines_{size}.png", dpi=150)
    plt.close(fig)

print(f"Wrote violins_<size>.png and lines_<size>.png for: {', '.join(sizes)}")
