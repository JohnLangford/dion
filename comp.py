import json
from pathlib import Path

import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

RESULTS = Path("results")
PLOT_METRICS = ["gpu_opt_ms", "cpu_opt_ms", "dt_ms"]  # metrics to plot


def stacked_fig(title, fname, xlabel, draw, rotate_x=False):
    """One column of len(PLOT_METRICS) panels; `draw(ax, metric, i)` fills each."""
    fig, axes = plt.subplots(len(PLOT_METRICS), 1, figsize=(10, 11), sharex=True)
    for i, (ax, m) in enumerate(zip(axes, PLOT_METRICS)):
        draw(ax, m, i)
        ax.set_xlabel("")
    axes[-1].set_xlabel(xlabel)
    if rotate_x:
        axes[-1].tick_params(axis="x", rotation=30)
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(RESULTS / fname, dpi=150)
    plt.close(fig)


# Load every per-run metrics JSON ({"config": {...}, "steps": [...]}) at
# results/<size>/<family>/<condition>/active_step_metrics_<ts>.json. The first
# three dir levels become columns; the whole config is merged onto each row.
frames = []
config_keys = None
metric_cols = None
for p in sorted(RESULTS.rglob("*.json")):
    rel = p.relative_to(RESULTS)
    if len(rel.parts) != 4:  # expect exactly size/family/condition/<file>.json
        continue
    size, family, condition = rel.parts[:3]
    ts = p.stem.replace("active_step_metrics_", "")
    doc = json.loads(p.read_text())
    steps = pd.DataFrame(doc["steps"])
    if config_keys is None:
        config_keys = list(doc["config"])
        metric_cols = [c for c in steps.columns if c != "step"]
    frames.append(steps.assign(
        size=size, family=family, condition=condition, datetime=ts, **doc["config"]
    ))

if not frames:
    raise SystemExit("No metrics JSONs found under results/<size>/<family>/<condition>/")

df = pd.concat(frames, ignore_index=True)
df.to_csv(RESULTS / "df.tsv", sep="\t", index=False)

# configs.tsv: one row per file, only config columns that differ somewhere.
ids = ["size", "family", "condition", "datetime"]
per_file = df.drop_duplicates(subset=ids)
varying = [k for k in config_keys if per_file[k].nunique(dropna=False) > 1]
per_file[ids + varying].sort_values(ids).to_csv(RESULTS / "configs.tsv", sep="\t", index=False)

# One violin / line / median dataframe per (size, family).
sns.set_theme(style="whitegrid")
grouped = df.groupby(["size", "family"], sort=False)
for (size, family), g in grouped:
    name = f"{size}_{family}"
    conds = sorted(g["condition"].unique())

    # One median dataframe: absolute median per condition + ratio of those
    # medians to the baseline condition (first sorted), interleaved as <metric>
    # and <metric>_ratio columns.
    med = g.groupby("condition")[metric_cols].median().reindex(conds)
    ratio = med.div(med.loc[conds[0]])
    combined = pd.DataFrame(index=med.index)
    for m in metric_cols:
        combined[m] = med[m]
        combined[f"{m}_ratio"] = ratio[m]
    combined.to_csv(RESULTS / f"{name}_median.tsv", sep="\t", index=True)

    stacked_fig(
        f"Per-step timing distributions — {name}", f"{name}_violin.png", "condition",
        lambda ax, m, i: sns.violinplot(data=g, x="condition", y=m, order=conds, ax=ax),
        rotate_x=True,
    )
    stacked_fig(
        f"Per-step timing vs step — {name}", f"{name}_lines.png", "active step index",
        lambda ax, m, i: sns.lineplot(data=g, x="step", y=m, hue="condition",
                                      hue_order=conds, marker=".", ax=ax, legend=(i == 0)),
    )

print(f"Processed {len(frames)} files into {grouped.ngroups} (size, family) group(s). "
      f"Wrote <size>_<family>_median.tsv/_violin.png/_lines.png + df.tsv + "
      f"configs.tsv under results/.")
