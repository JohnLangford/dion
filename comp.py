import json
from pathlib import Path

import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

RESULTS = Path("results")
metrics = ["gpu_opt_ms", "cpu_opt_ms", "dt_ms"]  # metrics to plot / summarize
sns.set_theme(style="whitegrid")

# Each metrics JSON is a {"config": {...}, "steps": [...]} envelope at
# results/<size>/<condition>/active_step_metrics_<ts>.json. Process one file at a
# time: emit its own steps + median dataframes and violin + line plots, named by
# size / condition / datetime. The whole config is merged onto every step row.
frames = []
config_keys = None
metric_cols = None

for p in sorted(RESULTS.rglob("*.json")):
    rel = p.relative_to(RESULTS)
    if len(rel.parts) != 3:  # expect exactly size/condition/<file>.json
        continue
    size, condition = rel.parts[0], rel.parts[1]
    ts = p.stem.replace("active_step_metrics_", "")
    name = f"{size}_{condition}_{ts}"

    with open(p) as f:
        doc = json.load(f)
    steps = pd.DataFrame(doc["steps"])
    if config_keys is None:
        config_keys = list(doc["config"])
        metric_cols = [c for c in steps.columns if c != "step"]

    # Full per-step frame: every config field merged onto each row.
    frame = steps.assign(size=size, condition=condition, datetime=ts, **doc["config"])
    frame.to_csv(RESULTS / f"{name}_steps.tsv", sep="\t", index=False)
    frames.append(frame)

    # One-row median summary of this run's timings.
    med = steps[metric_cols].median().to_frame().T
    med.insert(0, "datetime", ts)
    med.insert(0, "condition", condition)
    med.insert(0, "size", size)
    med.to_csv(RESULTS / f"{name}_median.tsv", sep="\t", index=False)

    # Violin: this run's per-step distribution for each metric.
    fig, axes = plt.subplots(len(metrics), 1, figsize=(5, 9))
    for ax, m in zip(axes, metrics):
        sns.violinplot(data=steps, y=m, ax=ax)
    fig.suptitle(name)
    fig.tight_layout()
    fig.savefig(RESULTS / f"{name}_violin.png", dpi=150)
    plt.close(fig)

    # Lines: each metric vs step (check for a flat steady-state plateau).
    fig, axes = plt.subplots(len(metrics), 1, figsize=(8, 9), sharex=True)
    for ax, m in zip(axes, metrics):
        sns.lineplot(data=steps, x="step", y=m, marker=".", ax=ax)
    axes[-1].set_xlabel("active step index")
    fig.suptitle(name)
    fig.tight_layout()
    fig.savefig(RESULTS / f"{name}_lines.png", dpi=150)
    plt.close(fig)

if not frames:
    raise SystemExit("No metrics JSONs found under results/<size>/<condition>/")

df = pd.concat(frames, ignore_index=True)
df.to_csv(RESULTS / "df.tsv", sep="\t", index=False)

# configs: one row per file, keeping only config columns that differ somewhere,
# so it's easy to see (per size/condition) how each file's config varies.
per_file = df.drop_duplicates(subset=["size", "condition", "datetime"])
varying = [k for k in config_keys if per_file[k].nunique(dropna=False) > 1]
configs = (
    per_file[["size", "condition", "datetime"] + varying]
    .sort_values(["size", "condition", "datetime"])
)
print(configs.to_string(index=False))
configs.to_csv(RESULTS / "configs.tsv", sep="\t", index=False)

print(f"\nProcessed {len(frames)} files. Per-file *_steps.tsv/_median.tsv/_violin.png/"
      f"_lines.png + df.tsv + configs.tsv written under results/.")
