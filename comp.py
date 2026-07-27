import pandas as pd

files = [
    "results/1-plain/active_step_metrics.json",
    "results/2-triton/active_step_metrics.json",
    "results/3-baseline/active_step_metrics.json",
    "results/4-cutlass/active_step_metrics.json",
    "results/5-gns/active_step_metrics.json",
    "results/6-gns-cutlass/active_step_metrics.json",
]

df = pd.concat(
    [pd.read_json(f).assign(file=f.split("/")[1]) for f in files],
    ignore_index=True,
)

# Preserve the condition order from `files`, not alphabetical.
order = [f.split("/")[1] for f in files]

avg = df.groupby("file").median(numeric_only=True).reindex(order)
print(avg)
avg.to_csv("results/avg.csv", index=True)

# Each column normalized by its value in the first row (the baseline condition).
rel = avg / avg.iloc[0]
print(rel)
rel.to_csv("results/avg_relative.csv", index=True)


import matplotlib.pyplot as plt

metrics = ["gpu_opt_ms", "cpu_opt_ms", "dt_ms"]
metrics = ["gpu_opt_ms", "cpu_opt_ms", "dt_ms"]

fig, axes = plt.subplots(len(metrics), 1, figsize=(10, 11), sharex=True)
positions = range(1, len(order) + 1)

for ax, metric in zip(axes, metrics):
    data = [df.loc[df["file"] == c, metric].to_numpy() for c in order]
    ax.violinplot(data, positions=positions, showmedians=True, showextrema=True)
    ax.set_ylabel(metric)
    ax.grid(axis="y", alpha=0.3)

axes[-1].set_xticks(list(positions))
axes[-1].set_xticklabels(order, rotation=30, ha="right")
fig.suptitle("Per-step timing distributions across conditions")
fig.tight_layout()
fig.savefig("results/violins.png", dpi=150)
print("Wrote results/violins.png")

# Per-step line plot: metric vs step index, one line per condition. Use this to
# verify the timed window is on the steady-state plateau (flat lines), not still
# in a warmup transient (a step-down partway through).
fig2, axes2 = plt.subplots(len(metrics), 1, figsize=(10, 11), sharex=True)
for ax, metric in zip(axes2, metrics):
    for c in order:
        d = df.loc[df["file"] == c].sort_values("step")
        ax.plot(d["step"], d[metric], marker=".", ms=4, label=c)
    ax.set_ylabel(metric)
    ax.grid(alpha=0.3)
axes2[0].legend(fontsize=8, ncol=2)
axes2[-1].set_xlabel("active step index")
fig2.suptitle("Per-step timing vs step (check for a flat plateau)")
fig2.tight_layout()
fig2.savefig("results/lines.png", dpi=150)
print("Wrote results/lines.png")


