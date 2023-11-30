# coding: utf-8
from pathlib import Path

import numpy as onp
import matplotlib.pyplot as plt

from mascotnum_plots import set_style

root = Path("shallow_sgd_experiments/data")
stem = "rate_experiment_NGD_projection_samples-optimal_steps-adaptive_threshold_width-{width}-relu_1.npz"

widths = onp.array([5, 10, 15, 20, 25, 30, 35, 40, 50, 75, 100])
losses = []
for width in widths:
    data_path = root / stem.format(width=width)
    z = onp.load(data_path)
    losses.append(z["losses"][-1])

plot_path = root.parent / "plot" / "shallow_sgd_rate.pdf"

textcolor, legendcolor, color_1, color_2, color_3 = set_style("bimos")
fig, ax = plt.subplots(1, 1, figsize=(4, 2.25))
ax.plot(widths, losses, "o", markersize=6, color=color_1, markeredgewidth=1.5, label="Loss")
ax.plot(widths, 1e-1 / widths**3, "--", color=color_2, label="$w^{-3}$ rate", zorder=-1)
ax.legend()
ax.set_xscale("log")
ax.set_yscale("log")
ax.set_xlabel("Width ($w$)")
ax.set_title(r"Approximating $\sin(2\pi x)$ on $[0, 1]$ by shallow ReLU NNs")
print(f"Saving: {plot_path}")
plt.savefig(
    plot_path, dpi=600, edgecolor="none", bbox_inches="tight", transparent=True
)
