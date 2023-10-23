import numpy as np
import matplotlib.pyplot as plt

# input_name = "data/sgd_legendre10_t10_optimal1_Sinf_sconstant_i10000_quasi.npz"
# output_name = "plots/optimal_recovery.png"
# trendline = False
# teaser = False

# input_name = "data/sgd_legendre10_t10_uniform1_Sinf_sconstant_i10000_quasi.npz"
# output_name = "plots/uniform_recovery.png"
# trendline = False
# teaser = False

# input_name = "data/sgd_legendre10_t15_optimal1_Sinf_sdeterministic_unbounded_i10000_quasi.npz"
# output_name = "plots/optimal_nonrecovery.png"
# trendline = True
# teaser = False

# input_name = "data/sgd_legendre10_t15_uniform1_Sinf_sdeterministic_unbounded_i10000_quasi.npz"
# output_name = "plots/uniform_nonrecovery.png"
# trendline = True
# teaser = False

# input_name = "data/sgd_legendre10_t15_optimal50_Sinf_sadaptive_i10000_quasi.npz"
# output_name = "plots/optimal_nonrecovery_adaptive.png"
# trendline = True
# teaser = False

input_name = "data/sgd_legendre10_t15_optimal50_Sinf_sadaptive_i10000_quasi.npz"
output_name = "plots/optimal_nonrecovery_adaptive_teaser.png"
trendline = True
teaser = True

yellow = "#D4D6C0"
blue = "#78AAD6"
red = "#FF7E79"
black = "#3B3B3B"
mud = "#A8AA89"
plt.rcParams.update({
    "lines.color": yellow,
    "patch.edgecolor": yellow,
    "axes.edgecolor": yellow,
    "axes.labelcolor": yellow,
    "xtick.color": yellow,
    "ytick.color": yellow,
})

z = np.load(input_name)
losses = z["losses"]
minimal_loss = z["minimal_loss"]
step_sizes = z["step_sizes"]

if teaser:
    fig, ax = plt.subplots(1, 1, figsize=(2, 2.25))
else:
    fig, ax = plt.subplots(1, 1, figsize=(4, 2.25))

steps = np.arange(1, len(losses)+1)
ax.loglog(steps, losses - minimal_loss, color=blue, label="loss")
ax.loglog(steps, step_sizes, color=red, label="step size")
if trendline:
    ax.loglog(steps, 1 / steps, "--", color=mud, label="$t^{-1}$ rate")
if teaser:
    ax.set_xticks([])
    ax.set_xticks([], minor=True)
    ax.set_yticks([])
    ax.set_yticks([], minor=True)
else:
    ax.set_xlabel("step")
ax.set_xlim(steps[0], steps[-1])
ax.legend(loc="lower left", facecolor=black, edgecolor=yellow, labelcolor=yellow)

plt.tight_layout()
plt.savefig(output_name, dpi=600, transparent=True)
