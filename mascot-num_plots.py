import argparse
import tomllib
from pathlib import Path
import re
import subprocess

import numpy as np
import matplotlib.pyplot as plt

from coloring import mix, bimosblack, bimosred, bimosyellow


def set_style(style):
    global textcolor, legendcolor, color_1, color_2, color_3
    if style == "keynote":
        yellow = "#D4D6C0"
        black = "#3B3B3B"
        blue = "#78AAD6"
        red = "#FF7E79"
        mud = "#A8AA89"
        textcolor = yellow
        legendcolor = black
        color_1 = blue
        color_2 = red
        color_3 = mud
    elif style == "bimos":
        textcolor = bimosblack
        legendcolor = bimosyellow
        color_1 = bimosred
        color_2 = bimosblack
        color_3 = mix(bimosblack, 50, bimosyellow).tolist()
    else:
        raise NotImplementedError(f"Unkonwn style '{style}'")
    plt.rcParams.update({
        "lines.color": textcolor,
        "patch.edgecolor": textcolor,
        "axes.edgecolor": textcolor,
        "axes.labelcolor": textcolor,
        "xtick.color": textcolor,
        "ytick.color": textcolor,
        "text.usetex": True,
        "text.latex.preamble": r"""
        \usepackage{amsmath}
        \usepackage{bbm}
    """,
    })
# \usepackage{newtxmath}
# mpl.rc("font", size=10, family="Times New Roman")

parser = argparse.ArgumentParser(
    description="Create a convergence plots for MASCOT-NUM and SMAI-SIGMA.",
)
parser.add_argument("config", help="config file containing the parameters")
args = parser.parse_args()

config_file = Path(args.config)
print(f"Config: {config_file}")
with open(args.config, "rb") as f:
    plots = tomllib.load(f)


def load_data(data_path):
    data_path = Path(data_path)
    # if not data_path.exists():
    #     assert len(data_path.parents) == 3 and data_path.parents[0] == "data"
    #     if data_path.parent == "compact_domain":
    #         script = "linear_sgd.py"
    #     elif data_path.parent == "unbounded_domain":
    #         script = "linear_sgd_hermite.py"
    #     else:
    #         raise NotImplementedError(f"Unknown data path: {data_path}")
    #     assert data_path.suffix == ".npz"
    #     stem = data_path.stem
    #     assert (basis := re.search(r"\d", stem)) is not None
    #     basis, stem = stem[:basis], stem[basis:]
    #     from IPython import embed; embed()
    #     exit()
    #     process = ["python", script, "-b", stem[:]]
    #     data_path = "data/compact_domain/legendre5_sin10_optimal1_Sinf_quasi_10000xmixed.npz"
    #     subprocess.run()
    z = np.load(data_path)
    if "losses" in z:
        losses = z["losses"]
    else:
        losses = z["errors"]
    if "minimal_loss" in z:
        minimal_loss = z["minimal_loss"]
    else:
        minimal_loss = None
    step_sizes = z["step_sizes"]
    return losses, minimal_loss, step_sizes


def plot_data(data_path, plot_path, trendline, teaser, style, relative):
    losses, minimal_loss, step_sizes = load_data(data_path)
    set_style(style)

    if teaser:
        fig, ax = plt.subplots(1, 1, figsize=(2, 2.25))
    else:
        fig, ax = plt.subplots(1, 1, figsize=(4, 2.25))

    steps = np.arange(len(losses))
    if relative:
        assert minimal_loss is not None
        ax.plot(steps, losses - minimal_loss, color=color_1, label=r"$\mathcal{L}(v_t) - \mathcal{L}_{\mathrm{min},\mathcal{M}}$")
        ax.plot(steps[1:], step_sizes, color=color_2, label=r"$s_t$")
        if trendline:
            ax.plot(steps[1:], 1 / steps[1:], "--", color=color_3, label=r"$t^{-1}$ rate")
    else:
        ax.plot(steps, losses, color=color_1, label=r"$\mathcal{L}(v_t)$")
        ax.plot(steps[1:], step_sizes[1:], color=color_2, label=r"$s_t$")
        if trendline:
            ax.plot(steps[1:], 1 / np.sqrt(steps[1:]), "--", color=color_3, label=r"$t^{-1/2}$ rate")
    if teaser:
        ax.set_xticks([])
        ax.set_xticks([], minor=True)
        ax.set_yticks([])
        ax.set_yticks([], minor=True)
    else:
        ax.set_xlabel("step")
    ax.set_xlim(steps[0], steps[-1])
    ax.set_yscale("log")
    ax.set_xscale("symlog", linthresh=1, linscale=0.1)
    xticks = ax.get_xticks()
    xticks = [tick for tick in xticks if tick > 0]
    ax.set_xticks(xticks)
    ax.legend(loc="lower left", facecolor=legendcolor, edgecolor=textcolor, labelcolor=textcolor)

    print(f"Saving: {plot_path}")
    plt.savefig(
        plot_path, dpi=600, edgecolor="none", bbox_inches="tight", transparent=True
    )


default_parameters = plots.pop("*", {})
for label in plots:
    parameters = default_parameters.copy()
    parameters.update(plots[label])
    plot_path = (config_file.parent / label).with_suffix(".pdf")
    plot_data(plot_path=plot_path, **parameters)
