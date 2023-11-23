# coding: utf-8
"""
Plot the expectation and empirical mean of a loss function.

We present two loss functions for which the expectation is convext,
yet the empirical mean is extremely non-covex.
However, these examples are not L-smooth!
"""
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

import coloring

b = 5

F = lambda z: 2/3 * z**(3/2)
f = lambda z: z**(1/2)        # F'
e = lambda z: 1/z**(1/2)      # F''
NAME = "root"
SALT = 7

# def F(z):
#     with np.errstate(divide='ignore', invalid='ignore'):
#         return np.nan_to_num(z*np.log(z)-z)
# f = lambda z: np.log(z)  # F'
# e = lambda z: 1/z        # F''
# NAME = "log"
# SALT = 2

rng = np.random.default_rng(SALT)

assert np.all(e(np.linspace(0, 2*b, 1000)[1:]) >= 0)  # F is convex on (0, 2*b)

ell = lambda u, z: f(abs(u-z))
d_ell = lambda u, z: e(abs(u-z)) * np.sign(u-z)
J = lambda u: 1/(2*b) * (F(b+u) + F(b-u) - 2*F(0))    # J is convex on (-b, b) since F is convex.

us = np.linspace(-b, b, 1000)
zs = rng.uniform(-b, b, 10)

geometry = {
    'top':    1,
    'bottom': 0,
    'left':   0,
    'right':  1,
    'wspace': 0.25,  # the default as defined in rcParams
    'hspace': 0.25  # the default as defined in rcParams
}
figshape = (1,1)
textwidth = 6.50127  # width of the text in inches
figwidth = textwidth / 2  # width of the figure in inches
phi = (1 + np.sqrt(5))/2
figsize = coloring.compute_figsize(geometry, figshape, aspect_ratio=phi, figwidth=figwidth)
fig,ax = plt.subplots(*figshape, figsize=figsize, dpi=300)

Jhat_us = np.mean([ell(us, z) for z in zs], axis=0)
ax.plot(us, Jhat_us, color='black', lw=1.5)
ylim = ax.get_ylim()

# d_Jhat_us = np.mean([d_ell(us, z) for z in zs], axis=0)
# ax.plot(us, d_Jhat_us, color='gray', lw=0.5)

# for k in range(10):
#     ax.plot(us, ell(us, zs[k]), color='black', lw=0.5)

ax.plot(us, J(us), '--', color='tab:red', lw=2)

plot_path = Path("plots")
plot_path.mkdir(exist_ok=True)
plot_path /= f"convex_expectation_{NAME}.png"

ax.set_xlim(-b, b)
# ax.set_ylim(*ylim)
fig.tight_layout()
plt.savefig(plot_path, dpi=300, facecolor=fig.get_facecolor(), edgecolor='none', bbox_inches="tight")
plt.close()
