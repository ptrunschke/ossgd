# coding: utf-8
from functools import cache

import numpy as onp
import matplotlib.pyplot as plt

c = 0.1
a = -2.
@cache
def xi_bar(t):
    if t == 0:
        return 0
    return t ** a + (1 - t ** (a / 2)) * xi_bar(t-1)

ts = onp.arange(1, 2000)
xis = [xi_bar(t) for t in ts]

plt.loglog(ts, xis, label=r"$\bar\xi_t$")
plt.loglog(ts, onp.log(ts)/ts**(-a/2), label=r"$\log(t)/t^{-1/2}$")
plt.legend()
plt.xlabel("t")
plt.show()
