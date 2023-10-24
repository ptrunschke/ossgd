import numpy as np
from numpy.polynomial import Legendre
from scipy import integrate


def compute_exponential_coefficients(degree):
    factors = np.sqrt(2 * np.arange(degree + 1) + 1)
    coeffs = np.zeros(degree + 1)
    for k in range(degree + 1):
        l = factors[k] * Legendre.basis(k)
        val, _ = integrate.quadrature(lambda x: 0.5 * l(x) * np.exp(x), -1.0, 1.0, tol=5e-16, rtol=5e-16)
        coeffs[k] = val
    return coeffs
