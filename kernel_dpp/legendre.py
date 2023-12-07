import pytest

import numpy as np
from scipy.linalg import solve_triangular
from numpy.polynomial.legendre import Legendre, legval
from joblib import Parallel, delayed


def hk_gramian(dimension, k):
    def L2Inner(p1, p2):
        l2 = (p1*p2).integ()
        return (l2(1) - l2(-1)) / 2

    def entry(i, j):
        ret = 0
        bi, bj = basis(i), basis(j)
        for _ in range(k+1):
            ret += L2Inner(bi, bj)
            bi, bj = bi.deriv(), bj.deriv()
        return ret

    def all_entries():
        for i in range(dimension):
            for j in range(i+1):
                yield delayed(entry)(i, j)

    basis = Legendre.basis
    G = np.zeros((dimension, dimension))
    allEntries = Parallel(n_jobs=15)(all_entries())
    e = 0
    for i in range(dimension):
        for j in range(i+1):
            G[j,i] = G[i,j] = allEntries[e]
            e += 1

    return G


def orthonormal_basis(gramian):
    assert gramian.ndim == 2
    dimension = gramian.shape[0]
    assert gramian.shape[1] == dimension
    L = np.linalg.cholesky(gramian)
    assert np.allclose(L @ L.T, gramian)

    def evaluate_basis(points, coefficients):
        points = np.asarray(points)
        coefficients = np.asarray(coefficients)
        assert points.ndim == 1 and coefficients.ndim <= 2
        assert coefficients.shape[0] <= dimension
        local_L = L[:coefficients.shape[0], :coefficients.shape[0]]
        coefficients = solve_triangular(local_L.T, coefficients, lower=True)
        return legval(points, coefficients)

    return evaluate_basis


@pytest.fixture(params=[3, 5, 10, 30])
def test_gramian_dimension(request):
    return request.param


@pytest.fixture(params=[3, 5, 8])
def test_coefficient_dimension(request):
    return request.param


def test_l2_gramian(test_gramian_dimension):
    dimension = test_gramian_dimension
    gramian = hk_gramian(dimension, 0)
    assert gramian.shape == (dimension, dimension)
    factors = 1 / (2 * np.arange(dimension) + 1)
    assert np.allclose(gramian, np.diag(factors))


def test_l2_legendre(test_gramian_dimension, test_coefficient_dimension):
    dimension = test_gramian_dimension
    local_dimension = test_coefficient_dimension
    if local_dimension > dimension:
        pytest.skip()
    gramian = hk_gramian(dimension, 0)
    legendre = orthonormal_basis(gramian)
    xs = np.linspace(-1, 1, 1000)
    measures = legendre(xs, np.eye(local_dimension))
    assert measures.shape == (local_dimension, len(xs))
    emp_gramian = 0.5 * np.trapz(measures[:, None, :] * measures[None, :, :], xs, axis=-1)
    assert emp_gramian.shape == (local_dimension, local_dimension)
    assert np.allclose(emp_gramian, np.eye(local_dimension), atol=1 / len(xs))


def derivative_operator(dimension):
    basis = Legendre.basis
    operator = np.zeros((dimension, dimension))
    for k in range(dimension):
        c = basis(k).deriv().coef
        operator[:len(c), k] = c
    return operator


def test_h1_gramian(test_gramian_dimension):
    dimension = test_gramian_dimension
    l2_gramian = hk_gramian(dimension, 0)
    h1_gramian = hk_gramian(dimension, 1)
    assert l2_gramian.shape == (dimension, dimension)
    assert h1_gramian.shape == (dimension, dimension)
    D = derivative_operator(dimension)
    reference_gramian = l2_gramian + D.T @ l2_gramian @ D
    assert np.allclose(h1_gramian, reference_gramian)
