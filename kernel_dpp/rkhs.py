import pytest
import numpy as np
from legendre import legval


def rkhs_kernel(x, y):
    domain = (-1, 1)
    low = np.minimum(x, y)
    high = np.maximum(x, y)
    return np.cosh(low - domain[0]) * np.cosh(domain[1] - high) * (domain[1] - domain[0]) / np.sinh(domain[1] - domain[0])


def rkhs_kernel_matrix(xs):
    return rkhs_kernel(xs[:, None], xs[None, :])


@pytest.fixture(params=[-0.75, -0.1, 0.3, 1.0])
def test_point(request):
    return request.param


@pytest.fixture(params=[10])
def test_dimension(request):
    return request.param


def test_rkhs_kernel_matrix():
    xs = np.linspace(-1, 1, 1000)
    K = rkhs_kernel_matrix(xs)
    assert K.shape == (len(xs), len(xs))
    assert np.allclose(K, K.T)
    assert np.all(np.linalg.eigvalsh(K) >= 0)


def test_rkhs_kernel(test_point, test_dimension):
    xs = np.linspace(-1, 1, 1000)
    ks = rkhs_kernel(xs, test_point)
    assert ks.shape == (len(xs),)
    measures = legval(xs, np.eye(test_dimension))
    assert measures.shape == (test_dimension, len(xs))

    l2_inner = 0.5 * np.trapz(measures * ks, xs, axis=1)
    assert l2_inner.shape == (test_dimension,)
    h = np.diff(xs)[0]
    assert np.allclose(np.diff(xs), h)
    d_ks = np.diff(ks) / h
    d_measures = np.diff(measures, axis=1) / h
    h10_inner = 0.5 * np.sum(d_measures * d_ks * h, axis=1)
    assert h10_inner.shape == (test_dimension,)

    inner = l2_inner + h10_inner
    tol = 1 / len(xs)
    assert np.allclose(inner, legval(test_point, np.eye(test_dimension)), atol=tol)


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    xs = np.linspace(-1, 1, 1000)
    for c in [-0.75, -0.1, 0.3, 1]:
        plt.plot(xs, rkhs_kernel(xs, c))
    plt.show()
