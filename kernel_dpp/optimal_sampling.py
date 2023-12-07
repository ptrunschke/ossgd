import pytest
import numpy as np
from rkhs import rkhs_kernel_matrix
from legendre import hk_gramian, orthonormal_basis


def mercer_decomposition(points, rank):
    K = rkhs_kernel_matrix(points)
    es, vs = np.linalg.eigh(K)
    es, vs = es[::-1], vs[:, ::-1]
    assert np.allclose(vs * es @ vs.T, K)
    assert np.all(es[:-1] >= es[1:])
    fs = (vs[:, :rank] * np.sqrt(es[:rank])).T
    assert fs.shape == (rank, len(points))
    return fs


def test_mercer_decomposition():
    xs = np.linspace(-1, 1, 1_000)
    R = len(xs)
    fs = mercer_decomposition(xs, R)
    assert fs.shape == (R, len(xs))
    assert np.allclose(fs.T @ fs, rkhs_kernel_matrix(xs))


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    dimension = 10
    R = 3 * dimension

    xs = np.linspace(-1, 1, 1000)
    fs = mercer_decomposition(xs, R)

    fig, ax = plt.subplots(1, 2)
    sigmas = np.trapz(fs**2, xs, axis=1) / 2
    ax[0].plot(sigmas)
    ax[0].set_yscale("log")
    ax[0].set_title("Eigenvalues of the kernel")
    for k in range(5):
        ax[1].plot(xs, fs[k], label=f"f_{{{k}}}")
    ax[1].legend()
    ax[1].set_title("Eigenfunctions of the kernel")
    plt.show()


    h1_legendre = orthonormal_basis(hk_gramian(dimension, 1))
    bs = h1_legendre(xs, np.eye(dimension))
    assert bs.shape == (dimension, len(xs))

    ch_b = np.sum(bs**2, axis=0)
    ch_b /= np.trapz(ch_b, xs)
    ch_f = np.sum(fs**2, axis=0)
    ch_f /= np.trapz(ch_f, xs)
    ch_bf = ch_b / ch_f
    ch_bf /= np.trapz(ch_bf, xs)
    plt.plot(xs, ch_b)
    plt.plot(xs, ch_f)
    plt.plot(xs, ch_bf)
    plt.show()


if __name__ == "__main__" and __debug__:
    d = 3
    X = np.random.randn(d+2, d)
    det = np.linalg.det(X.T @ X)
    prec_det = 1
    while X.shape[1] > 0:
        x1 = X[:, 0]
        X = X[:, 1:]
        projection = X @ np.linalg.pinv(X.T @ X) @ X.T
        projection_det = x1 @ x1 - x1 @ projection @ x1
        prec_det *= projection_det
        remaining_det = np.linalg.det(X.T @ X)
        assert np.isclose(prec_det * remaining_det, det)
    assert np.allclose(projection, 0)
