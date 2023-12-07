import numpy as np
from rkhs import rkhs_kernel_matrix
from legendre import hk_gramian, orthonormal_basis


def dpp_kernel_matrix(points, dimension, basis=None):
    points = np.asarray(points)
    assert points.ndim == 1
    K = rkhs_kernel_matrix(points)
    assert K.shape == (len(points), len(points))
    if basis is None:
        basis = orthonormal_basis(hk_gramian(dimension, 1))
    M = basis(points, np.eye(dimension))
    assert M.shape == (dimension, len(points))
    es, vs = np.linalg.eigh(K)
    assert np.allclose(vs * es @ vs.T, K)
    M = M @ vs
    return M / es @ M.T


def mu_W(points, dimension, basis=None):
    es = np.linalg.eigvalsh(dpp_kernel_matrix(points, dimension, basis))
    es = np.maximum(es, 0)
    return 1 / np.sqrt(np.min(es))


def quasi_optimality_constant(points, dimension, basis=None):
    return 1 + 2 * mu_W(points, dimension, basis)


def mu_p(points, dimension, basis=None):
    if basis is None:
        basis = orthonormal_basis(hk_gramian(dimension, 1))
    M = basis(points, np.eye(dimension))
    return np.linalg.norm(M, ord=-2)


def bias_constant(points, dimension, basis=None):
    return 1 + 2 * mu_p(points, dimension, basis)


def optimal_least_squares(function, points, dimension, basis=None):
    if basis is None:
        basis = orthonormal_basis(hk_gramian(dimension, 1))
    points = np.asarray(points)
    assert points.ndim == 1
    f = function(points)
    assert f.shape == (len(points),)
    M = basis(points, np.eye(dimension))
    assert M.shape == (dimension, len(points))
    K = rkhs_kernel_matrix(points)
    es, vs = np.linalg.eigh(K)
    assert np.allclose(vs * es @ vs.T, K)
    assert np.all(es >= -1e-12)
    es = np.maximum(es, 0)
    K_plus = vs / es @ vs.T
    V = M @ K_plus @ M.T
    v = M @ K_plus @ f
    return np.linalg.solve(V, v)


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    from matplotlib.colors import LogNorm

    dimension = 10
    target = lambda x: np.sin(2 * np.pi * x) + np.cos(2 * dimension * np.pi * x)

    rng = np.random.default_rng(7)
    points = rng.uniform(-1, 1, (2 * dimension,))

    c1 = quasi_optimality_constant(points, dimension)
    c2 = bias_constant(points, dimension)
    print(f"Quasi-optimality: {c1:.2f}")
    print(f"Bias: {c2:.2f}")

    K = rkhs_kernel_matrix(points)
    abs_K_plus = abs(np.linalg.pinv(K))
    abs_K_plus[abs_K_plus < 1e-12 * np.max(abs_K_plus)] = 0
    cmap = plt.colormaps.get_cmap("viridis")
    cmap.set_bad(color="black")
    plt.matshow(abs_K_plus, cmap=cmap, norm=LogNorm())
    plt.colorbar()
    plt.show()

    l2_legendre = orthonormal_basis(hk_gramian(dimension, 0))
    h1_legendre = orthonormal_basis(hk_gramian(dimension, 1))
    xs = np.linspace(-1, 1, 1000)
    c = optimal_least_squares(target, points, dimension, basis=h1_legendre)
    error = target(xs) - h1_legendre(xs, c)
    node_errors = target(points) - h1_legendre(points, c)
    plt.plot(xs, error, "C0-")
    plt.plot(points, node_errors, "C0o")
    c = np.linalg.lstsq(h1_legendre(points, np.eye(dimension)).T, target(points), rcond=None)[0]
    error = target(xs) - h1_legendre(xs, c)
    plt.plot(xs, error, "C1--")
    c = np.linalg.lstsq(l2_legendre(points, np.eye(dimension)).T, target(points), rcond=None)[0]
    error = target(xs) - l2_legendre(xs, c)
    plt.plot(xs, error, "C2-.")
    plt.show()
