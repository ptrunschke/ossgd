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


def matrix_pos(matrix):
    es, vs = np.linalg.eigh(matrix)
    es = np.maximum(es, 0)
    return vs * es @ vs.T


def sample(rng, bs, fs, plot=False, sample_indices=None):
    dimension, num_nodes = bs.shape
    rank = fs.shape[0]
    assert fs.shape == (rank, num_nodes)
    if sample_indices is None:
        sample_indices = []
    assert isinstance(sample_indices, list) and len(sample_indices) <= dimension <= R
    if len(sample_indices) == dimension and not plot:
        return sample_indices
    f_factor = fs[:, sample_indices]
    f_projection = f_factor @ np.linalg.solve(f_factor.T @ f_factor, f_factor.T)
    f_projection = matrix_pos(np.eye(rank) - f_projection)
    f_ch = np.einsum("dx, de, ex -> x", fs, f_projection, fs)
    assert np.all(f_ch >= -1e-12)
    f_ch = np.maximum(f_ch, 0)
    b_factor = bs[:, sample_indices]
    b_projection = b_factor @ np.linalg.solve(b_factor.T @ b_factor, b_factor.T)
    b_projection = matrix_pos(np.eye(dimension) - b_projection)
    b_ch = np.einsum("dx, de, ex -> x", bs, b_projection, bs)
    assert np.all(b_ch >= -1e-12)
    b_ch = np.maximum(b_ch, 0)
    f_ch /= np.sum(f_ch)
    alpha = 1e-1
    # f_ch = alpha / num_nodes + (1 - alpha) * f_ch
    f_ch = np.maximum(f_ch, alpha / num_nodes)
    pdf = b_ch / f_ch
    if plot:
        plt.plot(b_ch / np.sum(b_ch))
        plt.plot(f_ch / np.sum(f_ch))
        plt.plot(pdf / np.sum(pdf))
        for idx in sample_indices:
            plt.axvline(idx, color="tab:red")
        plt.title(f"Optimal sampling distribution (step {len(sample_indices)+1})")
        plt.show()
    if len(sample_indices) == dimension:
        return sample_indices
    pdf /= np.sum(pdf)
    sample_indices.append(rng.choice(num_nodes, p=pdf))
    return sample(rng, bs, fs, plot, sample_indices)


if __name__ == "__main__":
    from optimal_least_squares import quasi_optimality_constant, bias_constant, optimal_least_squares

    target = lambda x: np.sin(2 * np.pi * x) + np.cos(2 * dimension * np.pi * x)

    rng = np.random.default_rng(0)
    idcs = sample(rng, bs, fs, plot=True)
    points = xs[idcs]

    c1 = quasi_optimality_constant(points, dimension)
    c2 = bias_constant(points, dimension)
    print(f"Quasi-optimality: {c1:.2f}")
    print(f"Bias: {c2:.2f}")

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

    from tqdm import trange
    trials = 1_000
    samples = []
    for trial in trange(trials):
        samples.append(sample(rng, bs, fs))
    samples = np.concatenate(samples)
    plt.hist(xs[samples], density=True, bins=100)
    plt.show()
