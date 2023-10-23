# coding: utf-8
import numpy as np
import scipy.stats as ss


def fourval(x, c, tensor=True):
    """
    Evaluate a Fourier series at points `x`.
    If `c` is of length `n + 1`, this function returns the value:

    .. math:: p(x) = c_0 + \sum_{k=1}^{n} \sqrt{2}\cos(2*\pi*k*x) + \sum_{k=1}^{n} \sqrt{2}\sin(2*\pi*k*x)

    If `c` is a 1-D array, then `p(x)` will have the same shape as `x`.
    If `c` is multidimensional, then the shape of the result depends on the value of `tensor`.
    If `tensor` is true the shape will be `c.shape[1:] + x.shape`.
    If `tensor` is false the shape will be `c.shape[1:]`.
    Note that scalars have shape `()`.

    Trailing zeros in the coefficients will be used in the evaluation, so they should be avoided if efficiency is a concern.

    Note that the chosen Fourier basis is orthonormal in $L^2([0,1], \mathrm{d}x)$.

    Parameters
    ----------
    x : array_like, compatible object
        If `x` is a list or tuple, it is converted to an ndarray, otherwise
        it is left unchanged and treated as a scalar. In either case, `x`
        or its elements must support addition and multiplication with
        with themselves and with the elements of `c`.
    c : array_like
        Array of coefficients ordered so that the coefficients for terms of
        degree n are contained in c[n]. If `c` is multidimensional the
        remaining indices enumerate multiple polynomials. In the two
        dimensional case the coefficients may be thought of as stored in
        the columns of `c`.
    tensor : boolean, optional
        If True, the shape of the coefficient array is extended with ones
        on the right, one for each dimension of `x`. Scalars have dimension 0
        for this action. The result is that every column of coefficients in
        `c` is evaluated for every element of `x`. If False, `x` is broadcast
        over the columns of `c` for the evaluation.  This keyword is useful
        when `c` is multidimensional. The default value is True.
    """
    def basis(x, dim):
        index, kind = divmod(dim, 2)
        return np.sqrt(1+(dim>0)) * ((1-kind)*np.cos(2*np.pi*(index+kind)*x) + kind*np.sin(2*np.pi*(index+kind)*x))
    x = np.asarray(x)
    c = np.asarray(c)
    if tensor:
        c = np.reshape(c, np.shape(c)+(1,)*x.ndim)
    ret = np.zeros(np.broadcast(c, x).shape[1:])
    for dim in range(c.shape[0]):
        ret += c[dim] * basis(x, dim)
    return ret


if __name__ == "__main__":
    import matplotlib as mpl
    import matplotlib.pyplot as plt

    def trapz(xs, density=1):
        assert xs.ndim == 1
        ds = 0.5*np.diff(xs)
        diag = np.zeros(xs.shape[0])
        diag[:-1] += ds
        diag[1:] += ds
        return np.diag(density*diag)

    phi = lambda x: (x+1)/2
    xs = np.linspace(-1, 1, 1000)
    measures = fourval(phi(xs), np.eye(5))
    gramian = measures @ trapz(xs, density=0.5) @ measures.T
    assert np.allclose(gramian, np.eye(5))

    plt.figure()
    for dim in range(measures.shape[0]):
        plt.plot(xs, measures[dim], label=fr"$b_{dim}$")
    plt.xlim(-1,1)
    plt.legend()

    phi = lambda x: (ss.norm.cdf(x)-ss.norm.cdf(-4))/(ss.norm.cdf(4)-ss.norm.cdf(-4))
    xs = np.linspace(-4, 4, 1000)
    measures = fourval(phi(xs), np.eye(5))
    gramian = measures @ trapz(xs, density=ss.norm.pdf(xs)) @ measures.T
    assert np.linalg.norm(gramian - np.eye(5)) < 10*ss.norm.cdf(-4)

    plt.figure()
    plt.plot(xs, ss.norm.pdf(xs), color='xkcd:black', linewidth=2)
    for dim in range(measures.shape[0]):
        plt.plot(xs, measures[dim], label=fr"$b_{dim}$")
    plt.xlim(-4,4)
    plt.legend(loc='lower right')

    plt.show()
