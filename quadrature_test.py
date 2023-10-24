import numpy as np
from scipy.integrate import quad, dblquad

import matplotlib.pyplot as plt


b = np.array([1, 1])
e1 = np.array([1, -1])

Q = np.array([b, e1]).T
assert np.all(Q == np.array([[1, 1], [1, -1]]))

Qinv = 0.5 * Q
assert np.allclose(Qinv @ Q, np.eye(2))

def f(x1, x2):
    x = np.array([x1, x2])
    return b @ x


reference_value, reference_error = dblquad(f, 0, 1, 0, 1)
print(f"Reference value: {reference_value:.1f}  (error: {reference_error:.2e})")

# y1_lb = np.sum(np.minimum(b, 0))
# y1_ub = np.sum(np.maximum(b, 0))
# assert y1_lb == 0 and y1_ub == 2
y1_lb = 0
y1_ub = 1


def y2_bounds(y1):
    # Consider the element-wise inclusion
    #   z * e1 in [-y1 * b, 1 - y1 * b] .
    # If e1 is positive, then we can divide by e1 and obtain a sequence of bounds.
    # Taking the maximum of all lower bounds provides a lower bound for z.
    # Taking the minimum of all upper bounds provides a upper bound for z.
    # If e1 is negative, then we have to swap the upper and lower bounds.
    assert not np.any(e1 == 0)  # just not implemented ...
    positive_e1 = e1 > 0
    negative_e1 = e1 < 0
    lb = np.max(-y1 * b[positive_e1] / e1[positive_e1])
    ub = np.min((1 - y1 * b[positive_e1]) / e1[positive_e1])
    lb = max(lb, np.max((1 - y1 * b[negative_e1]) / e1[negative_e1]))
    ub = min(ub, np.min(-y1 * b[negative_e1] / e1[negative_e1]))
    assert lb <= ub
    return lb, ub


# y1s = np.linspace(y1_lb, y1_ub, 1000)
# y2_lbs = [y2_bounds(y1)[0] for y1 in y1s]
# y2_ubs = [y2_bounds(y1)[1] for y1 in y1s]
# plt.plot(y1s, y2_lbs)
# plt.plot(y1s, y2_ubs)
# plt.show()


def g(y1, y2):
    y = np.array([y1, y2])
    Qy = Q @ y
    assert np.allclose(Qy, y1 * b + y2 * e1)
    assert np.all(-1e-8 <= Qy), (y, Qy)
    assert np.all(Qy <= 1 + 1e-8)
    return f(Qy[0], Qy[1])


transformed_value, transformed_error = dblquad(lambda y2, y1: g(y1, y2), y1_lb, y1_ub, lambda y1: y2_bounds(y1)[0], lambda y1: y2_bounds(y1)[1])
detQ = -2
assert np.allclose(detQ, np.linalg.det(Q))
print(f"Transformed value: {abs(detQ) * transformed_value:.1f}  (error: {abs(detQ) * transformed_error:.2e})")

def y2_interval_length(y1):
    return np.diff(y2_bounds(y1))


dim1_value, dim1_error = quad(lambda y1: g(y1, 0) * y2_interval_length(y1), y1_lb, y1_ub)
print(f"1D value: {abs(detQ) * dim1_value:.1f}  (error: {abs(detQ) * dim1_error:.2e})")
