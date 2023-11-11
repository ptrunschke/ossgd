import jax
import jax.numpy as jnp
import jax.scipy as jsp

import numpy as onp
import scipy as osp

import matplotlib.pyplot as plt


# # ====================
# # Experiment 1
# # ====================
# target = lambda x: jnp.sin(2 * jnp.pi * x)
# input_dimension = 1
# width = 10
# output_dimension = 1
# activation = lambda x: (jnp.tanh(x) + 1) / 2
# finite_difference = 0
# method = "NGD_quasi_projection"
# sample_size = 10
# sampling = "uniform"
# num_epochs = 10
# epoch_length = 100
# step_size_rule = "constant"
# init_step_size = 0.01
# # NOTE: Running the code with these parameters, we see that
# #         - the algorithm following the gradient flow and
# #         - the algorithm reaches a stationaty point in the manifold with an error of about 8e-6.
# #       To verify these two claims, we can rerun the optimisation with a smaller step size
# #       and check if we obtain the same error curve.
# # TODO: Uncomment the following lines to perform this experiment.
# # num_epochs = 100
# # init_step_size = 0.001
# # TODO: But we reach a smaller error of 3e-6!
# # NOTE: An intuitive idea to increase performance is to use a larger step size in the begining
# #       and then decrease the step size. Interestingly, however, the GD converges to another,
# #       suboptimal stationary point in these first iterations and can not escape this point
# #       when the step size is reduced in later.
# # TODO: Uncomment the following lines to perform this experiment.
# # num_epochs = 10
# # init_step_size = 0.1
# # step_size_rule = "decreasing"
# # limit_epoch = 4
# # NOTE: In the first two experiments, we have seen that the basis functions of the tanh-activation
# #       look like polynomials of bounded degree. Consequently, the optimal sampling density looks
# #       very similar to the Legendre case. This happens due to the random initialisation
# #       and only remains during the optimisation if the approximation remains sufficiently smooth.
# #       This final example, however, produces an approximation with discontinuities
# #       and the optimal density becomes very peaky around the jump points.
# #       This must also happen for the non-continuous target functions for which neural networks
# #       achieve more advantageous approximation rates than classical approximation classes.
# #       (see Experiment 3)


# # ====================
# #     Experiment 2
# # ====================
# target = lambda x: jnp.sin(2 * jnp.pi * x)
# input_dimension = 1
# width = 10
# output_dimension = 1
# # NOTE: Using a larger width means that the approximation error is smaller.
# #       But we observe that with the previously well-chosen step size 0.01,
# #       the parameters converge to a suboptimal stationary point (loss: 1e-1),
# #       which chould also be achieved with width 10 (basis dimension == 3).
# #       Actually, not even width 10 achieves a global minimum,
# #       since the tangent space at the stationary point is still 7-dimensional.
# # TODO: Uncomment the following line to perform this experiment.
# # width = 100
# activation = lambda x: (jnp.tanh(x) + 1) / 2
# finite_difference = 0
# method = "NGD_quasi_projection"
# sample_size = 10
# sampling = "uniform"
# num_epochs = 10
# epoch_length = 100
# step_size_rule = "constant"
# init_step_size = 0.01


# # ====================
# # Experiment 3
# # ====================
# # NOTE: In the preceding experiments, we have seen that the basis functions of the tanh-activation
# #       look like polynomials of bounded degree. Consequently, the optimal sampling density looks
# #       very similar to the Legendre case. But this happens due to the random initialisation
# #       and only remains during the optimisation for the smooth sin target and when the approximation remains smooth.
# #       For the step function target, the optimal density becomes more and more peaky around the jump point.
# target = lambda x: 1e-4 + (x <= (1 / jnp.pi))
# input_dimension = 1
# width = 10
# output_dimension = 1
# activation = lambda x: (jnp.tanh(x) + 1) / 2
# finite_difference = 0
# method = "NGD_quasi_projection"
# sample_size = 10
# sampling = "uniform"
# num_epochs = 20
# step_size_rule = "decreasing"
# limit_epoch = 7
# init_step_size = 0.01
# # NOTE: We can use a larger initial step size but reach another stationary point.
# # TODO: Uncomment the following lines to perform this experiment.
# # init_step_size = 1
# # epoch_length = 100
# # NOTE: The L∞-norm of the inverse Christoffel function becomes extremely large (≈150).
# #       In this situation optimal sampling reduces the variance and speeds up convergence.
# #       Since convergence is faster, we can transition earlyer to an decreasing step size
# #       and also need fever epochs.
# # TODO: Uncomment the following lines to perform this experiment.
# # sampling = "optimal"
# # num_epochs = 10
# # limit_epoch = 4
# # NOTE: Finally, we show that standard SGD can not achieve these speeds, with or without optimal sampling.
# # TODO: Successively uncomment the following two lines to perform these two experiments.
# # method = "SGD"
# # sampling = "uniform"


# # ====================
# # Experiment 4
# # ====================
# # NOTE: Although the SGD in Experiment 3 seems to converge to the same local minimum with or without optimal sampling,
# #       this stationary point is different from the stationary point that is reached with NGD.
# #       These abundance of local minima makes it extremely difficult to compare the two algorihtms
# #       and this can also happen while using the same NGD algorithm but with different sampling methods.
# #       This is demonstrated in this experiment.
# target = lambda x: 1e-4 + (x <= (1 / jnp.pi))
# input_dimension = 1
# width = 10
# output_dimension = 1
# activation = lambda x: (jnp.tanh(x) + 1) / 2
# finite_difference = 0
# method = "NGD_quasi_projection"
# sample_size = 300
# sampling = "uniform"
# # TODO: Uncomment the following line to perform this experiment.
# # sampling = "optimal"
# num_epochs = 15
# step_size_rule = "decreasing"
# limit_epoch = 2
# init_step_size = 1
# epoch_length = 100
# # NOTE: That we are at a stationary point can be seen that the gradient norm converges to zero.
# #       Maybe contrary to the intuition, the dimension of the tangent space is not a valid
# #       indicator of a global stationary point, since there exists no global minimiser
# #       (the model class is not closed) and the limit can be reached with many different parameterisations.


# ====================
# Experiment 5
# ====================
# NOTE: Finally, we try a sample size of 1 and decrease the step size from the beginning.
target = lambda x: 1e-4 + (x <= (1 / jnp.pi))
input_dimension = 1
width = 10
output_dimension = 1
activation = lambda x: (jnp.tanh(x) + 1) / 2
finite_difference = 0
method = "NGD_quasi_projection"
sample_size = 1
sampling = "optimal"
num_epochs = 15
step_size_rule = "decreasing"
limit_epoch = 0
init_step_size = 1
epoch_length = 100


# ====================
# Experiment 6
# ====================
# NOTE: Here we try a ReLU activation.
# target = lambda x: 1e-4 + (x <= (1 / jnp.pi))
# target = lambda x: jnp.exp(x)  # Reaches an error of 4e-6
# num_epochs = 30
target = lambda x: jnp.sin(2 * jnp.pi * x)  # Reaches an error of 2e-4
num_epochs = 15
activation = lambda x: jnp.maximum(x, 0)
input_dimension = 1
width = 20
output_dimension = 1
finite_difference = 0
method = "NGD_quasi_projection"
# method = "SGD"
sample_size = 1
sampling = "optimal"
# step_size_rule = "decreasing"
step_size_rule = "constant"
limit_epoch = 0
init_step_size = 0.001
epoch_length = 500


# plot_intermediate = True
plot_intermediate = False
gramian_quadrature_points = 10_000


num_parameters = output_dimension + output_dimension * width + width + width * input_dimension


def prediction(parameters, x):
    A1, b1, A0, b0 = parameters
    assert A1.shape == (output_dimension, width) and b1.shape == (output_dimension,)
    assert A0.shape == (width, input_dimension) and b0.shape == (width,)
    assert x.ndim == 2 and x.shape[0] == input_dimension
    return jnp.dot(A1, activation(jnp.dot(A0, x) + b0[:, None])) + b1[:, None]


# def random_parameters(key):
#     A1_key, b1_key, A0_key, b0_key = jax.random.split(key, 4)
#     return [
#         jax.random.normal(A1_key, (output_dimension, width)),
#         jax.random.normal(b1_key, (output_dimension,)),
#         jax.random.normal(A0_key, (width, input_dimension)),
#         jax.random.normal(b0_key, (width,))
#     ]


def random_parameters(key):
    A1_key, b1_key, A0_key, b0_key = jax.random.split(key, 4)
    normalise = lambda x: x / jnp.linalg.norm(x)
    return [
        normalise(jax.random.uniform(A1_key, (output_dimension, width), minval=-1, maxval=1)),
        normalise(jax.random.uniform(b1_key, (output_dimension,), minval=-1, maxval=1)),
        jnp.ones((width, input_dimension)),
        jax.random.uniform(b0_key, (width,), minval=-1, maxval=0)
    ]


key = jax.random.PRNGKey(0)
parameters_key, key = jax.random.split(key, 2)
parameters = random_parameters(parameters_key)
assert sum(p.size for p in parameters) == num_parameters


def loss(parameters, xs, ys, ws):
    return 0.5 * jnp.mean(ws * (prediction(parameters, xs) - ys)**2)


@jax.jit
def loss_integrand(x, parameters):
    x = onp.reshape(x, (1, -1))
    y = target(x)
    return jnp.sum((y - prediction(parameters, x)) ** 2, axis=0)


def true_loss(parameters, epsrel=1e-2):
    grad_norm = osp.integrate.romberg(
        loss_integrand,
        0,
        1,
        args=(parameters,),
        tol=onp.finfo(onp.float32).eps,
        rtol=epsrel,
        vec_func=True
    )
    return 0.5 * grad_norm[0]


def vectorised_parameters(parameters):
    shapes = [(output_dimension, width), (output_dimension,), (width, input_dimension), (width,)]
    assert len(parameters) == len(shapes)
    offset_shape = parameters[-1].shape[:-1]
    assert len(offset_shape) == 2 and offset_shape[0] == output_dimension and output_dimension == 1
    vector = []
    for parameter, shape in zip(parameters, shapes):
        assert parameter.shape == offset_shape + shape
        vector.append(parameter[0].reshape(offset_shape[1], -1))
    return jnp.concatenate(vector, axis=1).T


def devectorised_parameters(vector):
    assert vector.shape == (num_parameters,)
    shapes = [(output_dimension, width), (output_dimension,), (width, input_dimension), (width,)]
    start = 0
    parameters = []
    for shape in shapes:
        stop = start + jnp.prod(jnp.array(shape))
        parameters.append(vector[start:stop].reshape(shape))
        start = stop
    assert vector.shape == (stop,)
    return parameters


def generating_system(parameters, fd=0):
    if fd == 0:
        generating_system = jax.jacfwd(prediction)

        def evaluate_generating_system(xs):
            return vectorised_parameters(generating_system(parameters, xs))
    else:
        assert fd > 0
        def evaluate_generating_system(xs):
            assert xs.ndim == 2 and xs.shape[0] == input_dimension
            Phi0 = prediction(parameters, xs)
            assert Phi0.ndim == 2 and Phi0.shape[0] == output_dimension and output_dimension == 1

            system = []
            for index in range(len(parameters)):
                for i in range(parameters[index].size):
                    Ei = jnp.zeros((parameters[index].size,)).at[i].set(1).reshape(parameters[index].shape)
                    parameters_variation = list(parameters)
                    parameters_variation[index] = parameters[index] + fd * Ei
                    Phi_variation = prediction(parameters_variation, xs)
                    assert Phi_variation.shape == (output_dimension, xs.shape[1]) and output_dimension == 1
                    system.append((Phi_variation - Phi0) / fd)
                    # TODO: Do not append (Phi_variation - Phi0) / fd but just Phi_variation.
                    #       The space that is spanned is the same, but Phi_variation directly has the
                    #       necessary ridge structure to perform a retraction.
            system = jnp.concatenate(system, axis=0)
            assert system.shape == (num_parameters, xs.shape[1])
            return system

    return evaluate_generating_system


def gramian(evaluate_basis):
    assert input_dimension == 1
    xs = jnp.linspace(0, 1, gramian_quadrature_points).reshape(1, gramian_quadrature_points)
    measures = evaluate_basis(xs).T
    return jsp.integrate.trapezoid(measures[:, :, None] * measures[:, None, :], xs[0], axis=0)


def squared_l2_norm(coefficients, gram):
    return coefficients @ gram @ coefficients


def basis_transform(system):
    gram = gramian(system)
    r = jnp.linalg.matrix_rank(gram)
    s, V = jnp.linalg.eigh(gram)
    return s[::-1], V.T[::-1], r


def basis(parameters):
    system = generating_system(parameters, fd=finite_difference)
    s, Vt, r = basis_transform(system)
    s, Vt = s[:r], Vt[:r]
    X = Vt / jnp.sqrt(s)[:, None]
    return system, X, r


# assert input_dimension == 1
# xs = jnp.linspace(0, 1, 1000).reshape(1, 1000)
# system, transform, basis_dimension = basis(parameters)
# fig, ax = plt.subplots(1, 1)
# for bs in system(xs):
#     ax.plot(xs[0], bs)
# plt.show()
# exit()


# def basis_old(parameters):
#     system = generating_system(parameters, fd=finite_difference)
#     gram = gramian(system)
#     r = jnp.linalg.matrix_rank(gram)
#     s, V = jnp.linalg.eigh(gram)
#     s, V = s[-r:], V[:, -r:]
#     X = V / jnp.sqrt(s)
#     return system, X.T, r
#
#
# assert input_dimension == 1
# xs = jnp.linspace(0, 1, 1000).reshape(1, 1000)
# system, transform, basis_dimension = basis_old(parameters)
# fig, ax = plt.subplots(2, int(jnp.ceil(basis_dimension / 2)))
# ax = ax.ravel()
# for i, bs in enumerate(transform @ system(xs)):
#     ax[i].plot(xs[0], bs)
# system, transform, basis_dimension = basis(parameters)
# for i, bs in enumerate(reversed(transform @ system(xs))):
#     ax[i].plot(xs[0], bs, "--")
# plt.show()
# exit()


def gradient(parameters, xs, ys):
    return prediction(parameters, xs) - ys


def quasi_projected_gradient(parameters, xs, ys, ws):
    # === NGD with true Gramian ===
    assert xs.ndim == 2 and ys.ndim == 2
    grad = gradient(parameters, xs, ys)
    sample_size = xs.shape[1]
    # Consider M_{jk} := (system[j], system[k])_{L2} and b_j := (system[j], grad)_{L2}.
    # Then the L2 projection (qs) of grad onto the space spanned by system solves the equation
    #     M @ qs = b .
    # In the projected_gradient(...) function both M and b are estimated from samples.
    # Here, M is computed explicitly as M = gramian(...) and only b is estimated from samples.
    # This ensures that qs = inv(M) @ b remains unbiased.
    assert grad.shape == (output_dimension, sample_size) and output_dimension == 1
    system = generating_system(parameters, fd=finite_difference)
    gram = gramian(system)
    qs = system(xs) * ws @ grad[0] / sample_size
    # NOTE: By Leibniz integral rule, the preceding line is equivalent to
    # qs = jnp.concatenate([p.ravel() for p in jax.grad(loss)(parameters, xs, ys, ws)])
    # NOTE: This indeed holds for all sufficiently regular loss functions.
    #       This means that the quasi-projection algorithm is EXACTLY equivalent to NGD for L2,
    #       when the update is performed by utilising Taylors theorem.
    eps = jnp.finfo(gram.dtype).resolution * jnp.linalg.norm(gram)
    gram += eps * jnp.eye(gram.shape[0])
    qs, *_ = jnp.linalg.lstsq(gram, qs)
    return devectorised_parameters(qs)


def projected_gradient(parameters, xs, ys, ws):
    # === NGD with estimated Gramian ===
    assert xs.ndim == 2 and ys.ndim == 2
    sample_size = xs.shape[1]
    grad = gradient(parameters, xs, ys)
    assert grad.shape == (output_dimension, sample_size) and output_dimension == 1
    system, transform, basis_dimension = basis(parameters)
    measures = system(xs)
    assert measures.shape == (num_parameters, sample_size)
    qs, *_ = jnp.linalg.lstsq((measures * ws).T, grad[0] * ws)
    # emp_gram = system(xs) * ws @ system(xs).T
    # print(f"    Stability: {jnp.linalg.norm(emp_gram - jnp.eye(num_parameters))}")
    return devectorised_parameters(qs)


# @jax.jit
def updated_parameters(parameters, xs, ys, ws, step_size):
    if method == "SGD":
        gradients = jax.grad(loss)(parameters, xs, ys, ws)
    elif method == "NGD_quasi_projection":
        gradients = quasi_projected_gradient(parameters, xs, ys, ws)
    else:
        assert method == "NGD_projection"
        gradients = projected_gradient(parameters, xs, ys, ws)
    return [θ - step_size * dθ for (θ, dθ) in zip(parameters, gradients)]


assert input_dimension == 1
xs = jnp.linspace(0, 1, 1000).reshape(1, 1000)
ws = jnp.ones((1000,))
ys = target(xs)
losses = []
variation_constants = []
def plot_state(title):
    fig, ax = plt.subplots(1, 3, figsize=(14, 7))

    ax[0].plot(xs[0], ys[0], "k-", lw=2, label="target")
    zs = prediction(parameters, xs)
    ax[0].plot(xs[0], zs[0], "k--", lw=2, label="estimate")
    ax[0].set_xlim(0, 1)
    ax[0].legend()

    system, transform, basis_dimension = basis(parameters)
    ks = 0
    for bs in transform @ system(xs):
        ax[1].plot(xs[0], bs, lw=1, alpha=0.5)
        assert bs.shape == (xs.shape[1],)
        ks += bs**2
    ks /= basis_dimension
    ax[1].plot(xs[0], ks, "-", color="tab:red", lw=2, label=r"$\mathfrak{K}$")
    ax[1].set_xlim(0, 1)
    ax[1].legend(loc="upper right")

    steps = jnp.arange(1, len(losses) + 1)
    ax[2].plot(steps, losses, color="tab:blue", label="Loss")
    ax[2].plot(steps, variation_constants, color="tab:red", label=r"$\|\mathfrak{K}\|_{L^\infty}$")
    ax[2].set_xlim(1, num_epochs * epoch_length + 1)
    ylim = ax[2].get_ylim()
    if ylim[0] <= 0:
        ylim[0] = 1e-5
    ylim = min(ylim[0], 1e-5), max(ylim, 1e3)
    ax[2].set_ylim(*ylim)
    ax[2].set_xscale("log")
    ax[2].set_yscale("log")
    ax[2].legend(loc="upper right")

    fig.suptitle(title)
    plt.show()


def latex_float(value, places=2):
    assert places > 0
    if jnp.isnan(value):
        return r"$\mathrm{NaN}$"
    s = "-" * int(value < 0)  # sign
    x = abs(value)
    m = f"{x:.{places}e}"  # mantissa
    assert m[2 + places : 4 + places] in ["e+", "e-"]
    e = int(m[3 + places :])  # exponent
    m = m[: 2 + places]
    return fr"${s}{m}\cdot 10^{{{e}}}$"


def optimal_sampling_density(parameters):
    system, transform, basis_dimension = basis(parameters)
    def density(xs):
        assert xs.ndim == 2 and xs.shape[0] == input_dimension
        ks = 0
        for bs in transform @ system(xs):
            assert bs.shape == (xs.shape[1],)
            ks += bs**2
        return ks / basis_dimension
    return density


# ks = optimal_sampling_density(parameters)(xs)
# ks_mass = jsp.integrate.trapezoid(ks, xs[0])
# assert jnp.isclose(ks_mass, 1, atol=1 / xs.shape[1])
# plt.plot(xs[0], ks)
# plt.title("Optimal sampling density")
# plt.show()
# exit()


# TODO: compute curvature at every step
# def hessian(f):
#     return jax.jacfwd(jax.jacrev(f))
# H = hessian(f)(W)
# print("hessian, with shape", H.shape)
# print(H)
# If f : R^n -> R^m then
# - f(x) in R^m  (value)
# - df(x) in R^{m * n}  (Jacobian)
# - d^2f(x) in R^{m * n * n}  (Hessian)


def embed(parameters):
    A1, b1 = parameters[:2]
    coefficients = jnp.zeros((num_parameters,))
    start, stop = 0, A1.size
    coefficients = coefficients.at[start:stop].set(A1.ravel())
    start, stop = stop, stop + b1.size
    coefficients = coefficients.at[start:stop].set(b1.ravel())
    return coefficients


system = generating_system(parameters, finite_difference)
s, Vt, basis_dimension = basis_transform(system)
zs = prediction(parameters, xs)
coefficients = embed(parameters)
measures = system(xs)
assert measures.shape == (num_parameters, xs.shape[1])
# assert jnp.allclose(coefficients @ measures, zs[0], atol=jnp.finfo(zs.dtype).resolution)
assert jnp.allclose(coefficients @ measures, zs[0], atol=1e-4)

transform = Vt[:basis_dimension] / jnp.sqrt(s[:basis_dimension, None])
assert jnp.allclose(transform, jnp.diag(1 / jnp.sqrt(s[:basis_dimension])) @ Vt[:basis_dimension])
onb_measures = transform @ measures
assert onb_measures.shape == (basis_dimension, xs.shape[1])
G = jsp.integrate.trapezoid(onb_measures[:, :, None] * onb_measures.T[None], xs[0], axis=1)
# assert jnp.allclose(G, jnp.eye(basis_dimension), atol=jnp.finfo(zs.dtype).resolution)
# TODO: All integration routines should just take rtol and return the error as well.
#       This includes gramian().
if not jnp.allclose(G, jnp.eye(basis_dimension), atol=1e-3):
    print("WARNING: Gramian is ill-conditioned or badly integrated.")
    print(f"         Orthogonalisation error: {jnp.linalg.norm(G - jnp.eye(basis_dimension)):.2e}")

# onb_coefficients, *_ = jnp.linalg.lstsq(transform.T, coefficients)
basis_dimension += 1
onb_measures = Vt[:basis_dimension] / jnp.sqrt(s[:basis_dimension, None]) @ measures
onb_coefficients = jnp.sqrt(s[:basis_dimension]) * (Vt[:basis_dimension] @ coefficients)

# plt.plot(xs[0], zs[0])
# plt.plot(xs[0], coefficients @ measures, "--")
# plt.plot(xs[0], onb_coefficients @ onb_measures, "-.")
# plt.show()
# exit()

gram = gramian(system)  # TODO: The gramian should be a parameter to the rounding routines.

def greedy_threshold(coefficients):
    assert coefficients.ndim == 1
    nonzero_indices = jnp.where(coefficients != 0)[0]
    errors = []
    for index in nonzero_indices:
        candidate = coefficients.at[index].set(0)
        errors.append(squared_l2_norm(coefficients - candidate, gram))
    optimal_index = jnp.argmin(jnp.asarray(errors))
    return coefficients.at[nonzero_indices[optimal_index]].set(0), errors[optimal_index]

# candidate = coefficients
# error = 0
# while jnp.linalg.norm(candidate) > 0:
#     fig, ax = plt.subplots(1, 2)
#     ax[0].plot(xs[0], zs[0], "k-", lw=2)
#     ax[0].plot(xs[0], candidate @ measures, "--", lw=1.5, color="tab:red")
#     ax[1].stem(candidate)
#     fig.suptitle(f"NNZ: {jnp.count_nonzero(candidate)}  |  L2 error: {latex_float(error)}")
#     plt.show()
#     candidate, error = greedy_threshold(candidate)
# exit()

# TODO: Note that the greedy_threshold is not good for computing the retraction,
#       since it will just truncate the gradient update away...

import numpy as onp
from lasso_lars import warnings, LarsState, ConvergenceWarning


def greedy_lars_threshold(coefficients, max_dimension):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        coefs = onp.array(coefficients)
        active = []
        for _ in range(max_dimension):
            lars = LarsState(onp.array(gram), coefs)
            lars.add_index()
            assert len(lars.active) == 1
            active.extend(lars.active)
            # Find candidate that minimises
            #     |transform @ coefficients - transform @ candidate|_2^2
            #     == coefficients @ gram @ coefficients - 2 * coefficients @ gram @ candidate + candidate @ gram @ candidate
            #     <--> coefficients @ gram == gram @ candidate
            # The constraint on active just reduces this first order optimality system to the corresponding rows.
            candidate, *_ = jnp.linalg.lstsq(gram[jnp.array(active)][:, jnp.array(active)], coefficients @ gram[:, jnp.array(active)])
            candidate = jnp.zeros(len(coefficients)).at[jnp.array(active)].set(candidate)
            yield candidate
            coefs[active] = 0


# for candidate in greedy_lars_threshold(coefficients, jnp.count_nonzero(coefficients)):
#     active = jnp.nonzero(candidate)[0]
#     print(active)
#     error = squared_l2_norm(coefficients - candidate, gram)
#     fig, ax = plt.subplots(1, 2)
#     ax[0].plot(xs[0], zs[0], "k-", lw=2)
#     ax[0].plot(xs[0], candidate @ measures, "--", lw=1.5, color="tab:red")
#     ax[1].stem(candidate)
#     fig.suptitle(f"NNZ: {jnp.count_nonzero(candidate)}  |  L2 error: {latex_float(error)}")
#     plt.show()
# exit()


def retract(coefficients, max_dimension=jnp.inf, squared_error_threshold=0):
    # candidate = coefficients
    # while jnp.count_nonzero(candidate) > max_dimension:
    #     candidate, squared_error = greedy_threshold(candidate)
    # tentative_candidate = candidate
    # while squared_error <= squared_error_threshold:
    #     candidate = tentative_candidate
    #     tentative_candidate, squared_error = greedy_threshold(candidate)
    candidate = jnp.zeros(len(coefficients))
    if max_dimension == jnp.inf:
        max_dimension = jnp.count_nonzero(coefficients)
    for candidate in greedy_lars_threshold(coefficients, max_dimension):
        squared_error = squared_l2_norm(coefficients - candidate, gram)
        if squared_error <= squared_error_threshold:
            break
    return candidate


# fig, ax = plt.subplots(2, 2)
# ax[0, 0].plot(xs[0], zs[0], "k-", lw=2)
# ax[0, 0].plot(xs[0], coefficients @ measures, "--", lw=1.5, color="tab:red")
# ax[0, 1].stem(coefficients)
# print(f"Max. dimension: {basis_dimension}")
# retracted_coefficients = retract(coefficients, basis_dimension, 1e-3)
# print(f"Used dimension: {jnp.count_nonzero(retracted_coefficients)}")
# ax[1, 0].plot(xs[0], zs[0], "k-", lw=2)
# ax[1, 0].plot(xs[0], retracted_coefficients @ measures, "--", lw=1.5, color="tab:red")
# ax[1, 1].stem(retracted_coefficients)
# plt.show()
# exit()


# NOTE: If u is the update vector, then we want
#         0.5 * norm(f - retract(u))**2 <= 0.5 * norm(f - u)**2 + 0.5 * C * norm(u)**2 .
#       However, with the current retraction, we only obtain
#         norm(f - retract(u)) <= norm(f - u) + norm(u - retract(u)) <= norm(f - u) + thresholding_error .
#       Although the thresholding_error can be chosen adaptively (and thus arbitrarily small),
#       it is not clear to me, how we can achieve the first bound without strong Lipschitz continuity
#       assumptions on the loss function.
# NOTE: On the bright side, I think that if we retract back to the same width,
#       then we can probably argue that the classical curvature bound bound (with C being the curvature) holds.
#       Then, whenever we stagnate we can adapt the model class by increasing the width with a random kick.


*_, basis_dimension = basis(parameters)
if plot_intermediate:
    plot_state(f"Initialisation  |  Loss: {latex_float(loss(parameters, xs, ys, ws), places=2)}  |  Basis dimension: {basis_dimension}")
for epoch in range(num_epochs):
    for step in range(epoch_length):
        if step_size_rule == "constant":
            step_size = init_step_size
        elif step_size_rule == "constant_epoch":
            step_size = init_step_size / 10 ** max(epoch - limit_epoch + 1, 0)
        else:
            assert step_size_rule == "decreasing"
            total_step = epoch * epoch_length + step
            limit_total_step = limit_epoch * epoch_length
            relative_step = max(total_step - limit_total_step, 0)
            step_size = init_step_size / jnp.sqrt(relative_step + 1)
        losses.append(true_loss(parameters))
        training_key, key = jax.random.split(key, 2)
        assert input_dimension == 1
        osd = optimal_sampling_density(parameters)
        ps = osd(xs)
        if sampling == "uniform":
            xs_train = jax.random.uniform(training_key, (input_dimension, sample_size), minval=0, maxval=1)
            ws_train = jnp.ones((sample_size,))
        else:
            assert sampling == "optimal"
            xs_train = jax.random.choice(training_key, xs[0], (sample_size,), replace=False, p=ps)[None]
            ws_train = 1 / osd(xs_train)
        ys_train = target(xs_train)

        system = generating_system(parameters)
        gram = gramian(system)
        basis_dimension = jnp.linalg.matrix_rank(gram)
        parameters = updated_parameters(parameters, xs_train, ys_train, ws_train, step_size)
        # NOTE: The gradient norm returned by updated_parameters(...) is not a valid indicator of a stationary point,
        #       since it is the L2 norm of the estimated projected gradient.
        #       This estiamte may not be zero even though the true projected gradient is.
        #       But estimating the true projected gradient is not feasible.
        variation_constants.append(jnp.max(ps * basis_dimension))
        print(f"[{epoch+1:0{len(str(num_epochs))}d} | {step+1:0{len(str(epoch_length))}d}] Loss: {losses[-1]:.2e}  |  Step size: {step_size:.2e}  |  Basis dimension: {basis_dimension}")
    if plot_intermediate is True:
        plot_state(f"Epoch {epoch+1}  |  Loss: {latex_float(losses[-1], places=2)}  |  Basis dimension: {basis_dimension}")
plot_state(f"Termination  |  Loss: {latex_float(losses[-1], places=2)}  |  Basis dimension: {basis_dimension}")


A1, b1, A0, b0 = parameters
jnp.savez("shallow_parameter.npz", A1=A1, b1=b1, A0=A0, b0=b0)