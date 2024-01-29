import argparse
import tomllib
from pathlib import Path

import jax
import jax.numpy as jnp
import jax.scipy as jsp

import numpy as onp
import scipy as osp

import matplotlib as mpl
import matplotlib.pyplot as plt

mpl.rc("font", size=10, family="Times New Roman")
mpl.rc("text", usetex=True)
mpl.rc(
    "text.latex",
    preamble=r"""
    \usepackage{newtxmath}
    \usepackage{amsmath}
    \usepackage{bbm}
""",
)

parser = argparse.ArgumentParser(
    description="Create a convergence plot for SGD/NGD on shallow neural networks.",
)
parser.add_argument("config", help="config file containing the parameters")
args = parser.parse_args()

config_file = Path(args.config)
print(f"Config: {config_file}")
with open(args.config, "rb") as f:
    parameters = tomllib.load(f)

target = parameters.pop("target")
assert isinstance(target, str)
if target == "sin":
    target = lambda x: jnp.sin(2 * jnp.pi * x)
elif target == "step":
    target = lambda x: 1e-4 + (x <= (1 / jnp.pi))
else:
    target = getattr(jnp, target)

activation = parameters.pop("activation")
assert isinstance(activation, str)
if activation == "tanh":
    activation = lambda x: (jnp.tanh(x) + 1) / 2
    activation.__name__ = "tanh"
elif activation.startswith("relu"):
    assert activation[4] == "_"
    power = int(activation[5:])
    assert power > 0
    activation = lambda x: jnp.maximum(x, 0) ** power
    activation.__name__ = f"relu_{power}"
else:
    activation = getattr(jnp, activation)

input_dimension = parameters.pop("input_dimension")
assert isinstance(input_dimension, int) and input_dimension == 1
width = parameters.pop("width")
assert isinstance(width, int) and width >= 1
output_dimension = parameters.pop("output_dimension")
assert isinstance(output_dimension, int) and output_dimension == 1

finite_difference = parameters.pop("finite_difference")
assert isinstance(finite_difference, (int, float)) and finite_difference >= 0

num_epochs = parameters.pop("num_epochs")
assert isinstance(num_epochs, int) and num_epochs > 0
epoch_length = parameters.pop("epoch_length")
assert isinstance(epoch_length, int) and epoch_length > 0

init = parameters.pop("initialisation", "random")
# init = "last_run"

sampling = parameters.pop("sampling")
assert isinstance(sampling, str)
sample_size = parameters.pop("sample_size")
assert sample_size == "adaptive" or (isinstance(sample_size, int) and sample_size > 0)
loss_estimate_sample_size_init = parameters.pop("loss_estimate_sample_size_init")
assert isinstance(loss_estimate_sample_size_init, int) and loss_estimate_sample_size_init > 0

method = parameters.pop("method")
assert isinstance(method, str)
step_size_rule = parameters.pop("step_size_rule")
assert isinstance(step_size_rule, str)
if step_size_rule == "constant":
    init_step_size = parameters.pop("init_step_size")
stability_bound = parameters.pop("stability_bound", float("inf"))
assert isinstance(stability_bound, (int, float)) and stability_bound > 0

label = parameters.pop("label", "")
assert isinstance(label, str)
label = label.format(**globals())
print(f"Label: {label}")

assert len(parameters) == 0, parameters
del args, parameters

base_path = config_file.parent
assert config_file.suffixes == [".toml"]
base_name = config_file.stem
assert config_file == base_path / Path(base_name).with_suffix(".toml")
if label:
    base_name = base_name + "_" + label
base_name = Path(base_name)

data_path = base_path / "data"
data_path.mkdir(exist_ok=True)
data_path /= base_name.with_suffix(".npz")

plot_path = base_path / "plot"
plot_path.mkdir(exist_ok=True)
plot_path /= base_name.with_suffix(".pdf")


#TODO: - organise the config files in sections
#      - use the section names as labels for the experiments
#        import string
#        def valid_stem(str):
#            valid_chars = "-_ " + string.ascii_letters + string.digits
#            filename = ''.join(c for c in s if c in valid_chars)
#            filename = filename.replace(' ','_')
#            return filename
#      - or: use sections to organise certain settings, e.g. step_size_rule and corresponding parameters
#      - or: use sections to organise different experiemtns in a single file all of these are then executed in one run.


# - implement different choices for the sample size
# - implement different choices to estimate the step step_size:
#     - estimate the curvature with the correct Lip (depending on the retraction error and not on the step_size)
#     - have a bounded bias term (like it is currently done)
#     - an estimated bias also induces a decay of the step size (s = 1/t).
#       Maybe we achieve the same rates with the step size sequece s = 1/t directly?
# - Run experiments for all three cases.
# - Also run an experiment for SGD.


L = 1   # Lipschitz smoothness constant for the least squares loss
mu = 1  # convexity constant for the least squares loss
plot_intermediate = True
gramian_quadrature_points = 1_000
# width += 20  # Increase the width by 20.
if isinstance(sample_size, int):
    loss_estimate_sample_size_init = max(loss_estimate_sample_size_init, sample_size)


# TODO: Try alternating optimisation?
# TODO: When adapting the network, try the following:
#         1. Retract the old network.
#         2. Initialise a new random network such that the sum has appropriate width.
#         3. Learn the residual of the old network with the new network.
#         4. Add the new network to the old network.


num_parameters = output_dimension + output_dimension * width + width + width * input_dimension


def prediction(parameters, x):
    A1, b1, A0, b0 = parameters
    width, = b0.shape
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


def random_parameters(key, width=width):
    A1_key, b1_key, A0_key, b0_key = jax.random.split(key, 4)
    normalise = lambda x: x / jnp.linalg.norm(x)
    return [
        normalise(jax.random.uniform(A1_key, (output_dimension, width), minval=-1, maxval=1)),
        normalise(jax.random.uniform(b1_key, (output_dimension,), minval=-1, maxval=1)),
        jnp.ones((width, input_dimension)),
        jax.random.uniform(b0_key, (width,), minval=-1, maxval=0)
    ]


if init in ["random", "projection"]:
    key = jax.random.PRNGKey(0)
    parameters_key, key = jax.random.split(key, 2)
    parameters = random_parameters(parameters_key)
    assert sum(p.size for p in parameters) == num_parameters
else:
    assert init == "last_run"
    key = jax.random.PRNGKey(1)
    z = jnp.load("shallow_parameters.npz")
    parameters = z["A1"], z["b1"], z["A0"], z["b0"]


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
    *offset_shape, width = parameters[-1].shape
    offset_shape = tuple(offset_shape)
    shapes = [(output_dimension, width), (output_dimension,), (width, input_dimension), (width,)]
    assert len(parameters) == len(shapes)
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
    gram += eps * jnp.eye(gram.shape[0])  # improve numerical stability
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


def update_direction(parameters, xs, ys, ws):
    if method == "SGD":
        gradients = jax.grad(loss)(parameters, xs, ys, ws)
    elif method == "NGD_quasi_projection":
        gradients = quasi_projected_gradient(parameters, xs, ys, ws)
    else:
        assert method == "NGD_projection"
        gradients = projected_gradient(parameters, xs, ys, ws)
    return gradients


# @jax.jit
def updated_parameters(parameters, gradients, step_size):
    return [θ - step_size * dθ for (θ, dθ) in zip(parameters, gradients)]


assert input_dimension == 1
xs = jnp.linspace(0, 1, 1000).reshape(1, 1000)
ws = jnp.ones((1000,))
ys = target(xs)
losses = []
sample_sizes = []
variation_constants = []
if activation.__name__.startswith("relu"):
    knotxs = [[] for _ in range(width)]
    knotys = [[] for _ in range(width)]
step_sizes = []
loss_estimates = []
retraction_errors = []
def plot_state(label):
    fig, ax = plt.subplots(1, 3, figsize=(14, 7))

    ax[0].plot(xs[0], ys[0], "k-", lw=2, label="target")
    zs = prediction(parameters, xs)
    ax[0].plot(xs[0], zs[0], "k--", lw=2, label="estimate")
    ylim = ax[0].get_ylim()
    if activation.__name__.startswith("relu"):
        for i in range(width):
            if i == 0:
                ax[0].plot(knotxs[i][-1:], knotys[i][-1:], "o-", markersize=6, fillstyle="full", color="tab:red", markeredgewidth=1.5, label="spline knots")
            else:
                ax[0].plot(knotxs[i][-1:], knotys[i][-1:], "o-", markersize=6, fillstyle="full", color="tab:red", markeredgewidth=1.5)
            ax[0].plot(knotxs[i], knotys[i], "-", linewidth=1.5, color="tab:red")
            ax[0].plot(knotxs[i][:1], knotys[i][:1], "o-", markersize=6, fillstyle="full", color="tab:red", markeredgewidth=1.5, markerfacecolor="white")
    ax[0].set_xlim(0, 1)
    ax[0].set_ylim(*ylim)
    ax[0].legend()
    ax[0].set_title("Approximation")

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
    ax[1].set_title("Current basis")

    # steps = onp.arange(1, len(losses) + 1)
    # samples = loss_estimate_sample_size_init + steps * sample_size
    samples = onp.cumsum(sample_sizes)[1:]
    ax[2].plot(samples, variation_constants, color="tab:red", label=r"$\|\mathfrak{K}\|_{L^\infty}$")
    ax[2].plot(samples, retraction_errors, color="tab:orange", label="retraction error")
    ax[2].plot(samples, step_sizes, color="tab:purple", label="step size")
    ax[2].plot(samples, loss_estimates, "--", color="tab:blue", label="loss estimate")
    ax[2].plot(samples, losses, color="tab:blue", label="loss")
    if isinstance(sample_size, int):
        xlim = loss_estimate_sample_size_init + sample_size, loss_estimate_sample_size_init + sample_size * num_epochs * epoch_length
    elif len(samples) > 0:
        xlim = samples[0], samples[-1]
    else:
        xlim = sample_sizes[0], 2 * sample_sizes[0]
    ax[2].set_xlim(*xlim)
    ax[2].set_xscale("log")
    xticks = set(xlim)
    xticks |= set(10 ** onp.arange(*onp.ceil(onp.log10(xlim)).astype(int)))
    xticks = onp.array(sorted(xticks))
    ax[2].set_xticks(xticks)
    xticklabels = ax[2].get_xticklabels()
    xticklabels[-1].set_text(fr"$\mathdefault{{{latex_float(xticks[-1], 1)[1:-1]}}}$")
    ax[2].set_xticklabels(xticklabels)
    ax[2].set_xlabel("cumulative sample size")
    ylim = (1e-4, 1e2)
    if len(losses) > 0:
        ls = jnp.array(losses)
        ks = jnp.array(variation_constants)
        ylim_0 = min(ls[ls > 0].min(), ks[ks>0].min(), ylim[0])
        assert ylim_0 < 1
        ylim_0 = 10 ** (1.1 * jnp.log10(ylim_0))
        ylim_1 = max(ls.max(), ks.max(), ylim[1])
        ylim_1 = 10 ** (1.1 * jnp.log10(ylim_1))
        ylim = (ylim_0, ylim_1)
    ax[2].set_ylim(*ylim)
    ax[2].set_yscale("log")
    ax[2].legend(loc="upper right")
    ax[2].set_title("Convergence")

    if len(losses) > 0:
        title = f"{label}  |  Loss: {latex_float(losses[-1], places=2)}  |  Basis dimension: {basis_dimension}"
    else:
        title = f"{label}  |  Loss: {latex_float(loss(parameters, xs, ys, ws), places=2)}  |  Basis dimension: {basis_dimension}"
    fig.suptitle(title)
    print(f"Saving convergence plot to '{plot_path}'")
    plt.savefig(
        plot_path, dpi=300, edgecolor="none", bbox_inches="tight", transparent=True
    )


def save_state():
    print(f"Saving convergence data to '{data_path}'")
    A1, b1, A0, b0 = parameters
    knts = {}
    if activation.__name__.startswith("relu"):
        knts["knotxs"] = knotxs
        knts["knotys"] = knotys
    jnp.savez(data_path,
              A1=A1,
              b1=b1,
              A0=A0,
              b0=b0,
              activation=activation.__name__,
              losses=losses,
              variation_constants=variation_constants,
              loss_estimates=loss_estimates,
              retraction_errors=retraction_errors,
              step_sizes=step_sizes,
              **knts
    )


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
    num_parameters = sum(p.size for p in parameters)
    A1, b1 = parameters[:2]
    coefficients = jnp.zeros((num_parameters,))
    start, stop = 0, A1.size
    coefficients = coefficients.at[start:stop].set(A1.ravel())
    start, stop = stop, stop + b1.size
    coefficients = coefficients.at[start:stop].set(b1.ravel())
    return coefficients


# system = generating_system(parameters, finite_difference)
# s, Vt, basis_dimension = basis_transform(system)
# zs = prediction(parameters, xs)
# coefficients = embed(parameters)
# measures = system(xs)
# assert measures.shape == (num_parameters, xs.shape[1])
# # assert jnp.allclose(coefficients @ measures, zs[0], atol=jnp.finfo(zs.dtype).resolution)
# assert jnp.allclose(coefficients @ measures, zs[0], atol=1e-4)

# transform = Vt[:basis_dimension] / jnp.sqrt(s[:basis_dimension, None])
# assert jnp.allclose(transform, jnp.diag(1 / jnp.sqrt(s[:basis_dimension])) @ Vt[:basis_dimension])
# onb_measures = transform @ measures
# assert onb_measures.shape == (basis_dimension, xs.shape[1])
# G = jsp.integrate.trapezoid(onb_measures[:, :, None] * onb_measures.T[None], xs[0], axis=1)
# # assert jnp.allclose(G, jnp.eye(basis_dimension), atol=jnp.finfo(zs.dtype).resolution)
# # TODO: All integration routines should just take rtol and return the error as well.
# #       This includes gramian().
# if not jnp.allclose(G, jnp.eye(basis_dimension), atol=1e-3):
#     print("WARNING: Gramian is ill-conditioned or badly integrated.")
#     print(f"         Orthogonalisation error: {jnp.linalg.norm(G - jnp.eye(basis_dimension)):.2e}")

# # onb_coefficients, *_ = jnp.linalg.lstsq(transform.T, coefficients)
# basis_dimension += 1
# onb_measures = Vt[:basis_dimension] / jnp.sqrt(s[:basis_dimension, None]) @ measures
# onb_coefficients = jnp.sqrt(s[:basis_dimension]) * (Vt[:basis_dimension] @ coefficients)

# plt.plot(xs[0], zs[0])
# plt.plot(xs[0], coefficients @ measures, "--")
# plt.plot(xs[0], onb_coefficients @ onb_measures, "-.")
# plt.show()
# exit()

# gram = gramian(system)  # TODO: The gramian should be a parameter to the rounding routines.

# def greedy_threshold(coefficients):
#     assert coefficients.ndim == 1
#     nonzero_indices = jnp.where(coefficients != 0)[0]
#     errors = []
#     for index in nonzero_indices:
#         candidate = coefficients.at[index].set(0)
#         errors.append(squared_l2_norm(coefficients - candidate, gram))
#     optimal_index = jnp.argmin(jnp.asarray(errors))
#     return coefficients.at[nonzero_indices[optimal_index]].set(0), errors[optimal_index]

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


def greedy_lars_threshold(coefficients, gram, max_dimension):
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


# for candidate in greedy_lars_threshold(coefficients, gram, jnp.count_nonzero(coefficients)):
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


def retract(coefficients, gram, max_dimension=jnp.inf, squared_error_threshold=0):
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
    for candidate in greedy_lars_threshold(coefficients, gram, max_dimension):
        squared_error = squared_l2_norm(coefficients - candidate, gram)
        if squared_error <= squared_error_threshold:
            break
    return candidate


# fig, ax = plt.subplots(2, 2)
# ax[0, 0].plot(xs[0], zs[0], "k-", lw=2)
# ax[0, 0].plot(xs[0], coefficients @ measures, "--", lw=1.5, color="tab:red")
# ax[0, 1].stem(coefficients)
# print(f"Max. dimension: {basis_dimension}")
# threshold = 1e-6 * squared_l2_norm(coefficients, gram)
# retracted_coefficients = retract(coefficients, gram, basis_dimension, threshold)
# print(f"Used dimension: {jnp.count_nonzero(retracted_coefficients)}")
# ax[1, 0].plot(xs[0], zs[0], "k-", lw=2)
# ax[1, 0].plot(xs[0], retracted_coefficients @ measures, "--", lw=1.5, color="tab:red")
# ax[1, 1].stem(retracted_coefficients)
# plt.show()


def add(parameters_1, parameters_2):
    A1_1, b1_1, A0_1, b0_1 = parameters_1
    A1_2, b1_2, A0_2, b0_2 = parameters_2
    A1 = jnp.concatenate([A1_1, A1_2], axis=1)
    b1 = b1_1 + b1_2
    A0 = jnp.concatenate([A0_1, A0_2], axis=0)
    b0 = jnp.concatenate([b0_1, b0_2], axis=0)
    return A1, b1, A0, b0


def network_width(parameters):
    return len(parameters[-1])


if init == "last_run":
    gramian_quadrature_points *= 10
    coefficients = embed(parameters)
    system = generating_system(parameters, finite_difference)
    gram = gramian(system)
    es = jnp.linalg.eigvalsh(gram)
    basis_dimension = int(jnp.count_nonzero(es > 1e-6 * abs(es).max()))
    print(f"Max. dimension: {basis_dimension}")
    threshold = 1e-6 * squared_l2_norm(coefficients, gram)
    # Every width larger than the basis_dimension is ill-conditioned.
    retracted_coefficients = retract(coefficients, gram, basis_dimension, threshold)
    print(f"Used dimension: {jnp.count_nonzero(retracted_coefficients)}")
    retraction_error = jnp.sqrt(squared_l2_norm(coefficients - retracted_coefficients, gram))
    print(f"Retraction error: {retraction_error:.2e}")
    old_width = network_width(parameters)
    assert jnp.allclose(retracted_coefficients[old_width + output_dimension:], 0)
    nz = jnp.nonzero(retracted_coefficients[:old_width])[0]
    retracted_parameters = retracted_coefficients[nz][None], 0 * parameters[1] + retracted_coefficients[old_width], parameters[2][nz], parameters[3][nz]
    assert network_width(retracted_parameters) == len(nz)
    print(f"Loss: {loss(parameters, xs, ys, ws):.2e} → {loss(retracted_parameters, xs, ys, ws):.2e}")

    # plt.plot(xs[0], prediction(parameters, xs)[0], "k-", lw=2)
    # plt.plot(xs[0], prediction(retracted_parameters, xs)[0], "--", lw=1.5, color="tab:red")
    # plt.show()

    parameters_key, key = jax.random.split(key, 2)
    random_kick_parameters = random_parameters(parameters_key, width=width-network_width(retracted_parameters))
    random_kick_parameters[0] = 10 * retraction_error * random_kick_parameters[0]
    random_kick_parameters[1] = 0 * random_kick_parameters[1]

    new_parameters = add(retracted_parameters, random_kick_parameters)
    assert network_width(new_parameters) == width

    # plt.plot(xs[0], prediction(parameters, xs)[0], "k-", lw=2)
    # plt.plot(xs[0], prediction(new_parameters, xs)[0], "--", lw=1.5, color="tab:red")
    # plt.show()

    parameters = new_parameters
    assert sum(p.size for p in parameters) == num_parameters


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


sample_sizes.append(0)
if init == "projection":
    system, transform, basis_dimension = basis(parameters)
    gram = gramian(system)

    assert input_dimension == 1
    osd = optimal_sampling_density(parameters)
    ps = osd(xs)

    I = jnp.eye(basis_dimension)
    def stability(xs, ws):
        onb_measures = transform @ system(xs)
        G = onb_measures * ws @ onb_measures.T / len(ws)
        return jnp.linalg.norm(G - I, ord=2)

    if sample_size == "adaptive":
        local_sample_size = int(jnp.ceil(10 * basis_dimension * (jnp.log2(basis_dimension) + 1)))
    else:
        local_sample_size = sample_size
    sample_sizes[0] += local_sample_size

    if sampling == "uniform":
        training_key, key = jax.random.split(key, 2)
        xs_train = jax.random.uniform(training_key, (input_dimension, local_sample_size), minval=0, maxval=1)
        ws_train = jnp.ones((local_sample_size,))
    else:
        assert sampling == "optimal"
        while True:
            training_key, key = jax.random.split(key, 2)
            xs_train = jax.random.choice(training_key, xs[0], (local_sample_size,), replace=True, p=ps)[None]
            ws_train = 1 / osd(xs_train)
            if stability_bound == jnp.inf:
                break
            assert 0 < stability_bound < 1
            if stability(xs_train, ws_train) < stability_bound:
                break
    ys_train = target(xs_train)

    A1, b1 = parameters[:2]
    init_size = A1.size + b1.size
    measures = system(xs_train)
    assert measures.shape == (num_parameters, local_sample_size)
    measures = (measures[:init_size] * jnp.sqrt(ws_train)).T
    assert ys_train.shape == (1, local_sample_size)
    values = jnp.sqrt(ws_train) * ys_train[0]
    assert values.shape == (local_sample_size,)
    init_value = jnp.linalg.lstsq(measures, values)[0]
    A1 = init_value[:A1.size].reshape(A1.shape)
    b1 = init_value[A1.size:].reshape(b1.shape)
    parameters[:2] = A1, b1

*_, basis_dimension = basis(parameters)
if activation.__name__.startswith("relu"):
    # A1 x + b1 == 0  <-->  x == -b1 / A1
    knotx = -parameters[-1] / parameters[-2][:, 0]
    assert knotx.shape == (width,)
    knoty = prediction(parameters, knotx[None])[0]
    for i in range(width):
        knotxs[i].append(knotx[i])
        knotys[i].append(knoty[i])

loss_estimate_key, key = jax.random.split(key, 2)
loss_estimate_sample_size = loss_estimate_sample_size_init
xs_le0 = jax.random.uniform(loss_estimate_key, (input_dimension, loss_estimate_sample_size), minval=0, maxval=1)
loss_estimate = loss(parameters, xs_le0, target(xs_le0), 1)
sample_sizes[0] += loss_estimate_sample_size

plot_state(f"Initial value")
save_state()
for epoch in range(num_epochs):
    for step in range(epoch_length):
        system, transform, basis_dimension = basis(parameters)
        gram = gramian(system)
        losses.append(true_loss(parameters))

        assert input_dimension == 1
        osd = optimal_sampling_density(parameters)
        ps = osd(xs)
        variation_constants.append(jnp.max(ps) * basis_dimension)

        I = jnp.eye(basis_dimension)
        def stability(xs, ws):
            onb_measures = transform @ system(xs)
            G = onb_measures * ws @ onb_measures.T / len(ws)
            return jnp.linalg.norm(G - I, ord=2)

        if sample_size == "adaptive":
            local_sample_size = int(jnp.ceil(10 * basis_dimension * (jnp.log2(basis_dimension) + 1)))
        else:
            local_sample_size = sample_size
        sample_sizes.append(local_sample_size)

        if sampling == "uniform":
            training_key, key = jax.random.split(key, 2)
            xs_train = jax.random.uniform(training_key, (input_dimension, local_sample_size), minval=0, maxval=1)
            ws_train = jnp.ones((local_sample_size,))
        else:
            assert sampling == "optimal"
            while True:
                training_key, key = jax.random.split(key, 2)
                xs_train = jax.random.choice(training_key, xs[0], (local_sample_size,), replace=True, p=ps)[None]
                ws_train = 1 / osd(xs_train)
                if stability_bound == jnp.inf:
                    break
                assert 0 < stability_bound < 1
                if stability(xs_train, ws_train) < stability_bound:
                    break
        ys_train = target(xs_train)

        # === Option 1: Cumulative mean ===
        # loss_estimate_sample_size += local_sample_size
        # relaxation = local_sample_size / loss_estimate_sample_size
        # loss_estimate = (1 - relaxation) * loss_estimate + relaxation * loss(parameters, xs_train, ys_train, ws_train)
        # === Option 2: Exponentially weighted moving average ===
        loss_estimate = 0.5 * loss_estimate + 0.5 * loss(parameters, xs_train, ys_train, ws_train)
        # === Option 3: Windowed moving average ===
        # try:
        #     partial_loss_estimates.append(loss(parameters, xs_train, ys_train, ws_train))
        # except:
        #     partial_loss_estimates = loss_estimates[:]
        #     partial_loss_estimates.append(loss(parameters, xs_train, ys_train, ws_train))
        # partial_loss_estimates = partial_loss_estimates[-10:]
        # loss_estimate = onp.mean(partial_loss_estimates)
        loss_estimates.append(loss_estimate)

        ud = update_direction(parameters, xs_train, ys_train, ws_train)
        assert jnp.all(xs == jnp.linspace(0, 1, 1000).reshape(1, -1))
        ys_0 = prediction(parameters, xs)
        ys_update =  prediction(ud, xs)
        assert ys_0.shape == ys_update.shape == (1, 1000)
        def retraction_error(s):
            ys_ret = prediction(updated_parameters(parameters, ud, s), xs)
            ys_lin = ys_0 - s * ys_update
            return jnp.sqrt(jnp.trapz(jnp.sum((ys_ret - ys_lin)**2, axis=0), xs[0]))

        def Lip(s):
            return jnp.sqrt(2 * loss_estimate) + retraction_error(s)

        if sampling == "optimal":
            V = basis_dimension
        else:
            V = variation_constants[-1]
        if method == "SGD":
            es = jnp.linalg.svd(gram)[1]
            lmin, lmax = es.min(), es.max()
            var_1 = (lmax**2 * (local_sample_size - 1) + lmax * V) / local_sample_size
        elif method == "NGD_quasi_projection":
            var_1 = (local_sample_size + V - 1) / local_sample_size
        elif method == "NGD_projection":
            assert stability_bound < 1
            var_1 = (local_sample_size + V - 1) / local_sample_size / (1 - stability_bound)**2
        else:
            raise NotImplementedError
        smin = 0
        smax = 1 / (L * var_1)
        def descent(s, C=0):
            return s - s**2 * (L + C) / 2 * var_1

        if step_size_rule == "constant":
            step_size = init_step_size
        elif step_size_rule == "constant_epoch":
            step_size = init_step_size / 10 ** max(epoch - limit_epoch + 1, 0)
        elif step_size_rule == "adaptive_optimal":  # This is the most trivial case.
            def objective(s):
                return true_loss(updated_parameters(parameters, ud, s))
            res = osp.optimize.minimize_scalar(objective, bounds=(smin, smax), method="bounded")
            step_size = res.x
        elif step_size_rule == "adaptive_curvature":
            # Recall the descent factor σ = s - s**2 * (L+C)/2 * var_1 .
            # The step size must be larger than 0 and σ is maximised for C=0 and s=1/(L*var_1).

            # It holds that C(s) = Lip(s) * Curv(s) with
            #     Lip(s) = jnp.sqrt(2 * losses[-1]) + L * s .

            # === Estimate the curvature ===
            ud_vec = jnp.concatenate([p.ravel() for p in ud])
            ud_norm_squared = ud_vec.T @ gram @ ud_vec
            def Curv(s):
                return 2 * retraction_error(s) / (ud_norm_squared * s**2)

            def objective(s):
                C = Lip(s) * Curv(s)
                sigma = descent(s, C)
                return -sigma
            res = osp.optimize.minimize_scalar(objective, bounds=(smin, smax), method="bounded")
            step_size = res.x

        elif step_size_rule.startswith("adaptive_threshold"):
            # This approach works, but it has two drawbacks.
            # 1. Let U(·) denote the parameter to function map,
            #    θ the current parameter vector and d the parameters of the update direction.
            #    It is true, that applying Taylor's theorem to
            #        s ↦ U(θ + s * d)
            #    yields the estimate
            #        U(θ + s * d) = U(θ) + s * ∇U(θ) * d + O(s**2)
            #    and hence
            #        U(θ + s * d) - (U(θ) + s * ∇U(θ) * d) = O(s**2) .
            #    So U indeed behaves quadratically in a neighbourhood of θ and we can compute the curvature.
            #    However, this neighborhood may be extremely tiny.
            # 2. Even if the curvature is not excessively tiny another problem arises due to the estimation
            #    of the Lipschitz constant. The principal idea behind the preceding objective is that
            #        b = descent(s, C) * norm(expectation(d))**2
            #    is a lower bound for the descent of the algorithm in a step in direction d with step size s.
            #    However,
            #    - If Lip(0) is underestimated, then b is not a valid lower bound for the descent.
            #    - If Lip(0) is overestimated, then b is a lower bound for the descent,
            #      but in many cases this bound is trivial (i.e. negative).
            #
            # To sidestep this issue, note that maximising
            #     sigma
            #     = descent(s, C)
            #     = s - s**2 * (L + C) / 2 * var_1
            #     = s - s**2 * L / 2 * var_1 - s**2 * C / 2 * var_1
            #     = descent(s, 0) - s**2 * C / 2 * var_1
            # is equivalent to to maximising sigma * ud_norm_squared, which can be expressed as
            #     descent(s, 0) * ud_norm_squared - Lip(s) * retraction_error(s) * var_1 .
            # We could hence choose the step size as to maximise descent(s, 0)
            # while interpreting the retraction error as an additional bias term
            #     Lip(s) * retraction_error(s) * var_1 .
            # We could hence chose the step size that maximises descent(s, 0),
            # while satisfying a given bound on this bias term.
            # Since the theory about convergence in expectation guarantees an exponential convergence to this bound.
            # Moreover, since s ∈ [0, smax], the step size that maximises descent(s, 0) is actually the largest s.
            # Finally, to ensure convergence, we do not choose a fixed bound to the bias but a decreasing sequence of bounds.
            #
            # This approach also has the pleasant property that we are not bound to a single connected component of the manifold.
            # We can jump between different connected components as long as the retraction error is not too large.
            #
            # Note that, even though we do not use the bounded curvature property directly, it is still important.
            # A bounded curvature guarantees that the retraction error decays with higher order.
            # This means, that the bound
            #     retraction_error(s) <= 1 / t^2
            # guarantees that the algorithm converges almost surely (since t^{-2} is summable),
            # but since the retraction error is of higher order, it does not imply that s must be summable.
            # Hence we retain the Robbins--Monro property that s is not in l1.

            total_step = epoch * epoch_length + step
            if step_size_rule == "adaptive_threshold_naive":
                retraction_threshold = loss_estimate
            else:
                # retraction_threshold = 1 / jnp.sqrt(total_step + 1)
                # retraction_threshold = 1 / (total_step + 1)
                retraction_threshold = min(loss_estimate, smax / jnp.sqrt(total_step + 1))
                # retraction_threshold = min(loss_estimate, 1 / jnp.sqrt(total_step + 1))
                # retraction_threshold = min(loss_estimate, 1 / (total_step + 1))
                # TODO: All these four work.
                #       But those that prohibit the retraction from destroying the current loss work better.
                # retraction_threshold = min(loss_estimate, 1 / (total_step + 1)**2)
                # retraction_threshold = min(loss_estimate, 1 / (total_step + 1)**1.5)
                # TODO: The last two do not work! --- Maybe the curvature is extremely large. Then we need a tiny step size.
                #       Or it does not hold. In both cases, it seems a good idea to choose a threshold that does not
                #       require the bounded curvature property!
                # TODO: instead of heavy rejection based sampling, we could try 54 steps of an SGD with optimal step size.
                # (We do this to find ONE linear update. Then we perform a single retraction step.)
                # Actually, thats basically a good idea and maybe the reason why the algorithm currently does not work so well for quasi-projection.
                # A single quasi-projection step does not provide much of a reduction.
                # So if the retraction error is too large, we may not get convergence.
                # Moreover, the algorithm seems to be very sensitive to the choice of the retraction threshold...

            # We want to find a point where
            #     Lip(s) * retraction_error(s) * var_1 <= retraction_threshold .
            # But we can do this only up to a certain relative tolerance rtol, i.e.
            #     Lip(s) * retraction_error(s) * var_1 <= (1 + rtol) * retraction_threshold .
            # This is not a problem when the upper bound is arbitrary.
            # But we also would like to ensure that the retraction error is larger than the current loss estimate
            # (intuitively, to ensure that a step can not have too large of an impact).
            def root_objective(s, rtol):
                return Lip(s) * retraction_error(s) * var_1 - retraction_threshold / (1 + rtol)

            def bisect(a, b, rtol=0.5, error_b=None):
                if error_b is None:
                    error_b = root_objective(b, rtol)
                if error_b <= rtol * retraction_threshold:
                    # retraction_error(b) <= retraction_threshold
                    return b
                m = (a + b) / 2
                if Lip(m) * abs(a - b) <= rtol * retraction_threshold:
                    return m
                error_m = root_objective(m, rtol)
                if error_m > 0:
                    return bisect(a, m, rtol=rtol, error_b=error_m)
                else:
                    return bisect(m, b, rtol=rtol, error_b=error_b)
            step_size = bisect(smin, smax)

            try:
                cumulative_a *= 1 - 2 * mu * descent(step_size)
            except NameError:
                cumulative_a = 1 - 2 * mu * descent(step_size)
            print(f"    a = {cumulative_a:.2e}")
            if cumulative_a < 1e-12:
                try:
                    k
                except NameError:
                    k = 0
                k += 1
                step_size = jnp.minimum(step_size, smax / k)

        # elif step_size_rule == "mixed":
        #     ...

        else:
            assert step_size_rule == "decreasing"
            total_step = epoch * epoch_length + step
            step_size = smax / jnp.sqrt(total_step + 1)

        parameters = updated_parameters(parameters, ud, step_size)

        step_sizes.append(step_size)
        retraction_errors.append(Lip(step_size) * retraction_error(step_size) * var_1)
        if activation.__name__.startswith("relu"):
            knotx = -parameters[-1] / parameters[-2][:, 0]
            knoty = prediction(parameters, knotx[None])[0]
            for i in range(width):
                knotxs[i].append(knotx[i])
                knotys[i].append(knoty[i])

        # NOTE: The gradient norm returned by updated_parameters(...) is not a valid indicator of a stationary point,
        #       since it is the L2 norm of the estimated projected gradient.
        #       This estiamte may not be zero even though the true projected gradient is.
        #       But estimating the true projected gradient is not feasible.
        print(f"[{epoch+1:0{len(str(num_epochs))}d} | {step+1:0{len(str(epoch_length))}d}] Loss: {losses[-1]:.2e}  |  Retraction error: {retraction_errors[-1]:.2e}  |  Step size: {step_size:.2e}  |  Basis dimension: {basis_dimension}  |  Stability: {stability(xs_train, ws_train):.2f}")
    if plot_intermediate:
        plot_state(f"Epoch {epoch+1}")
        save_state()
plot_state("Terminal value")
save_state()
