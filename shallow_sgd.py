import jax
import jax.numpy as jnp

import matplotlib.pyplot as plt


target = lambda x: jnp.sin(2 * jnp.pi * x)


input_dimension = 1
# width = 10
# NOTE: All sample size and epoch lenght parameters below are chosen for a width of 10.
#       Using a larger width means that the approximation error is smaller.
#       But we observe that with the previously well-chosen step size 0.01,
#       the parameters converge to a suboptimal stationary point (loss: 1e-1),
#       which chould also be achieved with width 10 (basis dimension == 3).
width = 100
output_dimension = 1
num_parameters = output_dimension + output_dimension * width + width + width * input_dimension
activation = lambda x: (jnp.tanh(x) + 1) / 2
# NOTE: Interestingly, the basis functions of the tanh-activation look like polynomials of bounded degree.
# activation = lambda x: 1 / (1 + jnp.exp(-x))
# activation = lambda x: jnp.maximum(x, 0)


def prediction(parameters, x):
    A1, b1, A0, b0 = parameters
    assert A1.shape == (output_dimension, width) and b1.shape == (output_dimension,)
    assert A0.shape == (width, input_dimension) and b0.shape == (width,)
    assert x.ndim == 2 and x.shape[0] == input_dimension
    return jnp.dot(A1, activation(jnp.dot(A0, x) + b0[:, None])) + b1[:, None]


def random_parameters(key):
    A1_key, b1_key, A0_key, b0_key = jax.random.split(key, 4)
    return [
        jax.random.normal(A1_key, (output_dimension, width)),
        jax.random.normal(b1_key, (output_dimension,)),
        jax.random.normal(A0_key, (width, input_dimension)),
        jax.random.normal(b0_key, (width,))
    ]


key = jax.random.PRNGKey(0)
parameters_key, key = jax.random.split(key, 2)
parameters = random_parameters(parameters_key)
assert sum(p.size for p in parameters) == num_parameters


def loss(parameters, xs, ys):
    return 0.5 * jnp.mean((prediction(parameters, xs) - ys)**2)


def generating_system(parameters):
    generating_system = jax.jacfwd(prediction)

    def evaluate_generating_system(xs):
        measures = generating_system(parameters, xs)
        # assert len(measures) == len(parameters)
        # assert output_dimension == 1
        sample_size = xs.shape[1]
        # assert xs.shape == (input_dimension, sample_size)
        for index in range(len(parameters)):
            # assert measures[index].shape == (output_dimension, sample_size) + parameters[index].shape
            measures[index] = measures[index].reshape(sample_size, -1)
        measures = jnp.concatenate(measures, axis=1)
        # assert measures.shape == (sample_size, num_parameters)
        return measures.T

    return evaluate_generating_system


def gramian(evaluate_basis, n=1_000):
    assert input_dimension == 1
    xs = jnp.linspace(0, 1, n).reshape(1, n)
    measures = evaluate_basis(xs).T
    return jnp.trapz(measures[:, :, None] * measures[:, None, :], xs[0], axis=0)


def basis(parameters, n=1_000):
    system = generating_system(parameters)
    gram = gramian(system, n=n)
    r = jnp.linalg.matrix_rank(gram)
    s, V = jnp.linalg.eigh(gram)
    s, V = s[-r:], V[:, -r:]
    X = V / jnp.sqrt(s)
    return system, X.T, r


def gradient(parameters, xs, ys):
    return prediction(parameters, xs) - ys


def devectorized_parameters(vector):
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


def quasi_projected_gradient(parameters, xs, ys):
    assert xs.ndim == 2 and ys.ndim == 2
    grad = gradient(parameters, xs, ys)
    sample_size = xs.shape[1]
    system, transform, basis_dimension = basis(parameters)
    bs = transform @ system(xs)
    assert bs.shape == (basis_dimension, sample_size)
    assert grad.shape == (output_dimension, sample_size) and output_dimension == 1
    qs = bs @ grad[0] / sample_size
    qs = transform.T @ qs
    return devectorized_parameters(qs)


# @jax.jit
def updated_parameters(parameters, xs, ys, step_size):
    # gradients = jax.grad(loss)(parameters, xs, ys)
    gradients = quasi_projected_gradient(parameters, xs, ys)
    return [θ - step_size * dθ for (θ, dθ) in zip(parameters, gradients)]

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
# TODO: instead of having a list of parameters, have one long parameter vector --> makes it easier to think about...


assert input_dimension == 1
xs = jnp.linspace(0, 1, 1000).reshape(1, 1000)
ys = target(xs)
def plot_state(title=""):
    plt.plot(xs[0], ys[0], "k-", lw=2)
    zs = prediction(parameters, xs)
    plt.plot(xs[0], zs[0], "k--", lw=2)
    system, transform, basis_dimension = basis(parameters)
    # for bs in system(xs):
    for bs in transform @ system(xs):
        plt.plot(xs[0], bs, lw=1)
    if len(title) > 0:
        title = title + "  |  "
    plt.title(title + f"Basis dimension: {basis_dimension}")
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


sample_size = 10
num_epochs = 10
# NOTE: The following step size is too large and yields convergence to a suboptimal stationary point.
# step_size = 0.1
step_size = 0.01
epoch_length = 100
# NOTE: Apparently, 1e-6 is the approximation error in this model class,
#       since a smaller step size yields the same error.
# step_size = 0.001
# epoch_length = 1_000
# NOTE: An intuitive idea is thus to use a step size schedule.
#       But, interestingly, the GD converges to a suboptimal stationary point,
#       which can not be escaped when the step size is reduced in later epochs.
# num_epochs = 4
# step_size_list = [1, 0.1, 0.01, 0.001]
# assert len(step_size_list) == num_epochs
# epoch_length = 50

plot_state(f"Initialisation  |  Loss: {latex_float(loss(parameters, xs, ys), places=2)}")
# exit()
losses = []
for epoch in range(num_epochs):
    # step_size = step_size_list[epoch]
    for step in range(epoch_length):
        # if step > 1000:
        #     step_size = 0.1 / jnp.sqrt(step - 1000)
        losses.append(loss(parameters, xs, ys))
        print(f"[{epoch+1:{len(str(num_epochs))}d} | {step+1:{len(str(epoch_length))}d}] Loss: {losses[-1]:.2e}")
        training_key, key = jax.random.split(key, 2)
        xs_train = jax.random.uniform(training_key, (input_dimension, sample_size), minval=0, maxval=1)
        ys_train = target(xs_train)
        parameters = updated_parameters(parameters, xs_train, ys_train, step_size)
    plot_state(f"Epoch {epoch+1}  |  Loss: {latex_float(losses[-1], places=2)}")

fig, ax = plt.subplots(1, 2)
ax[0].plot(xs[0], ys[0], "k-", lw=2)
zs = prediction(parameters, xs)
ax[0].plot(xs[0], zs[0], "k--", lw=2)
ax[1].plot(losses)
ax[1].set_xscale("log")
ax[1].set_yscale("log")
plt.show()
