import jax
import jax.numpy as jnp

from tqdm import trange
import matplotlib.pyplot as plt


activation = lambda x: (jnp.tanh(x) + 1) / 2
# activation = lambda x: 1 / (1 + jnp.exp(-x))


def predict(parameters, x):
    x = jnp.dot(parameters[1], x)
    b = parameters[2].reshape([width] + [1]*(x.ndim-1))
    assert b.ndim == x.ndim and b.shape[0] == x.shape[0]
    return parameters[0] @ activation(x + b)


input_dimension = 1
output_dimension = 1
# width = 100
width = 10

key = jax.random.PRNGKey(4)
layer_1_key, layer_2_key, bias_key, key = jax.random.split(key, 4)
parameters = [
    jax.random.normal(layer_1_key, (output_dimension, width)),
    jax.random.normal(layer_2_key, (width, input_dimension)),
    jax.random.normal(bias_key, (width,))
]
num_parameters = sum(p.size for p in parameters)


target = lambda x: jnp.sin(2 * jnp.pi * x)


assert input_dimension == 1
xs = jnp.linspace(0, 1, 1000).reshape(1, 1000)
ys = predict(parameters, xs)


def loss(parameters, xs, ys):
    return 0.5 * jnp.mean((predict(parameters, xs) - ys)**2)

def compute_basis(parameters):
    full_basis = jax.jacfwd(predict)

    def evaluate_full_basis(xs):
        measures = full_basis(parameters, xs)
        # assert len(measures) == len(parameters)
        # assert output_dimension == 1
        sample_size = xs.shape[1]
        # assert xs.shape == (input_dimension, sample_size)
        for index in range(len(parameters)):
            # assert measures[index].shape == (output_dimension, sample_size) + parameters[index].shape
            measures[index] = measures[index].reshape(sample_size, -1)
        measures = jnp.concatenate(measures, axis=1)
        # assert measures.shape == (sample_size, num_parameters)
        return measures

    measures = evaluate_full_basis(xs)
    gramian = jnp.trapz(measures[:, :, None] * measures[:, None, :], xs[0], axis=0)
    r = jnp.linalg.matrix_rank(gramian)
    s, V = jnp.linalg.eigh(gramian)
    # assert jnp.allclose(V * s @ V.T, gramian)
    # assert jnp.all(abs(s[:-r]) < 5e-6)
    s, V = s[-r:], V[:, -r:]
    # assert jnp.linalg.norm(V * s @ V.T - gramian) < 5e-6
    # assert jnp.all(s > 0)
    X = V / jnp.sqrt(s)
    measures = measures @ X
    gramian = jnp.trapz(measures[:, :, None] * measures[:, None, :], xs[0], axis=0)
    # assert jnp.linalg.norm(gramian - jnp.eye(r)) < 5e-4

    def basis(xs):
        return evaluate_full_basis(xs) @ X

    return evaluate_full_basis, X


# full_basis, orth = compute_basis(parameters)
# for bs in (full_basis(xs) @ orth).T:
#     plt.plot(xs[0], bs)
# plt.show()


def gradient(parameters, xs, ys):
    return predict(parameters, xs) - ys


def quasi_projection(parameters, xs, ys):
    g = gradient(parameters, xs, ys)
    assert input_dimension == output_dimension == 1
    sample_size = xs.shape[1]
    assert xs.shape == (input_dimension, sample_size)
    assert g.shape == (output_dimension, sample_size)
    full_basis, orth = compute_basis(parameters)
    fbs = full_basis(xs) @ orth
    assert fbs.shape[0] == sample_size
    qs = g[0] @ fbs / sample_size
    qs = qs @ orth.T
    grads = []
    start = 0
    for p in parameters:
        stop = start + p.size
        grads.append(qs[start:stop].reshape(p.shape))
        start = stop
    assert stop == len(qs)
    return grads

s = 0.1
# @jax.jit
def update(parameters, xs, ys):
    # gradients = jax.grad(loss)(parameters, xs, ys)
    gradients = quasi_projection(parameters, xs, ys)
    return [θ - s*dθ for (θ, dθ) in zip(parameters, gradients)]

num_steps = 10_000
sample_size = 10
losses = []
for step in range(num_steps):
    losses.append(loss(parameters, xs, target(xs)))
    if step > 1000:
        s = 0.1 / jnp.sqrt(step - 1000)
    print(f"[{step}] Loss: {losses[-1]:.2e}")
    training_key, key = jax.random.split(key, 2)
    xs_train = jax.random.uniform(training_key, (input_dimension, sample_size), minval=0, maxval=1)
    ys_train = target(xs_train)
    parameters = update(parameters, xs_train, ys_train)

ys_new = predict(parameters, xs)
fig, ax = plt.subplots(1, 2)
ax[0].plot(xs[0], ys[0])
ax[0].plot(xs[0], ys_new[0])
ax[0].plot(xs[0], target(xs)[0], "k--", lw=2)
ax[1].plot(losses)
ax[1].set_xscale("log")
ax[1].set_yscale("log")
plt.show()
