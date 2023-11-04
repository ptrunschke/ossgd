import jax
import jax.numpy as jnp
import jax.scipy as jsp

import matplotlib.pyplot as plt


# # ====================
# #     Experiment 1
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


# TODO: Switch the two experiments.


# # ====================
# # Experiment 2
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
# # TODO: Do this as a first experiemnt. Then do teh following:
# # NOTE: The following step size is too large and yields convergence to a suboptimal stationary point.
# # init_step_size = 0.1
# # NOTE: In the preceding experiment, we have seen that that ... converges to an error of about 1e-6.
# #       This is a stationary point in this model class, which can be seen by choosing an even smaller step size,
# #       which still yields the same error.
# # TODO: Uncomment the following line to perform this experiment.
# # init_step_size = 0.001
# # NOTE: An intuitive idea is thus to use a decreasing step size after the first few iterations.
# #       But, interestingly, the GD converges to a suboptimal stationary point in these first iterations
# #       and can not escape this point when the step size is reduced in later.
# # TODO: Uncomment the following line to perform this experiment.
# # init_step_size = 0.1
# # step_size_rule = "decreasing"
# # limit_epoch = 7


# # ====================
# # Experiment 3
# # ====================
# # NOTE: In the preceding experiments, we have seen that the basis functions of the tanh-activation
# #       look like polynomials of bounded degree. Consequently, the optimal sampling density looks
# #       very similar to the Legendre case. But this happens due to the random initialisation
# #       and only remains during the optimisation for the smooth sin target.
# #       For the step function target, the optimal density becomes more and more peaky around the jump point.
# target = lambda x: 1e-4 + (x <= (1 / jnp.pi))
# input_dimension = 1
# width = 10
# output_dimension = 1
# activation = lambda x: (jnp.tanh(x) + 1) / 2
# finite_difference = 0
# method = "NGD_quasi_projection"
# sample_size = 300
# sampling = "uniform"
# num_epochs = 15
# step_size_rule = "decreasing"
# limit_epoch = 7
# init_step_size = 0.01
# epoch_length = 100


# ====================
# Experiment 4
# ====================
target = lambda x: 1e-4 + (x <= (1 / jnp.pi))
input_dimension = 1
width = 10
output_dimension = 1
activation = lambda x: (jnp.tanh(x) + 1) / 2
finite_difference = 0
method = "NGD_quasi_projection"
sample_size = 300
sampling = "uniform"
num_epochs = 15
step_size_rule = "decreasing"
limit_epoch = 2
# NOTE: We can use a larger initial step size.
init_step_size = 1
epoch_length = 100


# TODO: Add an experiment that shows that SGD with 300 samples is worse than the new NGD!


# ====================
# Experiment 5
# ====================
# NOTE: Annoyingly, as we have already seen above, there exist many local minima and if we use optimal sampling, we just get stuck in a very bad one.
# This means that it is not easy to compare how fast we converge with uniform vs optimal sampling, since we converge to different points.
# TODO: This makes it extremely difficult to compare two algorihtms!
target = lambda x: 1e-4 + (x <= (1 / jnp.pi))
input_dimension = 1
width = 10
output_dimension = 1
activation = lambda x: (jnp.tanh(x) + 1) / 2
finite_difference = 0
method = "NGD_quasi_projection"
sample_size = 300
sampling = "optimal"
num_epochs = 15
step_size_rule = "decreasing"
limit_epoch = 2
init_step_size = 1
epoch_length = 100


# ====================
# Experiment 6
# ====================
# NOTE: Here we try what happens when we use optimal sampling instead.
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


# ======================
#     Old parameters
# ======================


# # target = lambda x: jnp.sin(2 * jnp.pi * x)
# target = lambda x: 1e-4 + (x <= (1 / jnp.pi))


# input_dimension = 1
# width = 10
# # NOTE: All sample size and epoch lenght parameters below are chosen for a width of 10.
# #       Using a larger width means that the approximation error is smaller.
# #       But we observe that with the previously well-chosen step size 0.01,
# #       the parameters converge to a suboptimal stationary point (loss: 1e-1),
# #       which chould also be achieved with width 10 (basis dimension == 3).
# # NOTE: Actually, not even width 10 achieves a global minimum,
# #       since the tangent space at the stationary point is still 7-dimensional.
# # width = 100
# output_dimension = 1
# activation = lambda x: (jnp.tanh(x) + 1) / 2
# # NOTE: Interestingly, the basis functions of the tanh-activation look like polynomials of bounded degree.
# #       Consequently, the optimal sampling density looks very similar to the Legendre case.
# # NOTE: But this happens due to the random initialisation and only remains during the optimisation for the smooth sin target.
# #       For the step function target, the optimal density becomes more and more peaky around the jump point.
# # activation = lambda x: 1 / (1 + jnp.exp(-x))
# # activation = lambda x: jnp.maximum(x, 0)

# # finite_difference = 0
# finite_difference = 0.01

# # method = "SGD"
# method = "NGD_quasi_projection"
# # method = "NGD_projection"
# # sample_size = 10
# sample_size = 300
# sampling = "uniform"
# # sampling = "optimal"
# num_epochs = 15
# # NOTE: The following step size is too large and yields convergence to a suboptimal stationary point.
# # step_size_rule = "constant"
# # step_size_rule = "constant_epochs"
# step_size_rule = "decreasing"
# limit_epoch = 7
# # init_step_size = 0.1
# init_step_size = 0.01
# epoch_length = 100
# # NOTE: Apparently, 1e-6 is the approximation error in this model class,
# #       since a smaller step size yields the same error.
# # step_size = 0.001
# # epoch_length = 1_000
# # NOTE: An intuitive idea is thus to use a step size schedule.
# #       But, interestingly, the GD converges to a suboptimal stationary point,
# #       which can not be escaped when the step size is reduced in later epochs.
# # num_epochs = 4
# # step_size_list = [1, 0.1, 0.01, 0.001]
# # assert len(step_size_list) == num_epochs
# # epoch_length = 50


num_parameters = output_dimension + output_dimension * width + width + width * input_dimension


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


def loss(parameters, xs, ys, ws):
    return 0.5 * jnp.mean(ws * (prediction(parameters, xs) - ys)**2)


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
            system = jnp.concatenate(system, axis=0)
            assert system.shape == (num_parameters, xs.shape[1])
            return system

    return evaluate_generating_system


def gramian(evaluate_basis, n=1_000):
    assert input_dimension == 1
    xs = jnp.linspace(0, 1, n).reshape(1, n)
    measures = evaluate_basis(xs).T
    return jsp.integrate.trapezoid(measures[:, :, None] * measures[:, None, :], xs[0], axis=0)


def squared_l2_norm(coefficients, gram):
    return coefficients @ gram @ coefficients


def basis_transform(system, n=1_000):
    gram = gramian(system, n=n)
    r = jnp.linalg.matrix_rank(gram)
    s, V = jnp.linalg.eigh(gram)
    return s[::-1], V.T[::-1], r


def basis(parameters, n=1_000):
    system = generating_system(parameters, fd=finite_difference)
    s, Vt, r = basis_transform(system)
    s, Vt = s[:r], Vt[:r]
    X = Vt / jnp.sqrt(s)[:, None]
    return system, X, r


# def basis_old(parameters, n=1_000):
#     system = generating_system(parameters, fd=finite_difference)
#     gram = gramian(system, n=n)
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
    gram = gramian(system, n=1_000)
    qs = system(xs) * ws @ grad[0] / sample_size
    # NOTE: By Leibniz integral rule, the preceding line is equivalent to
    # qs = jnp.concatenate([p.ravel() for p in jax.grad(loss)(parameters, xs, ys, ws)])
    # NOTE: This indeed holds for all sufficiently regular loss functions.
    #       This means that the quasi-projection algorithm is EXACTLY equivalent to NGD for L2,
    #       when the update is performed by utilising Taylors theorem.
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
    system = generating_system(parameters, fd=finite_difference)
    gram = gramian(system, n=1_000)
    vectorised_parameters = lambda ps: jnp.concatenate([p.ravel() for p in ps])  # TODO: Make uniform...
    gradient_norm = jnp.sqrt(squared_l2_norm(vectorised_parameters(gradients), gram))
    return [θ - step_size * dθ for (θ, dθ) in zip(parameters, gradients)], gradient_norm


assert input_dimension == 1
xs = jnp.linspace(0, 1, 1000).reshape(1, 1000)
ws = jnp.ones((1000,))
ys = target(xs)
def plot_state(title):
    plt.plot(xs[0], ys[0], "k-", lw=2)
    zs = prediction(parameters, xs)
    plt.plot(xs[0], zs[0], "k--", lw=2)
    system, transform, basis_dimension = basis(parameters)
    ks = 0
    for bs in transform @ system(xs):
        plt.plot(xs[0], bs, lw=1)
        assert bs.shape == (xs.shape[1],)
        ks += bs**2
    ks /= basis_dimension
    plt.plot(xs[0], ks, "-", color="tab:red", lw=2)
    plt.title(title)
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


*_, basis_dimension = basis(parameters)
plot_state(f"Initialisation  |  Loss: {latex_float(loss(parameters, xs, ys, ws), places=2)}  |  Basis dimension: {basis_dimension}")
# exit()
losses = []
gradients = []
variation_constants = []
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
        losses.append(loss(parameters, xs, ys, ws))
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

        if losses[-1] / 2 <= 1 / xs.shape[1]:
            new_size = 10**(-int(jnp.floor(jnp.log10(losses[-1] / 2))))
            assert new_size > xs.shape[1]
            xs = jnp.linspace(0, 1, new_size).reshape(1, new_size)
            ws = jnp.ones((new_size,))
            ys = target(xs)
        system = generating_system(parameters)
        grad = gradient(parameters, xs, ys)
        grad = jnp.trapz(system(xs) * ws * grad[0], xs, axis=1)
        gram = gramian(system, n=xs.shape[1])
        basis_dimension = jnp.linalg.matrix_rank(gram)
        gradient_norm = jnp.sqrt(squared_l2_norm(grad, gram))
        gradients.append(gradient_norm)
        parameters, _ = updated_parameters(parameters, xs_train, ys_train, ws_train, step_size)
        # NOTE: The gradient norm returned by updated_parameters(...) is not a valid indicator of a stationary point,
        #       since it is the L2 norm of the estimated projected gradient.
        #       This estiamte may not be zero even though the true projected gradient is.
        variation_constants.append(jnp.max(ps * basis_dimension))
        print(f"[{epoch+1:{len(str(num_epochs))}d} | {step+1:{len(str(epoch_length))}d}] Loss: {losses[-1]:.2e}  |  Gradient norm: {gradient_norm:.2e}  |  Step size: {step_size:.2e}  |  Basis dimension: {basis_dimension}")
    # plot_state(f"Epoch {epoch+1}  |  Loss: {latex_float(losses[-1], places=2)}  |  Basis dimension: {basis_dimension}")
plot_state(f"Termination  |  Loss: {latex_float(losses[-1], places=2)}  |  Basis dimension: {basis_dimension}")


fig, ax = plt.subplots(1, 2)
ax[0].plot(xs[0], ys[0], "k-", lw=2)
zs = prediction(parameters, xs)
ax[0].plot(xs[0], zs[0], "k--", lw=2)
ax[1].plot(gradients, color="tab:green", lw=1, alpha=0.5, label="Gradient norm")
ax[1].plot(losses, color="tab:blue", label="Loss")
ax[1].plot(variation_constants, color="tab:red", label=r"$\|\mathfrak{K}\|_{L^\infty}$")
ax[1].legend(loc="upper right")
ax[1].set_xscale("log")
ax[1].set_yscale("log")
plt.show()
