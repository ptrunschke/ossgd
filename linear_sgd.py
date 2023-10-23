import argparse
import os

import jax
import jax.numpy as jnp
import numpy as np
from numpy.polynomial.legendre import legval
from fourier import fourval
from exponential import compute_exponential_coefficients

from tqdm import trange
import matplotlib.pyplot as plt

descr = """Create a convergence plot for L2-SGD on a linear space."""
parser = argparse.ArgumentParser(description=descr)
parser.add_argument("-b", dest="basis", type=str, choices=["fourier", "legendre"], default="legendre", help="basis of the linear space")
parser.add_argument("-d", dest="space_dimension", type=int, default=10, help="dimension of the linear space")
parser.add_argument("-T", dest="target", type=str, choices=["random", "exp"], default="random", help="target function to use")
parser.add_argument("-t", dest="target_dimension", type=int, default=10, help="dimension needed to represent the target function")
#TODO: add a "stratified" sampling strategy
parser.add_argument("-Z", dest="sampling_strategy", type=str, choices=["optimal", "boosted", "uniform"], default="optimal", help="sampling density to use")
parser.add_argument("-z", dest="sample_size", type=int, default=1, help="number of sample points to use per iteration")
parser.add_argument("-S", dest="stability", type=float, default=jnp.inf, help="stability condition to satisfy (inf means no stability)")
parser.add_argument("-s", dest="step_size", type=str, default=1, help="step size rule to use")
parser.add_argument("-I", dest="initialisation", type=str, choices=["random", "least-squares"], default="random", help="initialisation strategy to use")
parser.add_argument("-i", dest="iterations", type=int, default=1000, help="number of iterations to perform")
parser.add_argument("-p", dest="projection", type=str, choices=["quasi", "least-squares"], default="quasi", help="the projection to use")
args = parser.parse_args()

try:
    args.step_size = float(args.step_size)
except ValueError:
    assert args.step_size in ["constant", "adaptive", "deterministic", "deterministic_unbounded", "nouy"], args.step_size

# xi_t = 1 / t^((1 + epsilon)/2)
epsilon = 0.1
# epsilon = 0.01
# epsilon = 0.5 - 0.01
assert 0 < epsilon < 0.5
domain = (-1, 1)
loss_density = lambda x: jnp.full(x.shape, 0.5)


os.makedirs("data", exist_ok=True)
os.makedirs("plots", exist_ok=True)
if args.step_size in ["constant", "adaptive", "deterministic", "deterministic_unbounded", "nouy"]:
    step_size_str = args.step_size
else:
    len_step_size_str = max(2, int(jnp.ceil(-jnp.log10(args.step_size))))
    step_size_str = f"{args.step_size:.{len_step_size_str}f}".replace(".", "-")
if args.stability == jnp.inf:
    stability_str = "inf"
else:
    stability_str = f"{args.stability:.2f}".replace(".", "-")
density_file_name = f"sgd_density_{args.basis}{args.space_dimension}_{args.sampling_strategy}.png"
rate_file_base = f"sgd_{args.basis}{args.space_dimension}_t{args.target_dimension}_{args.sampling_strategy}{args.sample_size}_S{stability_str}_s{step_size_str}_i{args.iterations}_{args.projection}"


def evaluate_legendre(points, coefs):
    dimension = coefs.shape[0]
    factors = np.sqrt(2 * np.arange(dimension) + 1)
    return jnp.asarray(legval(np.asarray(points), factors * np.asarray(coefs)).T, dtype=jnp.float32)


def evaluate_fourier(points, coefs):
    # We need to transform the points, since fourval is orthogonal on the interval (0,1).
    return jnp.asarray(fourval((points+1)/2, coefs).T, dtype=jnp.float32)


if args.basis == "legendre":
    evaluate_basis = evaluate_legendre
else:
    assert args.basis == "fourier"
    evaluate_basis = evaluate_fourier


def trapz(xs):
    assert xs.ndim == 1
    ds = 0.5 * jnp.diff(xs)
    return jnp.zeros(xs.shape[0]).at[:-1].add(ds).at[1:].add(ds)


xs = jnp.linspace(-1, 1, 10_000)
ps = loss_density(xs)
integral_weights = trapz(xs)
assert jnp.allclose(ps @ integral_weights, 1, atol=1/len(xs))


if __debug__:
    measures = evaluate_basis(xs, jnp.eye(5))
    gramian = measures.T * ps * integral_weights @ measures
    assert jnp.allclose(gramian, np.eye(5), atol=jnp.finfo(gramian.dtype).resolution)


key = jax.random.PRNGKey(1)
if args.target == "random":
    target_key, key = jax.random.split(key, 2)
    target_coefficients = jax.random.normal(key, shape=(args.target_dimension,))
    target_coefficients.at[:args.space_dimension].divide(jnp.linalg.norm(target_coefficients[:args.space_dimension]))
    target_coefficients.at[args.space_dimension:].divide(jnp.linalg.norm(target_coefficients[args.space_dimension:]))
else:
    assert args.target == "exp"
    target_coefficients = compute_exponential_coefficients(degree=args.target_dimension-1)
loss_gradient = lambda v: lambda x: evaluate_basis(x, v) - evaluate_basis(x, target_coefficients)
minimal_loss = 0.5 * jnp.linalg.norm(target_coefficients[args.space_dimension:])**2
print(f"Minimal loss: {minimal_loss:.2e}")
error = lambda v: 0.5 * jnp.linalg.norm(v - target_coefficients[:args.space_dimension])**2
loss = lambda v: minimal_loss + error(v)


# ========= Compute the optimal sampling density =========

assert 0 < args.stability
assert args.stability == jnp.inf or args.sampling_strategy in ["optimal", "boosted"]
if args.sampling_strategy in ["optimal", "boosted"]:
    pdf = jnp.sum(measures ** 2, axis=1) * ps
    cdf = jnp.cumsum(integral_weights * pdf)
    pdf /= cdf[-1]
    cdf /= cdf[-1]

if args.sampling_strategy == "optimal":
    def draw_sample(key, size, stability):
        assert size > 0
        assert 0 < stability < 1 or stability == jnp.inf
        assert stability == jnp.inf or size > args.space_dimension
        while True:
            us_key, key = jax.random.split(key, 2)
            us = jax.random.uniform(us_key, shape=(size,))
            indices = jnp.searchsorted(cdf, us)
            points = xs[indices]
            weights = (ps / pdf)[indices]
            M = evaluate_basis(points, jnp.eye(args.space_dimension))
            G = M.T * weights @ M / size
            assert G.shape == (args.space_dimension, args.space_dimension)
            sample_stability = jnp.linalg.norm(G - jnp.eye(args.space_dimension), ord=2)
            if sample_stability <= stability:
                break
        return points, weights, sample_stability
elif args.sampling_strategy == "boosted":
    def draw_sample(key, size, stability):
        assert size > 0
        assert stability == jnp.inf, stability
        best_trial = (None, None, np.inf)
        for m in range(20):
            us_key, key = jax.random.split(key, 2)
            us = jax.random.uniform(us_key, shape=(size,))
            indices = jnp.searchsorted(cdf, us)
            points = xs[indices]
            weights = (ps / pdf)[indices]
            M = evaluate_basis(points, jnp.eye(args.space_dimension))
            G = M.T * weights @ M / size
            assert G.shape == (args.space_dimension, args.space_dimension)
            sample_stability = jnp.linalg.norm(G - jnp.eye(args.space_dimension), ord=2)
            if sample_stability <= best_trial[2]:
                best_trial = (points, weights, sample_stability)
        return best_trial
else:
    assert args.sampling_strategy == "uniform"
    pdf = jnp.full_like(xs, 0.5)
    cdf = jnp.cumsum(integral_weights * pdf)
    pdf /= cdf[-1]
    cdf /= cdf[-1]

    def draw_sample(key, size, stability):
        assert size > 0
        assert stability == jnp.inf
        points = jax.random.uniform(key, (size,), minval=-1, maxval=1)
        weights = jnp.ones(size)
        M = evaluate_basis(points, jnp.eye(args.space_dimension))
        G = M.T * weights @ M / size
        assert G.shape == (args.space_dimension, args.space_dimension)
        sample_stability = jnp.linalg.norm(G - jnp.eye(args.space_dimension), ord=2)
        return points, weights, sample_stability


sample_key, key = jax.random.split(key, 2)
points, weights, sample_stability = draw_sample(sample_key, 1_000, np.inf)
fig, ax = plt.subplots(1, 1, figsize=(5, 5))
ax.plot(xs, pdf, label="pdf")
ax.plot(xs, cdf, label="cdf")
ax.hist(points, 40, density=True, histtype='step', label="sample points histogram")
ax.plot(points, weights, ".", label="sampe weights")
ax.legend()
ax.set_title(f"Sampling density (sample stability: ${sample_stability:.2f}$)")
plt.tight_layout()
print("Saving density plot to", density_file_name)
plt.savefig(f"plots/{density_file_name}")


L = 1
C = 0
μ = 1

d = args.space_dimension
n = args.sample_size

if args.sampling_strategy == "uniform":
    if args.basis == "legendre":
        V = jnp.sum(2*jnp.arange(d) + 1)
    else:
        assert args.basis == "fourier"
        V = d  # == jnp.sum(jnp.ones(d))
else:
    assert args.sampling_strategy in ["optimal", "boosted"]
    V = d

f = (L+C)/2 * (1 + (V-1)/n)
maximal_step_size = 1/f
print(f"Maximal step size: {maximal_step_size:.2e}")
optimal_step_size = maximal_step_size / 2


if args.step_size == "constant":
    def step_size(iteration, squared_gradient_norm):
        return optimal_step_size
elif args.step_size == "deterministic":
    def step_size(iteration, squared_gradient_norm):
        return maximal_step_size / iteration**(1 - epsilon)
elif args.step_size == "nouy":
    def step_size(iteration, squared_gradient_norm):
        return 1. / iteration**0.8
elif args.step_size == "deterministic_unbounded":
    def step_size(iteration, squared_gradient_norm):
        return 1 / iteration**(1 - epsilon)
elif args.step_size == "adaptive":
    step_size_factor = jnp.nan
    def step_size(iteration, squared_gradient_norm, indep_est):
        global step_size_factor
        s_t = 1 / iteration**(0.5 + epsilon)
        # s_t = 1 / iteration**(1 - epsilon)
        if jnp.isnan(step_size_factor):
            assert s_t == 1
            step_size_factor = jnp.sqrt(squared_gradient_norm)
        bias_step_size = s_t / jnp.sqrt(squared_gradient_norm)
        recovery_step_size = optimal_step_size * bias_step_size * jnp.sqrt(jnp.minimum(squared_gradient_norm, indep_est))
        return recovery_step_size
else:
    def step_size(iteration, _):
        return args.step_size


init_key, key = jax.random.split(key, 2)
if args.initialisation == "least-squares":
    # points, weights, sample_stability = draw_sample(init_key, args.sample_size, 0.5)
    points, weights, sample_stability = draw_sample(init_key, args.sample_size, args.stability)
    assert sample_stability < 1
    measures = evaluate_basis(points, jnp.eye(args.space_dimension))
    estimate = loss_gradient(jnp.zeros(args.space_dimension))(points) * weights @ measures / args.sample_size
    gramian = measures.T * weights @ measures / args.sample_size
    estimate = -jnp.linalg.solve(gramian, estimate)
else:
    assert args.initialisation == "random"
    estimate = jax.random.normal(init_key, shape=(args.space_dimension,))
errors = np.zeros(args.iterations + 1)
errors[0] = loss(estimate)
kappas = np.zeros(args.iterations + 1)
kappas[0] = np.nan
descent = np.zeros(args.iterations + 1)
descent[0] = np.nan
step_sizes = np.zeros(args.iterations + 1)
step_sizes[0] = np.nan
# sample_stabilities = np.zeros(iterations + 1)
# sample_stabilities[0] = np.nan
for it in trange(1, args.iterations + 1, desc="Iteration"):
    squared_projected_gradient_norm = jnp.linalg.norm(target_coefficients[:args.space_dimension] - estimate)**2
    squared_gradient_norm = squared_projected_gradient_norm + jnp.linalg.norm(target_coefficients[args.space_dimension:])**2
    # assert jnp.allclose(squared_gradient_norm, loss_gradient(estimate)(xs)**2 * ps @ integral_weights, atol=1e-4, rtol=1e-3)
    # assert jnp.allclose(0.5 * loss_gradient(estimate)(xs)**2 * ps @ integral_weights, loss(estimate), atol=1e-4, rtol=1e-3)
    kappas[it] = squared_gradient_norm / squared_projected_gradient_norm

    sample_key, key = jax.random.split(key, 2)
    points, weights, sample_stability = draw_sample(sample_key, args.sample_size, args.stability)
    if args.projection == "quasi":
        qp_update = loss_gradient(estimate)(points) * weights @ evaluate_basis(points, jnp.eye(args.space_dimension)) / args.sample_size
    else:
        assert args.projection == "least-squares"
        assert sample_stability < 1
        measures = evaluate_basis(points, jnp.eye(args.space_dimension))
        qp_update = loss_gradient(estimate)(points) * weights @ measures / args.sample_size
        gramian = measures.T * weights @ measures / args.sample_size
        qp_update = jnp.linalg.solve(gramian, qp_update)
    descent[it] = qp_update @ (estimate - target_coefficients[:args.space_dimension])
    squared_gradient_norm_estimate = loss_gradient(estimate)(points)**2 @ weights
    if args.step_size == "adaptive":
        points_2 = jax.random.uniform(key, (args.sample_size,), minval=-1, maxval=1)
        squared_gradient_norm_estimate_2 = jnp.mean(loss_gradient(estimate)(points_2)**2)
        s = step_size(it, squared_gradient_norm_estimate, squared_gradient_norm_estimate_2)
    else:
        s = step_size(it, squared_gradient_norm_estimate)
    estimate -= s * qp_update
    # print(s, jnp.linalg.norm(qp_update), jnp.sqrt(squared_gradient_norm_estimate))
    errors[it] = loss(estimate)
    step_sizes[it] = s
    # sample_stabilities[it] = sample_stability

np.savez_compressed(f"data/{rate_file_base}.npz", losses=errors, minimal_loss=minimal_loss, step_sizes=step_sizes)

fig, ax = plt.subplots(1, 1, figsize=(5, 5))

steps = jnp.arange(1, len(errors)+1)
ax.loglog(steps, errors - minimal_loss, color="tab:blue", label="loss")
# ax.loglog(steps, 2 * (errors - minimal_loss) / jnp.linalg.norm(target_coefficients)**2, color="tab:blue", label="relative error")
# ax.axhline(2 * minimal_loss / jnp.linalg.norm(target_coefficients)**2, color="tab:purple", label="relative bias limit")
ax.loglog(steps, step_sizes, color="tab:orange", label="step size")
# if args.step_size == "adaptive":
#     cs = exponential_convergence_factor(step_sizes, kappas)
#     ax.loglog(steps, cs, color="tab:purple", label="$c(\kappa)$")
ax.loglog(steps, 1 / np.sqrt(steps), "k:", label="$t^{-1/2}$ rate")
ax.loglog(steps, 1 / steps, "k-.", label="$t^{-1}$ rate")
ax.set_xlabel("step")
ax.set_xlim(steps[0], steps[-1])
ax.legend(loc="lower left")

# TODO:
# - dont plot against the steps but against the number of samples used.
# - be aware of the initialisation!

plt.tight_layout()
print("Saving convergence plot to", rate_file_base+".png")
plt.savefig(f"plots/{rate_file_base}.png")
