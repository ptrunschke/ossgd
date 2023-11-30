import argparse
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from numpy.polynomial import Legendre
from numpy.polynomial.legendre import legval
from fourier import fourval
from scipy import integrate

from tqdm import trange
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

descr = """Create a convergence plot for L2-SGD on a linear space."""
parser = argparse.ArgumentParser(description=descr)
parser.add_argument("-b", dest="basis", type=str, choices=["fourier", "legendre"], default="legendre", help="basis of the linear space")
parser.add_argument("-d", dest="space_dimension", type=int, default=10, help="dimension of the linear space")
parser.add_argument("-T", dest="target", type=str, default="random", help="target function to use")
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
    assert args.step_size in ["constant", "adaptive", "deterministic", "deterministic_unbounded", "nouy", "sls", "est", "mixed"], args.step_size

# xi_t = 1 / t^((1 + epsilon)/2)
epsilon = 0.1
# epsilon = 0.01
# epsilon = 0.5 - 0.01
assert 0 < epsilon < 0.5
domain = (-1, 1)
loss_density = lambda x: jnp.full(x.shape, 0.5)

if type(args.step_size) == str:
    step_size_str = args.step_size
else:
    len_step_size_str = max(2, int(jnp.ceil(-jnp.log10(args.step_size))))
    step_size_str = f"{args.step_size:.{len_step_size_str}f}".replace(".", "-")
if args.stability == jnp.inf:
    stability_str = "inf"
else:
    stability_str = f"{args.stability:.2f}".replace(".", "-")

file_base = Path(f"{args.basis}{args.space_dimension}_{args.target}{args.target_dimension}_{args.sampling_strategy}{args.sample_size}_S{stability_str}_{args.projection}_{args.iterations}x{step_size_str}")
data_path = Path("data") / "compact_domain"
data_path.mkdir(exist_ok=True)
data_path /= file_base.with_suffix(".npz")
plot_path = Path("plots") / "compact_domain"
plot_path.mkdir(exist_ok=True)
plot_path /= file_base.with_suffix(".pdf")


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


xs = jnp.linspace(domain[0], domain[1], 10_000)
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
    assert args.basis == "legendre"

    if args.target == "sin":
        def sin(x):
            return np.sin(2 * np.pi * x)
        target_function = sin
    else:
        target_function = getattr(np, args.target)
    print(f"Target function: {target_function}")

    def legendre_coefficients(function, dimension):
        assert dimension > 0
        factors = np.sqrt(2 * np.arange(dimension) + 1)
        coefficients = np.empty(dimension)
        for degree in range(dimension):
            basis_function = factors[degree] * Legendre.basis(degree)
            integrand = lambda x: 0.5 * basis_function(x) * function(x)
            coefficients[degree], _ = integrate.quadrature(integrand, *domain, tol=1e-12, rtol=1e-12)
        return coefficients

    target_coefficients = legendre_coefficients(target_function, args.target_dimension)

loss_integrand = lambda v: lambda x: 0.5 * (evaluate_basis(x, v) - evaluate_basis(x, target_coefficients))**2
loss_gradient = lambda v: lambda x: evaluate_basis(x, v) - evaluate_basis(x, target_coefficients)
minimal_loss = 0.5 * jnp.linalg.norm(target_coefficients[args.space_dimension:])**2
print(f"Minimal loss: {minimal_loss:.2e}")
if args.target != "random":
    approximation_loss = 0.5 * (target_function(xs) - evaluate_basis(xs, target_coefficients))**2 @ (ps * integral_weights)
    print(f"Approximation loss: {approximation_loss:.2e}")
    assert approximation_loss <= minimal_loss
error = lambda v: 0.5 * jnp.linalg.norm(v - target_coefficients[:args.space_dimension])**2
loss = lambda v: minimal_loss + error(v)


# ========= Compute the optimal sampling density =========

assert 0 < args.stability
assert args.stability == jnp.inf or args.sampling_strategy in ["optimal", "boosted"]
if args.sampling_strategy in ["optimal", "boosted"]:
    measures = evaluate_basis(xs, jnp.eye(args.space_dimension))
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
        points = jax.random.uniform(key, (size,), minval=domain[0], maxval=domain[1])
        weights = jnp.ones(size)
        M = evaluate_basis(points, jnp.eye(args.space_dimension))
        G = M.T * weights @ M / size
        assert G.shape == (args.space_dimension, args.space_dimension)
        sample_stability = jnp.linalg.norm(G - jnp.eye(args.space_dimension), ord=2)
        return points, weights, sample_stability


# density_file_name = f"sgd_density_{args.basis}{args.space_dimension}_{args.sampling_strategy}.png"
# sample_key, key = jax.random.split(key, 2)
# points, weights, sample_stability = draw_sample(sample_key, 1_000, np.inf)
# fig, ax = plt.subplots(1, 1, figsize=(5, 5))
# ax.plot(xs, pdf, label="pdf")
# ax.plot(xs, cdf, label="cdf")
# ax.hist(points, 40, density=True, histtype='step', label="sample points histogram")
# ax.plot(points, weights, ".", label="sampe weights")
# ax.legend()
# ax.set_title(f"Sampling density (sample stability: ${sample_stability:.2f}$)")
# plt.tight_layout()
# print("Saving density plot to", density_file_name)
# plt.savefig(f"plots/{density_file_name}")


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


c = n / (V - 1)
eps = jnp.finfo(jnp.float32).eps
target_error = minimal_loss / loss(jnp.zeros(args.space_dimension))
# t_max = min(54, int(np.ceil(np.log2(2 / target_error))))
# TODO: target_error should really be interpreted as a target error
#       and not necessarily as this unknown quantity we have here.
# 2 (1 + c)^(-t) == max(target_error, eps)  <-->  t == ln(2 / max(target_error, eps)) / ln(1 + c)
t_max = np.log(2 / max(target_error, eps)) / np.log(1 + c)
t_max = int(np.ceil(t_max))
assert (1+c)**(-t_max) <= max(target_error, eps)
print(f"ε = {eps:.2e} → t_max = {t_max} → N = {t_max * n}")


if args.step_size == "constant":
    def step_size(iteration, squared_gradient_norm):
        return optimal_step_size
elif args.step_size == "deterministic":
    def step_size(iteration, squared_gradient_norm):
        return maximal_step_size / iteration**(1 - epsilon)
elif args.step_size == "nouy":
    def step_size(iteration, squared_gradient_norm):
        return 1. / iteration**0.8
elif args.step_size == "mixed":
    def step_size(iteration, _):
        # return optimal_step_size / max(iteration - t_max, 1)**(1 - epsilon)
        return min(maximal_step_size / max(iteration - t_max, 1)**(1 - epsilon), optimal_step_size)
elif args.step_size == "deterministic_unbounded":
    # factor = 1
    # factor = 0.5
    # factor = 0.25
    # factor = 0.1
    # factor = optimal_step_size
    factor = maximal_step_size
    def step_size(iteration, squared_gradient_norm):
        return factor / iteration**(1 - epsilon)
    crit = jnp.ceil(factor**(-1/epsilon))
    print(f"Critical iteration: {crit:.0f} (theoretical)")
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
elif args.step_size == "sls":
    # from noisyopt import minimizeCompass

    def step_size(key, iterate, update):
        objective_key = key
        def objective(s):
            nonlocal objective_key
            points_key, objective_key = jax.random.split(objective_key)
            points = jax.random.uniform(points_key, (args.sample_size,), minval=domain[0], maxval=domain[1])
            return np.mean(loss_integrand(iterate - s * update)(points))
        bounds = np.array([0, maximal_step_size]).reshape(1, 2)
        # x0 = np.array([optimal_step_size])
        # res = minimizeCompass(objective, bounds=bounds, x0=x0, errorcontrol=True, paired=False)
        # return res.x, res.nfev * args.sample_size
        ss = np.linspace(bounds[0, 0], bounds[0, 1], 20)
        os = np.array([[objective(s) for s in ss] for _ in range(100)])
        assert os.shape == (100, 20)
        # plt.plot(ss, np.mean(os, axis=0), 'k-', linewidth=2)
        # for oss in os: 
        #     plt.plot(ss, oss, "o", color="C0", alpha=0.5)
        # plt.plot(optimal_step_size, objective(optimal_step_size), "o", color="tab:red")
        # plt.show()

        idx_min = np.argmin(np.mean(os, axis=0))
        ss[idx_min]
        return ss[idx_min]

elif args.step_size == "est":
    def step_size(iteration, key, iterate, update):
        # points, weights, _ = draw_sample(key, args.sample_size, args.stability)
        # independent_update = loss_gradient(iterate)(points) * weights @ evaluate_basis(points, jnp.eye(args.space_dimension)) / args.sample_size
        independent_update = iterate - target_coefficients[:args.space_dimension]
        b = (L + C) / 2 * update.T @ update
        a = independent_update.T @ update
        s_min = 0
        # s_max = optimal_step_size
        s_max = np.inf
        # s_min = maximal_step_size / iteration**(0.5 + epsilon)
        # s_max = maximal_step_size / iteration**(1 - epsilon)
        return np.clip(a / (2*b), s_min, s_max)
        
else:
    assert isinstance(args.step_size, float)

    def step_size(iteration, _):
        return args.step_size


if data_path.exists():
    z = np.load(data_path)
    errors = z["errors"]
    kappas = z["kappas"]
    step_sizes = z["step_sizes"]
else:
    try:
        # Find the iteration where step_size(iteration) >= 1 / iteration.
        its = jnp.arange(1, args.iterations+1)
        sss = jnp.array([step_size(it, None) for it in its])
        assert jnp.all(jnp.diff(sss) <= 0)
        crit = jnp.count_nonzero(sss <= 1 / its) + 1
        if sss[crit] >= 1 / its[crit]:
            print(f"Critical iteration: {crit}")
        else:
            print(f"Critical iteration not reachable.")
    except:
        print(f"Critical iteration not computable.")


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
    kappas = np.zeros(args.iterations)
    step_sizes = np.zeros(args.iterations)
    for it in trange(1, args.iterations + 1, desc="Iteration"):
        squared_projected_gradient_norm = jnp.linalg.norm(target_coefficients[:args.space_dimension] - estimate)**2
        squared_gradient_norm = squared_projected_gradient_norm + jnp.linalg.norm(target_coefficients[args.space_dimension:])**2
        # assert jnp.allclose(squared_gradient_norm, loss_gradient(estimate)(xs)**2 * ps @ integral_weights, atol=1e-4, rtol=1e-3)
        # assert jnp.allclose(0.5 * loss_gradient(estimate)(xs)**2 * ps @ integral_weights, loss(estimate), atol=1e-4, rtol=1e-3)
        kappas[it - 1] = squared_gradient_norm / squared_projected_gradient_norm
        # |Pg|^2 / |(I-P)g|^2 == (|(I-P)g|^2 / |Pg|^2)^{-1} == (|g|^2 / |Pg|^2 - 1)^{-1}
        kappas[it - 1] = 1 / (kappas[it - 1] - 1)

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
        squared_gradient_norm_estimate = loss_gradient(estimate)(points)**2 @ weights

        # test_points = jax.random.uniform(key, (100,), minval=domain[0], maxval=domain[1])
        # evaluate_gradient = evaluate_basis(test_points, estimate) - evaluate_basis(test_points, target_coefficients)
        # evaluate_projected_gradient = evaluate_basis(test_points, qp_update)
        # kappas[it] = jnp.linalg.norm(qp_update)**2 / jnp.mean((evaluate_gradient - evaluate_projected_gradient)**2)

        if args.step_size == "sls":
            step_size_key, key = jax.random.split(key, 2)
            s = step_size(key, estimate, qp_update)
        elif args.step_size == "est":
            step_size_key, key = jax.random.split(key, 2)
            s = step_size(it, key, estimate, qp_update)
        elif args.step_size == "adaptive":
            points_2 = jax.random.uniform(key, (args.sample_size,), minval=domain[0], maxval=domain[1])
            squared_gradient_norm_estimate_2 = jnp.mean(loss_gradient(estimate)(points_2)**2)
            s = step_size(it, squared_gradient_norm_estimate, squared_gradient_norm_estimate_2)
        else:
            s = step_size(it, squared_gradient_norm_estimate)
        step_sizes[it - 1] = s
        estimate -= s * qp_update
        errors[it] = loss(estimate)

    np.savez_compressed(data_path, errors=errors, minimal_loss=minimal_loss, kappas=kappas, step_sizes=step_sizes)

fig, ax = plt.subplots(1, 1, figsize=(4, 4), dpi=300)

steps = jnp.arange(len(errors))
ax.plot(steps, errors - minimal_loss, color="tab:blue", label=r"$\mathcal{L}(u_t) - \mathcal{L}_{\mathrm{min},\mathcal{M}}$")
# ax.loglog(steps, 2 * (errors - minimal_loss) / jnp.linalg.norm(target_coefficients)**2, color="tab:blue", label="relative error")
# ax.axhline(2 * minimal_loss / jnp.linalg.norm(target_coefficients)**2, color="tab:purple", label="relative bias limit")
ax.plot(steps[1:], step_sizes, color="tab:orange", label=r"$s_t$")

# ax.plot(steps[:-1], kappas[1:], color="tab:green", label=r"$\kappa$")
# if args.stability < 1 and args.projection == "least-squares":
#     bias_factor = jnp.sqrt(args.space_dimension / args.sample_size) / (1 - args.stability)
#     ax.axhline(bias_factor, lw=1, ls="--", color="tab:green", alpha=0.5)

ax.set_yscale("log")
ax.set_xscale("symlog", linthresh=1, linscale=0.1)

if not (args.step_size == "constant" or isinstance(args.step_size, float)):
    ax.plot(steps[1:], 1 / np.sqrt(steps[1:]), "k:", label="$t^{-1/2}$ rate")
    ax.plot(steps[1:], 1 / steps[1:], "k-.", label="$t^{-1}$ rate")

if (args.step_size in ["constant", "mixed"] or isinstance(args.step_size, float)):
    ylim = ax.get_ylim()
    # Find all c such that ylim[0] <= c * minimal_loss <= ylim[1].
    # These are ylim[0] / minimal_loss <= c <= ylim[1] / minimal_loss
    c_min = int(np.ceil(ylim[0] / minimal_loss))
    c_max = int(np.floor(ylim[1] / minimal_loss))
    # c_max = min(c_max, 15)
    c_max = min(c_max, 3)
    for c in range(c_min, c_max):
        ax.axhline(c * minimal_loss, color="tab:red", linestyle="--", linewidth=0.5, zorder=0)
    # label=r"$2\mathcal{L}_{\mathrm{min},\mathcal{M}}$"
    # label = fr"$\mathbb{{N}}_{{{c_max+1}}} \cdot (\text{{minimal loss}})$"
    label = r"$\{1, 2, 3\} \cdot \mathcal{L}_{\mathrm{min},\mathcal{M}}$"
    ax.axhline(c_max * minimal_loss, color="tab:red", linestyle="--", linewidth=0.5, zorder=0, label=label)

xticks = ax.get_xticks()
xticks = [tick for tick in xticks if tick > 0]
ax.set_xticks(xticks)

if (args.step_size in ["constant", "mixed"] or isinstance(args.step_size, float)) and args.projection == "quasi" and args.stability >= 1:
    ax.axvline(t_max, color="tab:red", linestyle="--", linewidth=0.5, zorder=0)
    # ax.set_xticks(ax.get_xticks().tolist() + [t_max])
    # tick_index = np.where(ax.get_xticks() == t_max)[0]
    # assert len(tick_index) == 1
    # tick_index = tick_index[0]
    # tick_labels = ax.get_xticklabels()
    # tick_labels[tick_index].set_text(r"$\mathdefault{t_{\mathrm{max}}}$")
    # tick_labels[tick_index].set_color("tab:red")
    # tick_labels[tick_index].set_fontweight("bold")
    # tick_labels[tick_index].set_horizontalalignment("left")
    # tick_labels[tick_index].set_verticalalignment("top")
    # ax.set_xticklabels(tick_labels)
    x = 10**(1.05 * np.log10(t_max))
    y = 10**(np.log10(ax.get_ylim()[1]) / 2)
    plt.text(x, y, r"$\mathdefault{\boldsymbol{t_{\mathrm{max}}}}$", color="tab:red", horizontalalignment="left", verticalalignment="top")
ax.set_xlabel("$t$")
ax.set_xlim(steps[0], steps[-1])
ax.legend(loc="lower left")

# TODO:
# - dont plot against the steps but against the number of samples used.
# - be aware of the initialisation!

print("Saving convergence plot to", plot_path)
plt.savefig(
    plot_path, dpi=300, edgecolor="none", bbox_inches="tight", transparent=True
)