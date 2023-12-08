import numpy as np
from tqdm import tqdm, trange

from least_squares import mu_W
from sampling import Sampler


def greedy_subsample(sample, sample_size):
    while len(sample) > sample_size:
        mus = []
        for idx in range(len(sample)):
            sample_without_idx = sample[:idx] + sample[idx + 1 :]
            mus.append(mu_W(sample_without_idx, sampler.dimension, sampler.basis))
        idx = np.argmin(mus)
        sample = sample[:idx] + sample[idx + 1 :]
    return sample

def greedy_swap(sample, candidates):
    mus = []
    for idx in range(len(candidates)):
        sample_with_idx = sample + [candidates[idx]]
        mus.append(mu_W(sample_with_idx, sampler.dimension, sampler.basis))
    idx = np.argmin(mus)
    sample = sample + [candidates[idx]]
    mus = []
    for idx in range(len(sample)):
        sample_without_idx = sample[:idx] + sample[idx + 1 :]
        mus.append(mu_W(sample_without_idx, sampler.dimension, sampler.basis))
    idx = np.argmin(mus)
    sample = sample[:idx] + sample[idx + 1 :]
    return sample

def greedy_optimise(sample_size, sampler, rng, max_iterations=100, tolerance=1e-1):
    sample = []
    while len(sample) < sample_size:
        sample.extend(sampler.draw(rng))
    sample = sample[:sample_size]
    mu_W_new = mu_W(sample, sampler.dimension, sampler.basis)
    for _ in range(max_iterations):
        mu_W_old = mu_W_new
        sample = greedy_swap(sample, sampler.draw(rng))
        mu_W_new = mu_W(sample, sampler.dimension, sampler.basis)
        assert mu_W_old >= mu_W_new
        if mu_W_old - mu_W_new < tolerance:
            break
    return sample, mu_W_new

if __name__ == "__main__":
    from pathlib import Path
    from plotting import plt

    dimension = 10
    rank = 3 * dimension
    discretisation = 1000

    # sample_size = int(2.5 * dimension)
    sample_size = dimension

    rng = np.random.default_rng(0)
    sampler = Sampler(dimension, rank, discretisation)

    sample = []
    while len(sample) < sample_size:
        sample.extend(sampler.draw(rng))
    sample = sample[:sample_size]

    tqdm.write("Start optimisation")
    tqdm.write(f"Initial quasi-optimality factor: {mu_W(sample, sampler.dimension, sampler.basis):.2f}")
    for step in trange(10):
        # sample.extend(sampler.draw(rng))
        # sample = greedy_subsample(sample, sample_size)
        sample = greedy_swap(sample, sampler.draw(rng))
        tqdm.write(f"[{step}] Quasi-optimality factor: {mu_W(sample, sampler.dimension, sampler.basis):.2f}")

    trials = 1_000
    samples = []
    mus = []
    for trial in trange(trials):
        sample, mu = greedy_optimise(sample_size, sampler, rng)
        samples.append(sample)
        mus.append(mu)

    plot_directory = Path(__file__).parent / "plot"
    plot_directory.mkdir(exist_ok=True)

    plot_path = plot_directory / "optimisation_statistics.png"
    fig, ax = plt.subplots(1, 2, figsize=(8, 4), dpi=300)
    ax[0].hist(np.concatenate(samples), density=True, bins=80)
    ax[0].set_title("Sample distribution")
    ax[1].hist(mus, bins=25)
    ax[1].set_title("Quasi-optimality factor")
    print("Saving optimisation statistics plot to", plot_path)
    plt.savefig(
        plot_path, dpi=300, edgecolor="none", bbox_inches="tight", transparent=True
    )
    plt.close(fig)
