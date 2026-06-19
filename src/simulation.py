from __future__ import annotations

import numpy as np


def simulate_jump_diffusion_paths(
    mu: float,
    sigma: float,
    lambda_jump: float,
    jump_mean: float,
    jump_std: float,
    horizon_steps: int,
    n_paths: int,
    seed: int | None = None,
) -> np.ndarray:
    """
    Simulate jump-diffusion log-price paths starting at log-price 0.

    Returns
    -------
    np.ndarray
        Shape (n_paths, horizon_steps + 1), where column 0 is 0.
    """
    rng = np.random.default_rng(seed)

    log_paths = np.zeros((n_paths, horizon_steps + 1), dtype=float)

    for t in range(1, horizon_steps + 1):
        z = rng.standard_normal(n_paths)
        diffusion = (mu - 0.5 * sigma**2) + sigma * z

        jump_counts = rng.poisson(lambda_jump, size=n_paths)
        jump_component = np.zeros(n_paths, dtype=float)
        jumping = jump_counts > 0

        if np.any(jumping):
            jump_component[jumping] = rng.normal(
                loc=jump_counts[jumping] * jump_mean,
                scale=np.sqrt(jump_counts[jumping]) * jump_std,
            )

        log_paths[:, t] = log_paths[:, t - 1] + diffusion + jump_component

    return log_paths


def probability_of_terminal_downside_crash(
    mu: float,
    sigma: float,
    lambda_jump: float,
    jump_mean: float,
    jump_std: float,
    horizon_steps: int,
    n_paths: int,
    threshold: float,
    seed: int | None = None,
) -> float:
    """Probability that terminal simple return is below -threshold."""
    log_paths = simulate_jump_diffusion_paths(
        mu=mu,
        sigma=sigma,
        lambda_jump=lambda_jump,
        jump_mean=jump_mean,
        jump_std=jump_std,
        horizon_steps=horizon_steps,
        n_paths=n_paths,
        seed=seed,
    )
    terminal_simple_returns = np.exp(log_paths[:, -1]) - 1.0
    return float(np.mean(terminal_simple_returns < -threshold))


def probability_of_barrier_hit(
    mu: float,
    sigma: float,
    lambda_jump: float,
    jump_mean: float,
    jump_std: float,
    horizon_steps: int,
    n_paths: int,
    threshold: float,
    seed: int | None = None,
) -> float:
    """
    Probability that simple return hits the downside barrier at any point.
    """
    log_paths = simulate_jump_diffusion_paths(
        mu=mu,
        sigma=sigma,
        lambda_jump=lambda_jump,
        jump_mean=jump_mean,
        jump_std=jump_std,
        horizon_steps=horizon_steps,
        n_paths=n_paths,
        seed=seed,
    )
    simple_returns = np.exp(log_paths) - 1.0
    min_path_returns = np.min(simple_returns, axis=1)
    return float(np.mean(min_path_returns < -threshold))


def build_probability_surface(
    metric: str,
    mu: float,
    sigma_grid: list[float],
    lambda_grid: list[float],
    jump_mean: float,
    jump_std: float,
    horizon_steps: int,
    n_paths: int,
    threshold: float,
    base_seed: int = 42,
) -> np.ndarray:
    """
    Build Z matrix for either terminal downside crash probability
    or path-dependent downside barrier-hit probability.
    """
    if metric not in {"terminal", "barrier"}:
        raise ValueError("metric must be either 'terminal' or 'barrier'")

    z = np.zeros((len(sigma_grid), len(lambda_grid)), dtype=float)

    for i, sigma in enumerate(sigma_grid):
        for j, lam in enumerate(lambda_grid):
            kwargs = dict(
                mu=mu,
                sigma=sigma,
                lambda_jump=lam,
                jump_mean=jump_mean,
                jump_std=jump_std,
                horizon_steps=horizon_steps,
                n_paths=n_paths,
                threshold=threshold,
                seed=base_seed + i * 1000 + j,
            )
            if metric == "terminal":
                z[i, j] = probability_of_terminal_downside_crash(**kwargs)
            else:
                z[i, j] = probability_of_barrier_hit(**kwargs)

    return z
