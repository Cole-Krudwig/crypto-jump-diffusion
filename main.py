from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from config import (
    SYMBOL,
    INTERVAL,
    LOOKBACK_DAYS,
    SAVE_CSV,
    CSV_FILENAME,
    RETURN_HIST_FIG,
    ROLLING_VOL_FIG,
    TERMINAL_SURFACE_HTML,
    BARRIER_SURFACE_HTML,
    COMBINED_SURFACES_HTML,
    ROLLING_VOL_WINDOW,
    JUMP_THRESHOLD_MULT,
    JUMP_DIRECTION,
    SIM_HORIZON_MINUTES,
    N_PATHS,
    DOWNSIDE_THRESHOLD,
    RNG_SEED,
    SIGMA_GRID,
    LAMBDA_GRID,
)
from src.data_fetch import load_historical_data
from src.features import (
    compute_log_returns,
    compute_rolling_volatility,
    detect_jumps,
    estimate_jump_parameters,
)
from src.simulation import build_probability_surface


def make_return_histogram(df):
    plt.figure(figsize=(10, 6))
    returns = df["log_return"].dropna()
    plt.hist(returns, bins=100, density=True)
    plt.title(f"{SYMBOL} 1-Minute Log Return Histogram")
    plt.xlabel("Log Return")
    plt.ylabel("Density")
    plt.tight_layout()
    plt.savefig(f"outputs/{RETURN_HIST_FIG}", dpi=200)
    plt.close()


def make_rolling_vol_plot(df):
    plt.figure(figsize=(12, 6))
    plt.plot(df["timestamp"], df["rolling_vol"])
    plt.title(f"{SYMBOL} Rolling Volatility ({ROLLING_VOL_WINDOW}-Minute Std)")
    plt.xlabel("Timestamp (UTC)")
    plt.ylabel("Rolling Volatility")
    plt.tight_layout()
    plt.savefig(f"outputs/{ROLLING_VOL_FIG}", dpi=200)
    plt.close()


def make_surface_plot(sigma_grid, lambda_grid, z, html_path, title, z_label):
    x = np.array(lambda_grid)   # jump intensity
    y = np.array(sigma_grid)    # volatility
    X, Y = np.meshgrid(x, y)

    fig = go.Figure(
        data=[
            go.Surface(
                x=X,
                y=Y,
                z=z,
                contours={
                    "z": {
                        "show": True,
                        "usecolormap": True,
                        "project": {"z": True},
                    }
                },
            )
        ]
    )

    fig.update_layout(
        title=title,
        scene=dict(
            xaxis_title="Jump Intensity (lambda, per minute)",
            yaxis_title="Volatility (sigma, per minute)",
            zaxis_title=z_label,
            camera=dict(eye=dict(x=1.6, y=1.5, z=0.9)),
        ),
        width=1000,
        height=780,
    )

    # fig.write_html(html_path)


def make_combined_surface_plot(sigma_grid, lambda_grid, terminal_z, barrier_z, html_path):
    x = np.array(lambda_grid)
    y = np.array(sigma_grid)
    X, Y = np.meshgrid(x, y)

    zmin = float(min(np.min(terminal_z), np.min(barrier_z)))
    zmax = float(max(np.max(terminal_z), np.max(barrier_z)))

    fig = make_subplots(
        rows=1,
        cols=2,
        specs=[[{"type": "surface"}, {"type": "surface"}]],
        subplot_titles=(
            f"Terminal Downside Probability: P(Return at {SIM_HORIZON_MINUTES}m < -{DOWNSIDE_THRESHOLD:.0%})",
            f"Path-Dependent Barrier Hit: P(Min Return in {SIM_HORIZON_MINUTES}m < -{DOWNSIDE_THRESHOLD:.0%})",
        ),
        horizontal_spacing=0.03,
    )

    common_surface = dict(
        x=X,
        y=Y,
        cmin=zmin,
        cmax=zmax,
        colorscale="Viridis",
        contours={
            "z": {
                "show": True,
                "usecolormap": True,
                "project": {"z": True},
            }
        },
        colorbar=dict(title="Probability", len=0.8),
    )

    fig.add_trace(
        go.Surface(z=terminal_z, showscale=True, **common_surface),
        row=1, col=1,
    )
    fig.add_trace(
        go.Surface(z=barrier_z, showscale=False, **common_surface),
        row=1, col=2,
    )

    scene_common = dict(
        xaxis_title="Jump Intensity (lambda, per minute)",
        yaxis_title="Volatility (sigma, per minute)",
        zaxis_title="Probability",
        zaxis=dict(range=[zmin, zmax]),
        camera=dict(eye=dict(x=1.55, y=1.45, z=0.9)),
    )

    fig.update_layout(
        title=(
            f"{SYMBOL} Downside Crash Probability Surfaces "
            f"({SIM_HORIZON_MINUTES}-Minute Horizon, Shared Z-Scale)"
        ),
        title_x=0.5,
        scene=scene_common,
        scene2=scene_common,
        width=1600,
        height=780,
        margin=dict(l=0, r=0, t=80, b=0),
    )

    fig.write_html(f"outputs/{html_path}")


def main():
    print("Loading Binance data...")
    df = load_historical_data(
        symbol=SYMBOL,
        timeframe=INTERVAL,
        lookback_days=LOOKBACK_DAYS,
    )

    print("Computing features...")
    df = compute_log_returns(df)
    df = compute_rolling_volatility(df, window=ROLLING_VOL_WINDOW)
    df = detect_jumps(
        df,
        threshold_mult=JUMP_THRESHOLD_MULT,
        direction=JUMP_DIRECTION,
    )

    # Drop early rows where rolling volatility is NaN
    df = df.dropna(subset=["log_return", "rolling_vol"]).reset_index(drop=True)

    if SAVE_CSV:
        df.to_csv(f"data/{CSV_FILENAME}", index=False)
        print(f"Saved feature dataset to {CSV_FILENAME}")

    print("Estimating parameters...")
    params = estimate_jump_parameters(df)

    print("\n===== Estimated Parameters =====")
    print(f"Symbol: {SYMBOL}")
    print(f"Interval: {INTERVAL}")
    print(f"Lookback days: {LOOKBACK_DAYS}")
    print(f"Jump direction: {params['jump_direction']}")
    print(f"Observations used: {params['n_obs']}")
    print(f"Detected jumps: {params['n_jumps']}")
    print(f"Drift (mu_hat, per minute): {params['mu_hat']:.8f}")
    print(f"Baseline sigma_hat (per minute): {params['sigma_hat']:.8f}")
    print(f"Mean rolling vol (per minute): {params['rolling_vol_mean']:.8f}")
    print(
        f"Median rolling vol (per minute): {params['rolling_vol_median']:.8f}")
    print(
        f"Jump intensity lambda_hat (per minute): {params['lambda_hat']:.8f}")
    print(f"Jump mean: {params['jump_mean']:.8f}")
    print(f"Jump std: {params['jump_std']:.8f}")

    print("\nBuilding terminal downside surface...")
    terminal_z = build_probability_surface(
        metric="terminal",
        mu=params["mu_hat"],
        sigma_grid=SIGMA_GRID,
        lambda_grid=LAMBDA_GRID,
        jump_mean=params["jump_mean"],
        jump_std=params["jump_std"],
        horizon_steps=SIM_HORIZON_MINUTES,
        n_paths=N_PATHS,
        threshold=DOWNSIDE_THRESHOLD,
        base_seed=RNG_SEED,
    )

    print("Building barrier-hit surface...")
    barrier_z = build_probability_surface(
        metric="barrier",
        mu=params["mu_hat"],
        sigma_grid=SIGMA_GRID,
        lambda_grid=LAMBDA_GRID,
        jump_mean=params["jump_mean"],
        jump_std=params["jump_std"],
        horizon_steps=SIM_HORIZON_MINUTES,
        n_paths=N_PATHS,
        threshold=DOWNSIDE_THRESHOLD,
        base_seed=RNG_SEED + 100_000,
    )

    print("Generating plots...")
    make_return_histogram(df)
    make_rolling_vol_plot(df)
    make_surface_plot(
        SIGMA_GRID,
        LAMBDA_GRID,
        terminal_z,
        TERMINAL_SURFACE_HTML,
        title=(
            f"{SYMBOL} Terminal Downside Crash Probability "
            f"over Next {SIM_HORIZON_MINUTES} Minutes"
        ),
        z_label=f"P(Terminal Return < -{DOWNSIDE_THRESHOLD:.0%})",
    )
    make_surface_plot(
        SIGMA_GRID,
        LAMBDA_GRID,
        barrier_z,
        BARRIER_SURFACE_HTML,
        title=(
            f"{SYMBOL} Path-Dependent Barrier-Hit Probability "
            f"over Next {SIM_HORIZON_MINUTES} Minutes"
        ),
        z_label=f"P(Min Return < -{DOWNSIDE_THRESHOLD:.0%})",
    )

    make_combined_surface_plot(
        SIGMA_GRID,
        LAMBDA_GRID,
        terminal_z,
        barrier_z,
        COMBINED_SURFACES_HTML,
    )

    print("\nDone.")
    print(f"Saved histogram to: outputs/{RETURN_HIST_FIG}")
    print(f"Saved rolling vol plot to: outputs/{ROLLING_VOL_FIG}")
    print(f"Saved terminal surface to: outputs/{TERMINAL_SURFACE_HTML}")
    print(f"Saved barrier surface to: outputs/{BARRIER_SURFACE_HTML}")


if __name__ == "__main__":
    main()
