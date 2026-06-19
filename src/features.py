from __future__ import annotations

import numpy as np
import pandas as pd

from config import ROLLING_VOL_WINDOW, JUMP_THRESHOLD_MULT, JUMP_DIRECTION


def compute_log_returns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["log_price"] = np.log(out["close"])
    out["log_return"] = out["log_price"].diff()
    return out


def compute_rolling_volatility(
    df: pd.DataFrame,
    window: int = ROLLING_VOL_WINDOW,
) -> pd.DataFrame:
    out = df.copy()
    out["rolling_vol"] = out["log_return"].rolling(window=window).std()
    out["abs_return"] = out["log_return"].abs()
    return out


def detect_jumps(
    df: pd.DataFrame,
    threshold_mult: float = JUMP_THRESHOLD_MULT,
    direction: str = JUMP_DIRECTION,
) -> pd.DataFrame:
    out = df.copy()
    threshold = threshold_mult * out["rolling_vol"]

    if direction == "negative":
        out["jump_flag"] = (out["log_return"] < -threshold).astype(int)
    elif direction == "positive":
        out["jump_flag"] = (out["log_return"] > threshold).astype(int)
    else:
        out["jump_flag"] = (out["abs_return"] > threshold).astype(int)

    return out


def estimate_jump_parameters(df: pd.DataFrame) -> dict:
    """
    Estimate:
    - drift per minute
    - baseline sigma per minute
    - jump intensity per minute
    - jump mean/std using returns flagged as jumps
    """
    valid_returns = df["log_return"].dropna()
    valid_vol = df["rolling_vol"].dropna()

    if valid_returns.empty:
        raise ValueError(
            "No valid returns available for parameter estimation.")

    mu_hat = float(valid_returns.mean())
    sigma_hat = float(valid_returns.std())

    jump_returns = df.loc[df["jump_flag"] == 1, "log_return"].dropna()

    # Per-minute jump probability / intensity proxy
    n_obs = int(df["log_return"].notna().sum())
    n_jumps = int(df["jump_flag"].sum())
    lambda_hat = float(n_jumps / n_obs) if n_obs > 0 else 0.0

    if len(jump_returns) >= 2:
        jump_mean = float(jump_returns.mean())
        jump_std = float(jump_returns.std())
    elif len(jump_returns) == 1:
        jump_mean = float(jump_returns.iloc[0])
        jump_std = float(abs(jump_returns.iloc[0]) * 0.5)
    else:
        jump_mean = 0.0
        jump_std = max(sigma_hat * 3.0, 1e-6)

    return {
        "mu_hat": mu_hat,
        "sigma_hat": sigma_hat,
        "rolling_vol_mean": float(valid_vol.mean()) if not valid_vol.empty else np.nan,
        "rolling_vol_median": float(valid_vol.median()) if not valid_vol.empty else np.nan,
        "lambda_hat": lambda_hat,
        "jump_mean": jump_mean,
        "jump_std": max(jump_std, 1e-8),
        "n_obs": n_obs,
        "n_jumps": n_jumps,
        "jump_direction": JUMP_DIRECTION,
    }
