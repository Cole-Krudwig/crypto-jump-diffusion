from __future__ import annotations

# ----------------------------
# Market data settings
# ----------------------------
SYMBOL = "BTCUSDT"
INTERVAL = "1m"
LOOKBACK_DAYS = 14
BINANCE_BASE_URL = "https://api.binance.com"

# ----------------------------
# Feature engineering
# ----------------------------
ROLLING_VOL_WINDOW = 60
JUMP_THRESHOLD_MULT = 6.0       # number of standard deviations from typical return
JUMP_DIRECTION = "negative"

# ----------------------------
# Simulation settings
# ----------------------------
SIM_HORIZON_MINUTES = 60         # next hour
N_PATHS = 5000
DOWNSIDE_THRESHOLD = 0.02
RNG_SEED = 42

# ----------------------------
# Grids for 3D surface
# Sigma and lambda are per-minute quantities
# ----------------------------
SIGMA_GRID = [
    0.0005, 0.0010, 0.0015, 0.0020,
    0.0025, 0.0030, 0.0035, 0.0040
]

LAMBDA_GRID = [
    0.000, 0.005, 0.010, 0.020,
    0.030, 0.050, 0.075, 0.100
]

# ----------------------------
# Output
# ----------------------------
SAVE_CSV = True
CSV_FILENAME = "btcusdt_1m_features.csv"

RETURN_HIST_FIG = "returns_histogram.png"
ROLLING_VOL_FIG = "rolling_volatility.png"
TERMINAL_SURFACE_HTML = "terminal_downside_probability_surface.html"
BARRIER_SURFACE_HTML = "barrier_hit_probability_surface.html"
COMBINED_SURFACES_HTML = "combined_crash_probability_surfaces.html"
