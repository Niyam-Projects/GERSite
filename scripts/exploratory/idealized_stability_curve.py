"""
Idealized percent-still-unchanged curve for fixed (delta, lambda) = (0.05, 0.5).

Mimics the axis and theme styling of scripts/osm_data/data_viz.py but plots
only the analytic trend implied by the ZIE turnover model, with no observed
data and no uncertainty band:

    P(still unchanged at t) = (1 - delta) * exp(-lambda * t)

with t in years.
"""

from pathlib import Path

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")  # noqa: E402
import plotnine as gg  # noqa: E402

# ----------------------------------------------------------------------------------------
# Fixed parameters
# ----------------------------------------------------------------------------------------

DELTA = 0.05
LAMBDA = 0.25  # per year
YEAR_RANGE = 10
N_POINTS = 1001

OUT_PATH = Path("~/data/openpois/exploratory/idealized_stability_curve.png").expanduser()

# ----------------------------------------------------------------------------------------
# Plot construction
# ----------------------------------------------------------------------------------------


def idealized_curve(
    delta: float,
    lam: float,
    year_range: float,
    n_points: int = 1001,
) -> pd.DataFrame:
    """Return a DataFrame with columns (year, y) for the analytic trend."""
    year = np.linspace(0, year_range, n_points)
    y = (1.0 - delta) * np.exp(-lam * year)
    return pd.DataFrame({'year': year, 'y': y})


def idealized_plot_create(
    df: pd.DataFrame,
    title: str | None = None,
    subtitle: str | None = None,
    x_label: str = '',
    y_label: str = '',
    year_range: float = 10,
) -> gg.ggplot:
    """Single-trend stability plot, styled to match change_plot_create."""
    fig = (
        gg.ggplot(
            data = df,
            mapping = gg.aes(x = 'year', y = 'y'),
        ) +
        gg.geom_line(color = 'darkred', size = 1) +
        gg.labs(
            title = title,
            subtitle = subtitle,
            x = x_label,
            y = y_label,
        ) +
        gg.scale_y_continuous(
            limits = (0, 1.01),
            breaks = np.arange(0, 1.01, 0.25),
            labels = [f"{x * 100:.0f}%" for x in np.arange(0, 1.01, 0.25)],
        ) +
        gg.scale_x_continuous(
            limits = (0, year_range + 0.01),
            breaks = np.arange(year_range + 1),
            labels = [f"{x:.0f}" for x in np.arange(year_range + 1)],
        ) +
        gg.theme_bw()
    )
    return fig


# ----------------------------------------------------------------------------------------
# Main workflow
# ----------------------------------------------------------------------------------------

if __name__ == "__main__":
    df = idealized_curve(
        delta = DELTA,
        lam = LAMBDA,
        year_range = YEAR_RANGE,
        n_points = N_POINTS,
    )
    fig = idealized_plot_create(
        df = df,
        title = "Idealized stability curve",
        subtitle = f"δ = {DELTA}, λ = {LAMBDA} / year",
        x_label = "Years since tag",
        y_label = "Proportion remaining unchanged",
        year_range = YEAR_RANGE,
    )
    OUT_PATH.parent.mkdir(parents = True, exist_ok = True)
    fig.save(
        filename = OUT_PATH,
        width = 10,
        height = 6,
        units = 'in',
        dpi = 300,
        verbose = False,
    )
    print(f"Saved: {OUT_PATH}")
