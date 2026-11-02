"""
Stage 2: H0 degree-2 Wasserstein distances and H1 time-series plots.

Uses:
    order=2.0      degree-2 Wasserstein distance
    internal_p=inf L-infinity ground metric

Changes applied:
  - Formatted Y-axis labels with LaTeX italic subscripts: $H_0$ and $H_1$.
  - Updated all plot X-axis labels to 'date'.
  - Removed all plot main titles.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from gudhi.wasserstein import wasserstein_distance


OUT = Path("tda_output")
INTERVAL_FILE = OUT / "tda_persistence_intervals.csv"
SUMMARY_FILE = OUT / "tda_daily_summary.csv"

WASSERSTEIN_ORDER = 2.0
WASSERSTEIN_INTERNAL_P = float("inf")


def load_finite_h0_diagrams(interval_file):
    intervals = pd.read_csv(interval_file)
    intervals["Date"] = pd.to_datetime(intervals["Date"])

    h0 = intervals.loc[intervals["Dimension"] == 0]
    diagrams = {}

    for date, group in h0.groupby("Date"):
        diagrams[pd.Timestamp(date)] = (
            group[["Birth", "Death"]]
            .to_numpy(dtype=float)
            .reshape(-1, 2)
        )

    return diagrams


def main():
    if not INTERVAL_FILE.exists() or not SUMMARY_FILE.exists():
        raise FileNotFoundError(
            "Run tda_features.py first. "
            "Expected files in tda_output/."
        )

    summary = pd.read_csv(SUMMARY_FILE)
    summary["Date"] = pd.to_datetime(summary["Date"])

    diagrams = load_finite_h0_diagrams(INTERVAL_FILE)

    # H0 Wasserstein distances between consecutive windows
    distance_rows = []

    for previous_date, current_date in zip(
        summary["Date"].iloc[:-1],
        summary["Date"].iloc[1:]
    ):

        previous = diagrams.get(
            pd.Timestamp(previous_date),
            np.empty((0, 2), dtype=float)
        )

        current = diagrams.get(
            pd.Timestamp(current_date),
            np.empty((0, 2), dtype=float)
        )

        if len(previous) == 0 and len(current) == 0:
            distance = 0.0

        else:
            distance = wasserstein_distance(
                previous,
                current,
                order=WASSERSTEIN_ORDER,
                internal_p=WASSERSTEIN_INTERNAL_P
            )

        distance_rows.append({
            "Previous_Date": pd.Timestamp(previous_date).strftime("%Y-%m-%d"),
            "Current_Date": pd.Timestamp(current_date).strftime("%Y-%m-%d"),
            "H0_Wasserstein_Distance": float(distance)
        })

    distance_df = pd.DataFrame(distance_rows)

    distance_df.to_csv(
        OUT / "h0_wasserstein_distances.csv",
        index=False
    )

    # H0 Wasserstein time series
    if not distance_df.empty:
        distance_df["Current_Date"] = pd.to_datetime(
            distance_df["Current_Date"]
        )

        fig, ax = plt.subplots(figsize=(9, 3.5))

        ax.plot(
            distance_df["Current_Date"],
            distance_df["H0_Wasserstein_Distance"],
            linewidth=1.0
        )

        ax.set_xlabel("date")
        ax.set_ylabel(r"$H_0$ Wasserstein distance")
        ax.grid(alpha=0.25)

        fig.tight_layout()

        fig.savefig(
            OUT / "h0_wasserstein_timeseries.png",
            dpi=250
        )

        plt.close(fig)

    # H1 loop count and total persistence
    fig, axes = plt.subplots(
        2, 1,
        figsize=(9, 5.5),
        sharex=True
    )

    axes[0].plot(
        summary["Date"],
        summary["H1_Loop_Count"],
        linewidth=1.0
    )

    axes[0].set_ylabel(r"Number of $H_1$ loops")
    axes[0].grid(alpha=0.25)

    axes[1].plot(
        summary["Date"],
        summary["H1_Total_Persistence"],
        linewidth=1.0
    )

    axes[1].set_ylabel(r"$H_1$ total persistence")
    axes[1].set_xlabel("date")
    axes[1].grid(alpha=0.25)

    fig.tight_layout()

    fig.savefig(
        OUT / "h1_timeseries.png",
        dpi=250
    )

    plt.close(fig)

    print("Stage 2 complete.")
    print(f"Outputs saved in: {OUT.resolve()}")


if __name__ == "__main__":
    main()
