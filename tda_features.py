"""
Stage 1: Rolling-window H0/H1 persistence computation with HomCloud.

Input: logret.tsv (first column dates, remaining columns stock returns).

Outputs in tda_output/:
  tda_daily_summary.csv
  tda_persistence_intervals.csv
  arag_tda_ftrs.csv
  persistence_comparison.png

Changes applied:
  - 1-column comparison layout showing the first window.
  - Reordered barcodes to show H0 on top and H1 on bottom.
  - Removed all essential/infinity features and arrow annotations.
  - Set barcode Y-axis label to 'Homology dimension'.
  - Updated title to display date only.
  - Exported PNG only (PDF export removed).
"""
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.lines import Line2D
from scipy.spatial.distance import pdist, squareform
import homcloud.interface as hc

INPUT_FILE = Path("logret.tsv")
OUT = Path("tda_output")
WINDOW_SIZE = 60
MAX_DIM = 2
NORMALIZE_EACH_STOCK = True
PLOT_FILTRATION_MAX = 1.5  # Plotting limit only
MAKE_FIGURE = True

OUT.mkdir(exist_ok=True)

if not INPUT_FILE.exists():
    raise FileNotFoundError(f"Place logret.tsv beside this script: {INPUT_FILE.resolve()}")

data = pd.read_csv(INPUT_FILE, sep="\t")
date_col = data.columns[0]
data[date_col] = pd.to_datetime(data[date_col], errors="coerce")
data = data.dropna(subset=[date_col]).sort_values(date_col)
data = data.drop_duplicates(subset=[date_col], keep="last").set_index(date_col)
data = data.apply(pd.to_numeric, errors="coerce")

if len(data) < WINDOW_SIZE:
    raise ValueError(f"Need at least {WINDOW_SIZE} rows; found {len(data)}.")

n_windows = len(data) - WINDOW_SIZE + 1
selected_indices = sorted(set([0]))  # Only first window needed for 1-column layout
daily_rows, interval_rows = [], []
selected = {}

def plot_barcode(ax, h0b, h0d, h1b, h1d):
    # Sort by lifetime descending
    h0_order = np.argsort(h0d - h0b)[::-1] if len(h0b) else []
    h1_order = np.argsort(h1d - h1b)[::-1] if len(h1b) else []

    # Bands: H0 on top (y roughly 1.2 to 2.4), H1 on bottom (y roughly 0.1 to 0.9)
    if len(h0b):
        y_h0 = np.linspace(1.25, 2.45, len(h0b))
        for y, idx in zip(y_h0, h0_order):
            ax.hlines(y, h0b[idx], h0d[idx], color="red", linewidth=0.25, alpha=0.35)

    if len(h1b):
        y_h1 = np.linspace(0.10, 0.90, len(h1b))
        for y, idx in zip(y_h1, h1_order):
            ax.hlines(y, h1b[idx], h1d[idx], color="royalblue", linewidth=0.25, alpha=0.35)

    ax.set_xlim(0, PLOT_FILTRATION_MAX)
    ax.set_ylim(-0.05, 2.60)
    ax.set_yticks([1.85, 0.50])
    ax.set_yticklabels([r"$H_0$", r"$H_1$"])
    ax.set_xlabel("Filtration value")
    ax.set_ylabel("Homology dimension")

    legend_handles = [
        Line2D([0], [0], color="red", linewidth=1.2, label=r"$H_0$"),
        Line2D([0], [0], color="royalblue", linewidth=1.2, label=r"$H_1$"),
    ]
    ax.legend(handles=legend_handles, loc="lower right", frameon=True, framealpha=0.9)

def plot_diagram(ax, h0b, h0d, h1b, h1d):
    ax.scatter(h1b, h1d, s=7, color="royalblue", alpha=0.55, edgecolors="none", label=r"$H_1$", zorder=2)
    ax.scatter(h0b, h0d, s=9, color="red", alpha=0.95, edgecolors="none", label=r"$H_0$", zorder=5)
    ax.axvline(0, color="red", linewidth=0.8, alpha=0.55, zorder=4)

    ax.plot([0, PLOT_FILTRATION_MAX], [0, PLOT_FILTRATION_MAX], "--", color="black", linewidth=0.7)
    ax.set_xlim(0, PLOT_FILTRATION_MAX)
    ax.set_ylim(0, PLOT_FILTRATION_MAX)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("Birth radius")
    ax.set_ylabel("Death radius")
    ax.legend(fontsize=8, loc="lower right")
    ax.grid(alpha=0.2, linewidth=0.4)

for wi in range(n_windows):
    window = data.iloc[wi:wi + WINDOW_SIZE]
    if window.isna().any().any():
        raise ValueError(
            f"Missing values in window ending {window.index[-1].date()}. "
            "Resolve missing values before running TDA."
        )

    cloud = window.to_numpy(dtype=float).T
    if NORMALIZE_EACH_STOCK:
        norms = np.linalg.norm(cloud, axis=1)
        keep = norms > 0
        cloud = cloud[keep]
        norms = norms[keep]
        cloud = cloud / norms[:, None]

    if cloud.shape[0] < 2:
        raise ValueError("Fewer than two valid stock vectors in a window.")

    dist = squareform(pdist(cloud, metric="euclidean"))
    pdlist = hc.PDList.from_rips_filtration(dist, maxdim=MAX_DIM)
    h0, h1 = pdlist.dth_diagram(0), pdlist.dth_diagram(1)

    h0b, h0d = np.asarray(h0.births, float), np.asarray(h0.deaths, float)
    h1b, h1d = np.asarray(h1.births, float), np.asarray(h1.deaths, float)
    date_text = pd.Timestamp(window.index[-1]).strftime("%Y-%m-%d")

    for dim, births, deaths in [(0, h0b, h0d), (1, h1b, h1d)]:
        for b, d in zip(births, deaths):
            interval_rows.append({
                "Date": date_text, "Dimension": dim,
                "Birth": float(b), "Death": float(d),
                "Lifetime": float(d - b)
            })

    h1life = h1d - h1b
    daily_rows.append({
        "Date": date_text,
        "Number_of_Stocks": int(cloud.shape[0]),
        "H0_Finite_Intervals": len(h0b),
        "H1_Loop_Count": len(h1b),
        "H1_Total_Persistence": float(np.sum(h1life))
    })

    if wi in selected_indices:
        selected[wi] = {
            "date": date_text, "cloud": cloud[:, :3],
            "h0b": h0b, "h0d": h0d,
            "h1b": h1b, "h1d": h1d
        }

    pdlist.close()
    print(f"{wi + 1}/{n_windows}: {date_text} | H0 finite={len(h0b)} | H1={len(h1b)}")

pd.DataFrame(daily_rows).to_csv(OUT / "tda_daily_summary.csv", index=False)
interval_df = pd.DataFrame(interval_rows)
interval_df.to_csv(OUT / "tda_persistence_intervals.csv", index=False)

if not interval_df.empty:
    compat = interval_df.copy()
    compat["Dimension"] = compat["Dimension"].map({0: "H0", 1: "H1"})
    compat[["Date", "Dimension", "Birth", "Death", "Lifetime"]].to_csv(
        OUT / "arag_tda_ftrs.csv", index=False, header=False
    )

if MAKE_FIGURE:
    fig = plt.figure(figsize=(6.0, 11.0))
    gs = GridSpec(3, 1, figure=fig, hspace=0.42)
    item = selected[0]

    # Row 1: Point Cloud (First three coordinates, date title only)
    ax1 = fig.add_subplot(gs[0, 0], projection="3d")
    xyz = item["cloud"]
    ax1.scatter(xyz[:, 0], xyz[:, 1], xyz[:, 2],
                s=7, alpha=0.65, color="royalblue", edgecolors="none")
    ax1.set_xlabel("Coordinate 1")
    ax1.set_ylabel("Coordinate 2")
    ax1.set_zlabel("Coordinate 3")
    ax1.set_title(item["date"])
    ax1.set_box_aspect((1, 1, 1))
    ax1.view_init(elev=24, azim=-58)

    # Row 2: Persistence Barcode
    ax2 = fig.add_subplot(gs[1, 0])
    plot_barcode(ax2, item["h0b"], item["h0d"], item["h1b"], item["h1d"])
    ax2.grid(alpha=0.2, linewidth=0.4)

    # Row 3: Persistence Diagram
    ax3 = fig.add_subplot(gs[2, 0])
    plot_diagram(ax3, item["h0b"], item["h0d"], item["h1b"], item["h1d"])

    fig.tight_layout()
    fig.savefig(OUT / "persistence_comparison.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

print(f"Stage 1 complete. Outputs: {OUT.resolve()}")
