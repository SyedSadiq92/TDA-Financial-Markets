"""
Stage 3: HomCloud inverse analysis of H1 persistence intervals.

For every rolling window:
1. Construct a normalized stock point cloud.
2. Compute the H1 persistence diagram with full graph tracking.
3. Extract an optimal representative 1-cycle for every finite H1 interval.
4. Record stock membership of each representative cycle.
5. Calculate stock participation frequencies across all extracted cycles.
6. Select the highest-lifetime cycle per date.
7. Calculate Jaccard distances between consecutive selected cycles.
"""

from pathlib import Path
import os
import numpy as np
import pandas as pd
from scipy.spatial.distance import pdist, squareform
import homcloud.interface as hc
import matplotlib.pyplot as plt

# ================================================================
# 1. CONFIGURATION
# ================================================================

INPUT_FILE = Path("logret.tsv")
OUTPUT_DIR = Path("tda_output")

WINDOW_SIZE = 60
MAX_DIM = 1
NORMALIZE = True

OUTPUT_DIR.mkdir(exist_ok=True)


# ================================================================
# 2. DATA PREPARATION
# ================================================================

def load_data():
    data = pd.read_csv(INPUT_FILE, sep="\t")
    date_column = data.columns[0]

    data[date_column] = pd.to_datetime(data[date_column], errors="coerce")

    data = (
        data.dropna(subset=[date_column])
        .sort_values(date_column)
        .drop_duplicates(subset=[date_column])
        .set_index(date_column)
    )

    data = data.apply(pd.to_numeric, errors="coerce")
    return data


def build_point_cloud(window):
    point_cloud = window.to_numpy(dtype=float).T
    stock_names = list(window.columns)

    if NORMALIZE:
        norms = np.linalg.norm(point_cloud, axis=1)
        valid = norms > 0
        point_cloud = point_cloud[valid]
        point_cloud = point_cloud / norms[valid, None]

        stock_names = [
            stock for stock, keep in zip(stock_names, valid)
            if keep
        ]

    return point_cloud, stock_names


# ================================================================
# 3. JACCARD DISTANCE
# ================================================================

def jaccard_distance(set_a, set_b):
    union = set_a | set_b
    if len(union) == 0:
        return 0.0
    intersection = set_a & set_b
    return 1.0 - len(intersection) / len(union)


# ================================================================
# 4. REPRESENTATIVE-CYCLE EXTRACTION
# ================================================================

def extract_cycles(point_cloud, stock_names, date):
    distance_matrix = squareform(pdist(point_cloud, metric="euclidean"))
    temporary_file = OUTPUT_DIR / f"temp_{date}_rips.pdgm"

    if temporary_file.exists():
        try:
            temporary_file.unlink()
        except Exception:
            pass

    cycles = []

    try:
        # Correct HomCloud argument: save_to
        pd_list = hc.PDList.from_rips_filtration(
            distance_matrix,
            maxdim=MAX_DIM,
            save_to=str(temporary_file),
            save_graph=True
        )

        diagram = pd_list.dth_diagram(1)

        # Retain finite H1 persistence intervals sorted by lifetime descending
        pairs = sorted(
            [p for p in diagram.pairs() if np.isfinite(p.birth_time()) and np.isfinite(p.death_time())],
            key=lambda p: p.lifetime(),
            reverse=True
        )

        for rank, pair in enumerate(pairs, start=1):
            try:
                cycle = pair.optimal_1_cycle()
                indices = [int(idx) for idx in cycle.boundary_points()]
                indices = list(dict.fromkeys(indices))  # Unique indices maintaining order

                members = [
                    stock_names[idx]
                    for idx in indices
                    if 0 <= idx < len(stock_names)
                ]

                if members:
                    cycles.append({
                        "Date": date,
                        "Rank": rank,
                        "Birth": float(pair.birth_time()),
                        "Death": float(pair.death_time()),
                        "Lifetime": float(pair.lifetime()),
                        "Point_Count": len(members),
                        "Stock_IDs": ";".join(members)
                    })

            except Exception as error:
                print(f"Warning ({date}, rank {rank}): Failed cycle extraction - {error}")

        pd_list.close()

    except Exception as error:
        print(f"Error processing window {date}: {error}")

    finally:
        if temporary_file.exists():
            try:
                temporary_file.unlink()
            except Exception:
                pass

    return cycles

# ================================================================
# 5. STOCK PARTICIPATION HEATMAP
# ================================================================

def create_participation_heatmap(participation_df, dates, stock_names):

    if participation_df.empty:
        print("No representative-cycle participation data available.")
        return

    date_index = {
        date: i for i, date in enumerate(dates)
    }

    stock_index = {
        stock: i for i, stock in enumerate(stock_names)
    }

    # Rows = stocks; columns = rolling-window dates
    grid = np.zeros(
        (len(stock_names), len(dates)),
        dtype=np.int32
    )

    for _, row in participation_df.iterrows():

        stock = row["Stock_ID"]
        date = row["Date"]

        if stock in stock_index and date in date_index:

            grid[
                stock_index[stock],
                date_index[date]
            ] = int(row["Cycle_Count"])

    positive_values = grid[grid > 0]

    if positive_values.size == 0:
        print("No positive participation counts to plot.")
        return

    vmax = max(
        float(np.percentile(positive_values, 98)),
        1.0
    )

    fig, ax = plt.subplots(
        figsize=(12, 6)
    )

    image = ax.imshow(
        grid,
        aspect="auto",
        cmap="magma",
        origin="upper",
        vmin=0,
        vmax=vmax,
        interpolation="nearest"
    )

    ax.set_xlabel("Date")
    ax.set_ylabel("Stock")

    # Display a small number of stock labels.
    tick_positions = np.linspace(
        0,
        len(stock_names) - 1,
        min(5, len(stock_names)),
        dtype=int
    )

    ax.set_yticks(tick_positions)
    ax.set_yticklabels(
        [stock_names[i] for i in tick_positions]
    )

    # Display approximately six date labels.
    date_positions = np.linspace(
        0,
        len(dates) - 1,
        min(6, len(dates)),
        dtype=int
    )

    ax.set_xticks(date_positions)
    ax.set_xticklabels(
        [pd.Timestamp(dates[i]).strftime("%Y-%m-%d")
         for i in date_positions],
        rotation=30,
        ha="right"
    )

    colorbar = fig.colorbar(image, ax=ax)
    colorbar.set_label(
        "Daily loop participation frequency"
    )

    fig.tight_layout()

    fig.savefig(
        OUTPUT_DIR / "stock_participation_heatmap.png",
        dpi=300,
        bbox_inches="tight"
    )

    plt.close(fig)

    print("Saved stock_participation_heatmap.png")
    
# ================================================================
# 6. MAIN ROLLING-WINDOW ANALYSIS
# ================================================================

def main():

    data = load_data()

    if len(data) < WINDOW_SIZE:
        raise ValueError(
            f"At least {WINDOW_SIZE} trading days are required."
        )

    all_cycles = []

    total_windows = len(data) - WINDOW_SIZE + 1

    window_dates = []

    for window_number, end_index in enumerate(
        range(WINDOW_SIZE - 1, len(data)),
        start=1
    ):

        window = data.iloc[
            end_index - WINDOW_SIZE + 1:end_index + 1
        ]

        date = window.index[-1].strftime("%Y-%m-%d")

        window_dates.append(date)

        if window.isna().any().any():

            print(
                f"Skipping window {window_number}: "
                f"missing values ({date})"
            )

            continue

        point_cloud, stock_names = build_point_cloud(window)

        if len(stock_names) < 3:

            print(
                f"Skipping window {window_number}: "
                f"insufficient non-zero vectors ({date})"
            )

            continue

        extracted = extract_cycles(
            point_cloud,
            stock_names,
            date
        )

        all_cycles.extend(extracted)

        print(
            f"Window {window_number}/{total_windows}: "
            f"{date} | Extracted cycles: {len(extracted)}"
        )

    # ============================================================
    # 7. SAVE REPRESENTATIVE CYCLES
    # ============================================================

    cycle_columns = [
        "Date",
        "Rank",
        "Birth",
        "Death",
        "Lifetime",
        "Point_Count",
        "Stock_IDs"
    ]

    cycle_df = pd.DataFrame(
        all_cycles,
        columns=cycle_columns
    )

    cycle_df.to_csv(
        OUTPUT_DIR / "representative_cycles.csv",
        index=False
    )

    # ============================================================
    # 8. STOCK PARTICIPATION COUNTS
    # ============================================================

    participation_rows = []

    # Include every rolling-window date, including dates
    # with no extracted finite H1 representative cycles.
    for date in window_dates:

        group = cycle_df.loc[
            cycle_df["Date"] == date
        ]

        stock_counts = {}

        total_cycles = len(group)

        for stock_string in group["Stock_IDs"]:

            # Count each stock once per representative cycle.
            members = set(
                filter(None, stock_string.split(";"))
            )

            for stock in members:

                stock_counts[stock] = (
                    stock_counts.get(stock, 0) + 1
                )

        for stock in data.columns:

            count = stock_counts.get(stock, 0)

            participation_rows.append({

                "Date": date,
                "Stock_ID": stock,
                "Cycle_Count": count,
                "Total_Cycles": total_cycles

            })

    participation_df = pd.DataFrame(
        participation_rows
    )

    participation_df.to_csv(
        OUTPUT_DIR / "stock_participation_frequency.csv",
        index=False
    )

    # ============================================================
    # 9. CREATE STOCK PARTICIPATION HEATMAP
    # ============================================================

    create_participation_heatmap(
        participation_df,
        window_dates,
        list(data.columns)
    )

    # ============================================================
    # 10. HIGHEST-LIFETIME CYCLE AND JACCARD DISTANCE
    # ============================================================

    jaccard_rows = []

    if not cycle_df.empty:

        selected = (
            cycle_df
            .sort_values(
                ["Date", "Lifetime"],
                ascending=[True, False]
            )
            .drop_duplicates(
                subset="Date",
                keep="first"
            )
            .sort_values("Date")
        )

        previous_members = None
        previous_date = None

        for _, row in selected.iterrows():

            current_members = set(
                filter(None, row["Stock_IDs"].split(";"))
            )

            if previous_members is None:

                distance = np.nan

            else:

                distance = jaccard_distance(
                    previous_members,
                    current_members
                )

            jaccard_rows.append({

                "Previous_Date": previous_date,
                "Date": row["Date"],
                "Lifetime": row["Lifetime"],
                "Stock_Count": len(current_members),
                "Jaccard_Distance": distance,
                "Stock_IDs": row["Stock_IDs"]

            })

            previous_members = current_members
            previous_date = row["Date"]

    jaccard_df = pd.DataFrame(
        jaccard_rows
    )

    jaccard_df.to_csv(
        OUTPUT_DIR / "highest_lifetime_cycle_jaccard.csv",
        index=False
    )

    # ============================================================
    # 11. COMPLETION
    # ============================================================

    print("\nInverse analysis completed.")

    print(
        f"Total representative cycles: {len(cycle_df)}"
    )

    print(
        f"Rolling windows processed: {total_windows}"
    )

    print(
        f"Output directory: {OUTPUT_DIR.resolve()}"
    )


if __name__ == "__main__":
    main()
