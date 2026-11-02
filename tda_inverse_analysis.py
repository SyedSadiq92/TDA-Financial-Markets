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
# 5. MAIN ROLLING-WINDOW ANALYSIS
# ================================================================

def main():
    data = load_data()

    if len(data) < WINDOW_SIZE:
        raise ValueError(f"At least {WINDOW_SIZE} trading days are required.")

    all_cycles = []
    warnings = []
    total_windows = len(data) - WINDOW_SIZE + 1

    for window_number, end_index in enumerate(
        range(WINDOW_SIZE - 1, len(data)),
        start=1
    ):
        window = data.iloc[end_index - WINDOW_SIZE + 1:end_index + 1]
        date = window.index[-1].strftime("%Y-%m-%d")

        if window.isna().any().any():
            warnings.append({"Date": date, "Reason": "Missing values in window"})
            continue

        point_cloud, stock_names = build_point_cloud(window)

        if len(stock_names) < 3:
            warnings.append({"Date": date, "Reason": "Insufficient non-zero vectors"})
            continue

        extracted = extract_cycles(point_cloud, stock_names, date)
        all_cycles.extend(extracted)

        print(
            f"Window {window_number}/{total_windows}: {date} | "
            f"Extracted cycles: {len(extracted)}"
        )

    # ============================================================
    # 6. SAVE REPRESENTATIVE CYCLES
    # ============================================================

    cycle_columns = ["Date", "Rank", "Birth", "Death", "Lifetime", "Point_Count", "Stock_IDs"]
    cycle_df = pd.DataFrame(all_cycles, columns=cycle_columns)
    cycle_df.to_csv(OUTPUT_DIR / "representative_cycles.csv", index=False)

    # ============================================================
    # 7. STOCK PARTICIPATION FREQUENCY
    # ============================================================

    participation_rows = []
    if not cycle_df.empty:
        for date, group in cycle_df.groupby("Date"):
            stock_counts = {}
            for stock_string in group["Stock_IDs"]:
                members = set(filter(None, stock_string.split(";")))
                for stock in members:
                    stock_counts[stock] = stock_counts.get(stock, 0) + 1

            total_cycles = len(group)
            for stock in data.columns:
                count = stock_counts.get(stock, 0)
                participation_rows.append({
                    "Date": date,
                    "Stock_ID": stock,
                    "Cycle_Count": count,
                    "Participation_Frequency": (count / total_cycles) if total_cycles > 0 else 0,
                    "Total_Cycles": total_cycles
                })

    participation_df = pd.DataFrame(participation_rows)
    participation_df.to_csv(OUTPUT_DIR / "stock_participation_frequency.csv", index=False)

    # ============================================================
    # 8. HIGHEST-LIFETIME CYCLE AND JACCARD DISTANCE
    # ============================================================

    jaccard_rows = []
    if not cycle_df.empty:
        selected = (
            cycle_df
            .sort_values(["Date", "Lifetime"], ascending=[True, False])
            .drop_duplicates(subset="Date", keep="first")
            .sort_values("Date")
        )

        previous_members = None
        previous_date = None

        for _, row in selected.iterrows():
            current_members = set(filter(None, row["Stock_IDs"].split(";")))

            if previous_members is None:
                distance = np.nan
            else:
                distance = jaccard_distance(previous_members, current_members)

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

    jaccard_df = pd.DataFrame(jaccard_rows)
    jaccard_df.to_csv(OUTPUT_DIR / "highest_lifetime_cycle_jaccard.csv", index=False)

    # ============================================================
    # 9. SAVE WARNINGS
    # ============================================================

    pd.DataFrame(warnings).to_csv(OUTPUT_DIR / "inverse_analysis_warnings.csv", index=False)

    print("\nInverse analysis completed.")
    print(f"Total representative cycles saved: {len(cycle_df)}")
    print(f"Output directory: {OUTPUT_DIR.resolve()}")


if __name__ == "__main__":
    main()
