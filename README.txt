# TDA Demonstration Package

## 1. Purpose

This package demonstrates a computational workflow for applying persistent homology and representative-cycle analysis to rolling-window financial return data.

The workflow consists of three stages:

1. Persistent homology computation and visualisation.
2. Wasserstein distance calculation and persistence-based time-series analysis.
3. HomCloud inverse analysis and representative-cycle analysis.

The included dataset is synthetic and is intended to demonstrate the computational procedures rather than reproduce the empirical results of the accompanying research.

## 2. Package Contents

| File | Description |
|---|---|
| `logret.tsv` | Synthetic daily log-return dataset. |
| `tda_features.py` | Constructs rolling-window point clouds, computes H0 and H1 persistence, and generates persistence visualisations. |
| `tda_vectorise.py` | Calculates H0 Wasserstein distances and generates H0 and H1 time-series plots. |
| `tda_inverse_analysis.py` | Extracts representative cycles, calculates stock participation counts, generates a participation heatmap, and computes Jaccard distances. |

## 3. Installation

Install the required Python packages:

```bash
pip install numpy pandas scipy matplotlib gudhi pot
```

Install HomCloud according to its official installation instructions:

https://homcloud.dev/

## 4. Running the Scripts

Place all package files in the same directory and execute the scripts in the following order:

```bash
python tda_features.py
python tda_vectorise.py
python tda_inverse_analysis.py
```

All scripts read the input dataset `logret.tsv` and save their outputs in the `tda_output` directory.

## 5. Stage 1: Persistent Homology

Each stock is represented by a return vector over a rolling window of 60 trading days. The vectors are normalised by their Euclidean norms.

Vietoris–Rips filtrations are constructed to calculate H0 and H1 persistence.

The script produces:

- `tda_daily_summary.csv`: Daily H1 loop count and total persistence.
- `tda_persistence_intervals.csv`: Finite H0 and H1 persistence intervals.
- `arag_tda_ftrs.csv`: Headerless feature export for visualisation.
- `persistence_comparison.png`: Persistence comparison for the first rolling window, showing the point cloud, H0 and H1 barcodes, and H0 and H1 persistence diagrams.

## 6. Stage 2: Vectorisation and Time-Series Analysis

The degree-2 Wasserstein distance is calculated between consecutive H0 persistence diagrams.

The calculation uses:

- Wasserstein order: 2.
- Ground metric: L-infinity.

The script produces:

- `h0_wasserstein_timeseries.png`: Time series of H0 Wasserstein distances.
- `h1_timeseries.png`: Time series of H1 loop counts and H1 total persistence.

## 7. Stage 3: Representative-Cycle Analysis

HomCloud inverse analysis is used to extract an optimal representative 1-cycle for each finite H1 persistence interval in every rolling window.

The script produces:

- `representative_cycles.csv`: Persistence information and stock membership for all extracted representative cycles.
- `stock_participation_frequency.csv`: Number of extracted representative cycles containing each stock in each rolling window.
- `stock_participation_heatmap.png`: Heatmap showing stock participation counts across the rolling windows.
- `highest_lifetime_cycle_jaccard.csv`: Stock membership of the highest-lifetime representative cycle in each window and its Jaccard distance from the selected cycle in the preceding window.

Two different cycle-selection procedures are used:

1. **Stock participation analysis:** Considers all extracted representative cycles in each rolling window. The participation count records how many cycles contain each stock; it is not normalised by the total number of cycles.
2. **Jaccard distance analysis:** Considers only the representative cycle associated with the highest-lifetime H1 persistence interval in each rolling window. The Jaccard distance measures changes in stock membership between consecutive selected cycles.

The first Jaccard value is undefined because there is no preceding rolling window for comparison.

## 8. Methodological Notes

1. The demonstration uses a 60-trading-day rolling window.
2. Each stock is represented by its normalised return vector.
3. The first complete rolling window contains the first 60 trading days.
4. H1 loop count is the number of finite H1 persistence intervals.
5. H1 total persistence is the sum of the lifetimes of finite H1 persistence intervals.
6. The Wasserstein calculation uses the L-infinity ground metric and degree 2.
7. Representative cycles are algorithmically selected representatives of homology classes and are not necessarily unique.
8. Stock participation counts and Jaccard distances describe different aspects of representative-cycle membership.
9. The synthetic dataset contains no designated financial events. Event detection, phase comparisons, peak offsets, and event-level boxplots are therefore not included.

## 9. Scope and Limitations

The synthetic dataset is designed to illustrate the computational workflow and the generation of representative-cycle outputs. It does not reproduce the historical financial events or empirical findings analysed in the research.

The results should therefore be interpreted as demonstrations of the computational procedures rather than empirical findings about actual financial markets.
