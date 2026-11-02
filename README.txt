# TDA Demonstration Package

## Purpose

This package demonstrates the computational workflow for applying persistent homology and representative-cycle analysis to rolling-window financial return data.

The workflow consists of three stages:
1. Persistent homology computation and visualisation.
2. H0 Wasserstein distance and H1 time-series analysis.
3. HomCloud inverse analysis and representative-cycle analysis.

The included dataset is synthetic and is intended to demonstrate the computational procedures rather than reproduce the empirical results.

## Files

- `logret.tsv`: Synthetic daily log-return dataset.
- `tda_features.py`: Constructs rolling-window point clouds, computes H0 and H1 persistence, and generates persistence comparison figures.
- `tda_vectorise.py`: Computes H0 degree-2 Wasserstein distances and generates H0 and H1 time-series plots.
- `tda_inverse_analysis.py`: Extracts representative cycles, calculates stock participation frequencies, and computes Jaccard distances.

## Installation

Install the required Python packages:

    pip install numpy pandas scipy matplotlib gudhi

Install HomCloud according to its official installation instructions:

    https://homcloud.dev/

## Running the Scripts

Place all files in the same directory and execute:

    python tda_features.py
    python tda_vectorise.py
    python tda_inverse_analysis.py

The scripts read `logret.tsv` and save their outputs in the `tda_output` directory.

## Stage 1: Persistent Homology

Each stock is represented by a return vector over a rolling window of 60 trading days. The vectors are normalised by their Euclidean norms.

Vietoris–Rips filtrations are constructed to calculate H0 and H1 persistence.

The script produces:
- Daily H1 loop count and total persistence (`tda_daily_summary.csv`).
- Finite H0 and H1 persistence intervals (`tda_persistence_intervals.csv`).
- Headerless features export for visualization (`arag_tda_ftrs.csv`).
- A persistence comparison figure (`persistence_comparison.png`) showing the point cloud, H0/H1 barcodes, and H0/H1 persistence diagrams for the first rolling window.

## Stage 2: Vectorisation and Time-Series Analysis

The degree-2 Wasserstein distance is calculated between consecutive finite H0 persistence diagrams.

The calculation uses:
- Wasserstein order: 2.
- Ground metric: L-infinity.

The script produces:
- H0 Wasserstein distance time series (`h0_wasserstein_timeseries.png`).
- H1 loop-count and H1 total-persistence time series (`h1_timeseries.png`).

## Stage 3: Representative-Cycle Analysis

HomCloud inverse analysis is used to extract an optimal representative 1-cycle for every finite H1 persistence interval in each rolling window.

The script produces:
- `representative_cycles.csv`: Persistence information and stock membership for all extracted representative cycles.
- `stock_participation_frequency.csv`: Stock participation frequency across all extracted representative cycles in each window.
- `highest_lifetime_cycle_jaccard.csv`: Stock membership of the highest-lifetime representative cycle and its Jaccard distance from the selected cycle in the preceding window.
- `inverse_analysis_warnings.csv`: Records windows or cycles that could not be processed due to missing data or insufficient stocks.

Two different cycle-selection procedures are used:
1. Stock participation frequency considers all extracted representative cycles in each rolling window.
2. Jaccard distance considers only the representative cycle associated with the highest-lifetime H1 interval in each rolling window.

The first Jaccard value is undefined because there is no preceding window for comparison.

## Methodological Notes

1. The demonstration uses a 60-trading-day rolling window.
2. Each stock is represented by its normalised return vector.
3. The first complete rolling window contains the first 60 trading days.
4. H1 loop count is the number of finite H1 persistence intervals.
5. H1 total persistence is the sum of the lifetimes of finite H1 intervals.
6. The Wasserstein calculation uses the L-infinity ground metric and degree 2.
7. Representative cycles are algorithmically selected representatives of homology classes and are not necessarily unique.
8. Stock participation frequency and Jaccard distance describe different aspects of representative-cycle membership.
9. The synthetic dataset contains no designated financial events. Event detection, phase comparisons, peak offsets, and event-level boxplots are therefore not included.

The results are intended to illustrate the computational workflow and should not be interpreted as empirical findings about actual financial markets.
