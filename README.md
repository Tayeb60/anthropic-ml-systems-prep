# Anthropic ML Systems & Performance: Data Pipeline Benchmarks

## Overview
This repository contains benchmarks for high-performance data processing pipelines in Python, created as part of my preparation for the Anthropic Fellows Program (ML Systems & Performance).

## The Problem
Processing large datasets (1 million+ rows) is a core task in ML systems. The choice of library and the way the code is written can have a significant impact on performance. This project compares the performance of two popular Python data libraries: **Pandas** and **Polars**.

## The Benchmark
- **Dataset:** 1,000,000 rows of synthetic sales data (Transaction_ID, Category, Price, Quantity).
- **Task:** Load the CSV, filter for "Electronics", and calculate the total revenue (Price * Quantity).
- **Hardware:** Apple MacBook Air (M4)

## Results

| Library | Processing Time | Notes |
| :--- | :--- | :--- |
| Pandas | 0.37s | Baseline. Uses C engine under the hood. |
| Polars  | 0.29s | **22% faster**. Written in Rust, optimized for parallelism. |

**Key Finding:** Polars outperforms Pandas by approximately 22% on this task. For a 100-million-row dataset, this translates to a time saving of roughly 8 seconds per run. At scale, this saves significant compute costs.

## How to Reproduce
1. Clone this repo.
2. Install dependencies: `pip install pandas polars`
3. Generate the data: `python generate_data.py`
4. Run the Pandas version: `python process_data.py`
5. Run the Polars version: `python process_data_v2.py`

## Future Work
- Implement multiprocessing to use all CPU cores.
- Extend benchmarks to GPU-accelerated libraries (CuPy, PyTorch).
- Build a distributed training pipeline using PyTorch DDP.