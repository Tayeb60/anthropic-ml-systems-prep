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

### Benchmark 1: Pandas vs Polars (1M Rows)

| Library | Processing Time | Notes |
| :--- | :--- | :--- |
| Pandas | 0.37s | Baseline. Uses C engine under the hood. |
| Polars  | 0.29s | **22% faster**. Written in Rust, optimized for parallelism. |

**Key Finding:** Polars outperforms Pandas by approximately 22% on this task. For a 100-million-row dataset, this translates to a time saving of roughly 8 seconds per run. At scale, this saves significant compute costs.

### Benchmark 2: File Formats (10M Rows)

| Format | Load Type | Time | Speedup |
| :--- | :--- | :--- | :--- |
| CSV | Full Load | 2.82s | Baseline |
| Parquet | Full Load | 0.30s | **9.4x faster** |
| Parquet | Partial Load (2 cols) | 0.14s | **20.1x faster** |

**Key Finding:** Parquet is drastically faster than CSV at scale. The columnar format allows you to read only the columns you need, which is a massive advantage for ML training pipelines.

### Profiling Results

Using `cProfile`, I found that **93% of the execution time** was spent on `pd.read_csv()`. This proved that parallelizing the math (using `multiprocessing`) was counterproductive, because the data loading was the true bottleneck.

## How to Reproduce
1. Clone this repo.
2. Install dependencies: `pip install pandas polars pyarrow`
3. Generate the data: `python generate_data.py`
4. Run the Pandas version: `python process_data.py`
5. Run the Polars version: `python process_data_v2.py`
6. Run the parquet test: `python convert_to_parquet.py`
7. Run the large parquet test: `python parquet_column_test.py`

## Future Work
- Implement multiprocessing to use all CPU cores (for larger datasets where the math dominates).
- Extend benchmarks to GPU-accelerated libraries (CuPy, PyTorch).
- Build a distributed training pipeline using PyTorch DDP.
- Benchmark Parquet with different compression algorithms (Snappy vs Gzip).

### Benchmark 3: Distributed Data Parallel (DDP) Training

I implemented a minimal DDP training script in PyTorch to demonstrate how gradients are synchronized across multiple processes.

- **What it does:** Simulates training a model across 2 GPUs (or CPUs).
- **Key concepts:** Process groups, rendezvous, gradient averaging.
- **Output:** Both ranks complete successfully with different data slices.

This is the foundation for training large language models across clusters.

### Benchmark 4: CPU vs GPU Training (M4)

I benchmarked training a simple neural network on both the CPU and the M4 GPU (via MPS). 

- **Small model (4M params, batch 256):** CPU was faster (1.25s vs 1.66s). The GPU was underutilized.
- **Large model (32M params, batch 1024):** GPU was faster (11.55s vs 14.82s). The larger workload allowed the GPU to use its cores effectively.

**Key Finding:** GPUs only outperform CPUs when the workload is large enough to saturate their cores. For small models, the CPU is often competitive.