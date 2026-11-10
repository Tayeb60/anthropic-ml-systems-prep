import pandas as pd
import numpy as np
import time

# --- 1. Generate a BIGGER dataset (100 million rows) ---
print("Generating 10 million rows (smaller to save disk space, but big enough to test)...")
num_rows = 10_000_000
df = pd.DataFrame({
    'Transaction_ID': np.arange(1, num_rows + 1),
    'Category': np.random.choice(['Electronics', 'Clothing', 'Home', 'Toys', 'Books'], num_rows),
    'Price': np.random.uniform(5.0, 500.0, num_rows).round(2),
    'Quantity': np.random.randint(1, 10, num_rows)
})

print("Saving as CSV...")
df.to_csv('big_sales_data.csv', index=False)

print("Saving as Parquet...")
df.to_parquet('big_sales_data.parquet', index=False)

# --- 2. Test 1: Full CSV Load ---
print("\n--- Test 1: Full CSV Load ---")
start = time.time()
df_csv = pd.read_csv('big_sales_data.csv')
print(f"CSV full load: {time.time() - start:.2f} seconds")

# --- 3. Test 2: Full Parquet Load ---
print("\n--- Test 2: Full Parquet Load ---")
start = time.time()
df_parquet = pd.read_parquet('big_sales_data.parquet')
print(f"Parquet full load: {time.time() - start:.2f} seconds")

# --- 4. Test 3: Parquet Column-Selective Load ---
print("\n--- Test 3: Parquet Column-Selective Load (Category & Price only) ---")
start = time.time()
df_subset = pd.read_parquet('big_sales_data.parquet', columns=['Category', 'Price'])
print(f"Parquet partial load: {time.time() - start:.2f} seconds")