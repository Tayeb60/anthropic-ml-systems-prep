import pandas as pd
import time

print("Loading CSV...")
start_time = time.time()
df = pd.read_csv('sales_data.csv')
csv_load_time = time.time() - start_time
print(f"CSV loaded in {csv_load_time:.2f} seconds.")

print("Saving as Parquet...")
start_time = time.time()
df.to_parquet('sales_data.parquet')
parquet_save_time = time.time() - start_time
print(f"Parquet saved in {parquet_save_time:.2f} seconds.")

print("Loading Parquet...")
start_time = time.time()
df_parquet = pd.read_parquet('sales_data.parquet')
parquet_load_time = time.time() - start_time
print(f"Parquet loaded in {parquet_load_time:.2f} seconds.")