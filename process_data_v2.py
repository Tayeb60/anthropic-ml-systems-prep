import polars as pl
import time

start_time = time.time()

# 1. Load the data
df = pl.read_csv('sales_data.csv')

# 2. Filter and calculate the sum in one go
revenue = (
    df.filter(pl.col("Category") == "Electronics")
    .select((pl.col("Price") * pl.col("Quantity")).sum())
    .item() # .item() extracts the single number from the result
)

end_time = time.time()

print(f"Revenue from Electronics category: ${revenue:.2f}")
print(f"Processing time: {end_time - start_time:.2f} seconds")