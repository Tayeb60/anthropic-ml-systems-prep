import pandas as pd
import time
import os
from concurrent.futures import ProcessPoolExecutor

def process_chunk(chunk):
    """This function processes a small chunk of data."""
    # Filter for Electronics
    electronics = chunk[chunk['Category'] == 'Electronics']
    # Calculate revenue
    revenue = (electronics['Price'] * electronics['Quantity']).sum()
    return revenue

def main():
    start_time = time.time()
    
    # 1. Get the number of CPU cores on this machine
    num_cores = os.cpu_count()
    print(f"Found {num_cores} CPU cores. Starting parallel processing...")

    # 2. Load the data
    df = pd.read_csv('sales_data.csv')
    
    # 3. Split the data into chunks the PANDAS way (keeps column names)
    chunk_size = len(df) // num_cores
    chunks = [df.iloc[i:i + chunk_size] for i in range(0, len(df), chunk_size)]
    
    # 4. Process the chunks in parallel
    total_revenue = 0
    with ProcessPoolExecutor(max_workers=num_cores) as executor:
        # executor.map sends each chunk to a separate core and gets the result back
        results = executor.map(process_chunk, chunks)
        
        # Sum up the results from each core
        total_revenue = sum(results)

    end_time = time.time()
    print(f"Revenue from Electronics category: ${total_revenue:.2f}")
    print(f"Parallel processing time: {end_time - start_time:.2f} seconds")

if __name__ == "__main__":
    main()