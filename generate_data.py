import csv
import random
import time

def create_large_csv(filename, num_rows=1000000):
    """Generates a large CSV file with random data."""
    print(f"Generating {num_rows} rows of data...")
    
    # Define some sample data
    categories = ["Electronics", "Clothing", "Home", "Toys", "Books"]
    
    start_time = time.time()
    
    # Open the file and write the data
    with open(filename, mode='w', newline='') as file:
        writer = csv.writer(file)
        # Write the header
        writer.writerow(["Transaction_ID", "Category", "Price", "Quantity"])
        
        # Write 1 million rows
        for i in range(1, num_rows + 1):
            writer.writerow([
                i,
                random.choice(categories),
                round(random.uniform(5.0, 500.0), 2),
                random.randint(1, 10)
            ])
            
    end_time = time.time()
    print(f"Done! Generated {num_rows} rows in {end_time - start_time:.2f} seconds.")

if __name__ == "__main__":
    create_large_csv("sales_data.csv", num_rows=1000000)