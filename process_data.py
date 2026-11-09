import pandas as pd
import time

start_time = time.time()

df = pd.read_csv('sales_data.csv')

electronics_df = df[df['Category'] == 'Electronics']

revenue = (electronics_df['Price'] * electronics_df['Quantity']).sum()

end_time = time.time()
print(f"Revenue from Electronics category: ${revenue:.2f}")
print(f"Processing time: {end_time - start_time:.2f} seconds")