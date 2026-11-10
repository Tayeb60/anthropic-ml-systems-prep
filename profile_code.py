import cProfile
import pstats
import pandas as pd

def load_and_process():
    # 1. Load data
    df = pd.read_csv('sales_data.csv')
    
    # 2. Filter and calculate
    electronics = df[df['Category'] == 'Electronics']
    revenue = (electronics['Price'] * electronics['Quantity']).sum()
    return revenue

# Run the profiler
cProfile.run('load_and_process()', 'profile_results')

# Print the results
p = pstats.Stats('profile_results')
p.sort_stats('cumulative').print_stats(10)