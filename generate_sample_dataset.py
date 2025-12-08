#!/usr/bin/env python
"""
Generate a sample dataset suitable for the fraud detection model.
This creates a CSV with base features that can be used to demonstrate the model.
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import random

def generate_sample_dataset(n_samples=100, output_path="sample_transactions.csv"):
    """
    Generate a sample dataset with realistic transaction data.
    
    Args:
        n_samples: Number of transactions to generate
        output_path: Path to save the CSV file
    """
    np.random.seed(42)
    random.seed(42)
    
    # Base timestamp (recent date)
    base_time = int(datetime(2024, 1, 1).timestamp())
    
    # Categories
    categories = ['grocery', 'shopping', 'travel', 'gas_transport', 'food_dining', 'entertainment']
    
    # Generate transactions
    data = {
        'unix_time': [],
        'amt': [],
        'city_pop': [],
        'dist_home_merch': [],
        'category': []
    }
    
    for i in range(n_samples):
        # Time: spread over a few days
        time_offset = random.randint(0, 7 * 24 * 3600)  # 7 days
        data['unix_time'].append(base_time + time_offset)
        
        # Amount: realistic transaction amounts (some high for potential fraud)
        if random.random() < 0.1:  # 10% chance of high amount (potential fraud indicator)
            amt = np.random.lognormal(mean=5.5, sigma=0.8)  # Higher amounts
        else:
            amt = np.random.lognormal(mean=4.0, sigma=0.6)  # Normal amounts
        data['amt'].append(round(amt, 2))
        
        # City population: realistic city sizes
        city_pop = int(np.random.lognormal(mean=11.5, sigma=0.8))
        data['city_pop'].append(city_pop)
        
        # Distance from home to merchant: some far distances (fraud indicator)
        if random.random() < 0.15:  # 15% chance of far distance
            dist = np.random.exponential(scale=50)  # Far distances
        else:
            dist = np.random.exponential(scale=5)  # Normal distances
        data['dist_home_merch'].append(round(dist, 2))
        
        # Category
        data['category'].append(random.choice(categories))
    
    # Create DataFrame
    df = pd.DataFrame(data)
    
    # Sort by time
    df = df.sort_values('unix_time').reset_index(drop=True)
    
    # Save to CSV
    df.to_csv(output_path, index=False)
    print(f"✅ Generated {n_samples} sample transactions saved to {output_path}")
    print(f"   Columns: {list(df.columns)}")
    print(f"   Amount range: ${df['amt'].min():.2f} - ${df['amt'].max():.2f}")
    print(f"   Time range: {datetime.fromtimestamp(df['unix_time'].min())} to {datetime.fromtimestamp(df['unix_time'].max())}")
    
    return df

if __name__ == "__main__":
    # Generate sample dataset
    df = generate_sample_dataset(n_samples=100, output_path="sample_transactions.csv")
    print("\n✅ Sample dataset ready for use!")
