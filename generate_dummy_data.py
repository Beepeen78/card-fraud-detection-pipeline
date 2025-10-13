"""
Generate dummy transaction data for testing the fraud detection pipeline.
Creates realistic transaction patterns with both normal and fraudulent transactions.
"""
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random
from pathlib import Path

def generate_dummy_transactions(num_transactions=1000, fraud_rate=0.05):
    """Generate dummy transaction data with realistic patterns."""
    
    # Set random seed for reproducibility
    np.random.seed(42)
    random.seed(42)
    
    # Generate transaction IDs
    transaction_ids = [f"TXN_{i:06d}" for i in range(1, num_transactions + 1)]
    
    # Generate customer IDs (some customers have multiple transactions)
    customer_ids = [f"CUST_{random.randint(1, 200):04d}" for _ in range(num_transactions)]
    
    # Generate timestamps (last 30 days)
    start_date = datetime.now() - timedelta(days=30)
    timestamps = []
    for _ in range(num_transactions):
        random_days = random.randint(0, 30)
        random_hours = random.randint(0, 23)
        random_minutes = random.randint(0, 59)
        timestamp = start_date + timedelta(days=random_days, hours=random_hours, minutes=random_minutes)
        timestamps.append(timestamp)
    
    # Generate amounts (log-normal distribution for realistic amounts)
    amounts = np.random.lognormal(mean=3, sigma=1, size=num_transactions)
    amounts = np.round(amounts, 2)
    
    # Generate merchant categories
    categories = ['grocery', 'gas', 'restaurant', 'retail', 'online', 'pharmacy', 'entertainment', 'travel']
    merchant_categories = [random.choice(categories) for _ in range(num_transactions)]
    
    # Generate merchant names
    merchant_names = [f"Merchant_{random.randint(1, 500):03d}" for _ in range(num_transactions)]
    
    # Generate merchant cities
    cities = ['New York', 'Los Angeles', 'Chicago', 'Houston', 'Phoenix', 'Philadelphia', 'San Antonio', 'San Diego', 'Dallas', 'San Jose']
    merchant_cities = [random.choice(cities) for _ in range(num_transactions)]
    
    # Generate customer locations (home cities)
    customer_cities = [random.choice(cities) for _ in range(num_transactions)]
    
    # Generate coordinates (roughly US coordinates)
    # Customer coordinates (home locations)
    customer_lat = np.random.uniform(25.0, 49.0, num_transactions)  # US latitude range
    customer_long = np.random.uniform(-125.0, -66.0, num_transactions)  # US longitude range
    
    # Merchant coordinates (can be far from customer for fraud)
    merchant_lat = np.random.uniform(25.0, 49.0, num_transactions)
    merchant_long = np.random.uniform(-125.0, -66.0, num_transactions)
    
    # Generate gender
    genders = ['M', 'F']
    customer_genders = [random.choice(genders) for _ in range(num_transactions)]
    
    # Generate jobs
    jobs = ['Engineer', 'Teacher', 'Doctor', 'Lawyer', 'Student', 'Retail', 'Manager', 'Analyst', 'Consultant', 'Sales']
    customer_jobs = [random.choice(jobs) for _ in range(num_transactions)]
    
    # Generate city population
    city_populations = np.random.randint(10000, 2000000, num_transactions)
    
    # Generate merchant ZIP codes
    merchant_zipcodes = [f"{random.randint(10000, 99999):05d}" for _ in range(num_transactions)]
    
    # Generate months
    months = [random.randint(1, 12) for _ in range(num_transactions)]
    
    # Now generate fraud labels and adjust patterns for fraudulent transactions
    is_fraud = np.random.random(num_transactions) < fraud_rate
    fraud_indices = np.where(is_fraud)[0]
    
    # Adjust fraudulent transactions to have suspicious patterns
    for idx in fraud_indices:
        # Fraudulent transactions often have:
        # 1. Higher amounts
        amounts[idx] *= np.random.uniform(2, 10)
        
        # 2. Transactions at unusual hours (late night/early morning)
        hour = timestamps[idx].hour
        if random.random() < 0.7:  # 70% chance of unusual time
            new_hour = random.choice([0, 1, 2, 3, 4, 5, 22, 23])
            timestamps[idx] = timestamps[idx].replace(hour=new_hour)
        
        # 3. Large distance from home (travel fraud)
        if random.random() < 0.6:  # 60% chance of far transaction
            # Place merchant far from customer
            merchant_lat[idx] = np.random.uniform(25.0, 49.0)
            merchant_long[idx] = np.random.uniform(-125.0, -66.0)
        
        # 4. High-risk categories
        if random.random() < 0.4:  # 40% chance of high-risk category
            merchant_categories[idx] = random.choice(['online', 'travel', 'entertainment'])
    
    # Round amounts to 2 decimal places
    amounts = np.round(amounts, 2)
    
    # Create DataFrame
    df = pd.DataFrame({
        'transaction_id': transaction_ids,
        'cc_num': customer_ids,
        'customer_id': customer_ids,  # Alternative column name
        'amount': amounts,
        'trans_date_trans_time': timestamps,
        'merch_lat': merchant_lat,
        'merch_long': merchant_long,
        'lat': customer_lat,
        'long': customer_long,
        'gender': customer_genders,
        'job': customer_jobs,
        'merch_zipcode': merchant_zipcodes,
        'city_pop': city_populations,
        'month': months,
        'merchant_category': merchant_categories,
        'merchant_name': merchant_names,
        'merchant_city': merchant_cities,
        'customer_city': customer_cities,
        'is_fraud': is_fraud.astype(int)
    })
    
    # Sort by timestamp
    df = df.sort_values('trans_date_trans_time').reset_index(drop=True)
    
    return df

def save_dummy_data(num_transactions=1000, fraud_rate=0.05):
    """Generate and save dummy transaction data."""
    
    print(f"Generating {num_transactions} dummy transactions with {fraud_rate*100}% fraud rate...")
    
    # Generate data
    df = generate_dummy_transactions(num_transactions, fraud_rate)
    
    # Save to CSV
    output_file = Path("dummy_transactions.csv")
    df.to_csv(output_file, index=False)
    
    print(f"Dummy data saved to: {output_file}")
    print(f"Dataset summary:")
    print(f"   - Total transactions: {len(df):,}")
    print(f"   - Fraudulent transactions: {df['is_fraud'].sum():,}")
    print(f"   - Fraud rate: {df['is_fraud'].mean()*100:.2f}%")
    print(f"   - Date range: {df['trans_date_trans_time'].min()} to {df['trans_date_trans_time'].max()}")
    print(f"   - Amount range: ${df['amount'].min():.2f} to ${df['amount'].max():.2f}")
    
    # Show some sample data
    print(f"\nSample data:")
    print(df.head(10).to_string(index=False))
    
    # Show fraud distribution
    print(f"\nFraud distribution by category:")
    fraud_by_category = df.groupby('merchant_category')['is_fraud'].agg(['count', 'sum', 'mean'])
    fraud_by_category.columns = ['total_transactions', 'fraud_count', 'fraud_rate']
    fraud_by_category['fraud_rate'] = fraud_by_category['fraud_rate'] * 100
    print(fraud_by_category.sort_values('fraud_rate', ascending=False))
    
    return df

if __name__ == "__main__":
    # Generate different sized datasets
    print("Generating dummy transaction datasets for testing...")
    print("=" * 60)
    
    # Small dataset for quick testing
    print("\n1. Small dataset (500 transactions):")
    save_dummy_data(500, 0.08)
    
    # Medium dataset for more realistic testing
    print("\n2. Medium dataset (1000 transactions):")
    save_dummy_data(1000, 0.05)
    
    # Large dataset for performance testing
    print("\n3. Large dataset (5000 transactions):")
    save_dummy_data(5000, 0.03)
    
    print("\nAll dummy datasets generated successfully!")
    print("You can now upload any of these CSV files to test your fraud detection pipeline.")
