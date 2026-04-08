import pandas as pd
import numpy as np
from utils import add_time_features, create_lag_features

def load_and_preprocess(data_path):
    df = pd.read_csv(data_path)
    
    # Basic cleaning
    df = df.dropna(subset=['Productio'])
    df = df[df['Productio'] >= 0]  # remove negative if any
    
    # Add time features
    df = add_time_features(df)
    
    # Create lags and rolling means per (Crop_nam, Location_district)
    df = create_lag_features(
        df,
        group_cols=['Crop_nam', 'Location_district'],
        target_col='Productio',
        lags=[1,2,3,6,12],
        windows=[3,6,12]
    )
    
    # Optional: use harvested area and yield as features
    df['harvested'] = df['harvested'].fillna(0)
    df['yield'] = df['yield'].fillna(0)
    
    # Drop rows where target is missing after lag creation (first months of each group)
    df = df.dropna(subset=['Productio'])
    
    # Create a date column for chronological splitting
    df['date'] = pd.to_datetime(df['year'].astype(str) + '-' + df['month_num'].astype(str) + '-01')
    df = df.sort_values('date')
    
    return df

if __name__ == "__main__":
    df = load_and_preprocess("../data/processed/monthly_agricultural_data.csv")
    print(f"Preprocessed shape: {df.shape}")
    print(df.head())
    df.to_csv("../data/processed/preprocessed_full.csv", index=False)