import pandas as pd
import numpy as np

MONTH_MAP = {
    'January': 1, 'February': 2, 'March': 3, 'April': 4, 'May': 5, 'June': 6,
    'July': 7, 'August': 8, 'September': 9, 'October': 10, 'November': 11, 'December': 12
}

def load_and_preprocess(data_path):
    df = pd.read_csv(data_path)
    
    # Basic cleaning
    df = df.dropna(subset=['Productio'])
    df = df[df['Productio'] >= 0]
    
    # Add time features inline instead of importing broken utils
    if 'month_num' not in df.columns:
        df['month_num'] = df['month'].map(MONTH_MAP)
        
    df['date'] = pd.to_datetime(df['year'].astype(str) + '-' + df['month_num'].astype(str) + '-01')
    df['month_sin'] = np.sin(2 * np.pi * df['month_num']/12)
    df['month_cos'] = np.cos(2 * np.pi * df['month_num']/12)
    
    # Sort for lag creation
    df = df.sort_values(by=['Crop_nam', 'Location_district', 'date'])
    
    # Create lags
    for lag in [1, 2, 3, 6, 12]:
        df[f'Productio_lag_{lag}'] = df.groupby(['Crop_nam', 'Location_district'])['Productio'].shift(lag)

    # Create rolling means
    for win in [3, 6, 12]:
        df[f'Productio_rolling_mean_{win}'] = df.groupby(['Crop_nam', 'Location_district'])['Productio'].transform(lambda x: x.rolling(win).mean())
    
    # Optional: use harvested area and yield as features
    if 'harvested' in df.columns:
        df['harvested'] = df['harvested'].fillna(0)
    else:
        df['harvested'] = 0
        
    if 'yield' in df.columns:
        df['yield'] = df['yield'].fillna(0)
    else:
        df['yield'] = 0
    
    # Drop rows where target is missing after lag creation (first months of each group)
    df = df.dropna(subset=['Productio'])
    
    df = df.sort_values('date')
    return df

if __name__ == "__main__":
    df = load_and_preprocess("../data/processed/monthly_agricultural_data.csv")
    print(f"Preprocessed shape: {df.shape}")
    print(df.head())
    df.to_csv("../data/processed/preprocessed_full.csv", index=False)