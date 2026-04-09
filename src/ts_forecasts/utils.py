import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error

# Month mapping (same as in your main utils.py, but duplicated for independence)
MONTH_MAP = {
    'January': 1, 'February': 2, 'March': 3, 'April': 4, 'May': 5, 'June': 6,
    'July': 7, 'August': 8, 'September': 9, 'October': 10, 'November': 11, 'December': 12
}

def load_series(crop, district, data_path="../../data/processed/monthly_agricultural_data.csv"):
    """
    Load time series for a specific crop and district.
    Returns pandas Series with monthly frequency and 'Productio' values.
    """
    df = pd.read_csv(data_path)
    df = df[(df['Crop_nam'] == crop) & (df['Location_district'] == district)]
    if df.empty:
        return None
    df['month_num'] = df['month'].map(MONTH_MAP)
    df['date'] = pd.to_datetime(df['year'].astype(str) + '-' + df['month_num'].astype(str) + '-01')
    df = df[['date', 'Productio']].dropna().set_index('date').asfreq('MS')
    # Forward fill missing values
    df['Productio'] = df['Productio'].fillna(method='ffill')
    return df['Productio']

def evaluate_forecast(y_true, y_pred):
    """Compute MAE, RMSE, MAPE, MASE."""
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    # MAPE (avoid division by zero)
    mape = np.mean(np.abs((y_true - y_pred) / (y_true + 1e-8))) * 100
    # MASE: scale by in-sample seasonal naive (difference from 12 months ago)
    if len(y_true) >= 12:
        naive_errors = np.abs(np.diff(y_true, n=12))
        scale = np.mean(naive_errors) if len(naive_errors) > 0 else 1
    else:
        scale = 1
    mase = mae / scale
    return {'MAE': mae, 'RMSE': rmse, 'MAPE': mape, 'MASE': mase}