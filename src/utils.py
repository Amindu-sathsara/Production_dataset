import pandas as pd
import numpy as np

MONTH_MAP = {
    'January': 1, 'February': 2, 'March': 3, 'April': 4, 'May': 5, 'June': 6,
    'July': 7, 'August': 8, 'September': 9, 'October': 10, 'November': 11, 'December': 12
}
MONTH_LIST = list(MONTH_MAP.keys())

def add_time_features(df):
    """Adds cyclic month features (sin/cos) and ensures year is int."""
    df = df.copy()
    if 'month' in df.columns and df['month'].dtype == object:
        df['month_num'] = df['month'].map(MONTH_MAP)
    else:
        df['month_num'] = df['month']
    df['month_sin'] = np.sin(2 * np.pi * df['month_num'] / 12)
    df['month_cos'] = np.cos(2 * np.pi * df['month_num'] / 12)
    df['year'] = df['year'].astype(int)
    return df

def create_lag_features(df, group_cols, target_col, lags=[1,2,3,6,12], windows=[3,6,12]):
    """Adds lag and rolling mean features per group (crop, district)."""
    df = df.copy()
    df = df.sort_values(group_cols + ['year', 'month_num'])
    for lag in lags:
        df[f'{target_col}_lag_{lag}'] = df.groupby(group_cols)[target_col].shift(lag)
    for win in windows:
        df[f'{target_col}_rolling_mean_{win}'] = df.groupby(group_cols)[target_col].transform(
            lambda x: x.rolling(win, min_periods=1).mean()
        )
    return df