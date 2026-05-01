import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import xgboost as xgb
import warnings
warnings.filterwarnings('ignore')

from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
import os

def mean_absolute_percentage_error(y_true, y_pred): 
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    non_zero = y_true != 0
    if not np.any(non_zero): return 0.0
    return np.mean(np.abs((y_true[non_zero] - y_pred[non_zero]) / y_true[non_zero])) * 100

def mase(y_train, y_test, y_pred):
    y_train = np.array(y_train)
    naive_err = np.mean(np.abs(np.diff(y_train)))
    if naive_err == 0: return np.nan
    mae = mean_absolute_error(y_test, y_pred)
    return mae / naive_err

print("Loading data...")
df = pd.read_csv('../data/processed/monthly_agricultural_data.csv')

MONTH_MAP = {
    'January': 1, 'February': 2, 'March': 3, 'April': 4, 'May': 5, 'June': 6,
    'July': 7, 'August': 8, 'September': 9, 'October': 10, 'November': 11, 'December': 12
}
if 'month_num' not in df.columns:
    df['month_num'] = df['month'].map(MONTH_MAP)

df['date'] = pd.to_datetime(df['year'].astype(str) + '-' + df['month_num'].astype(str) + '-01')
df['month_sin'] = np.sin(2 * np.pi * df['month_num']/12)
df['month_cos'] = np.cos(2 * np.pi * df['month_num']/12)

df = df.sort_values(by=['Crop_nam', 'Location_district', 'date'])

for lag in [1, 2, 3, 6, 12]:
    df[f'Productio_lag_{lag}'] = df.groupby(['Crop_nam', 'Location_district'])['Productio'].shift(lag)
for win in [3, 6, 12]:
    df[f'Productio_rolling_mean_{win}'] = df.groupby(['Crop_nam', 'Location_district'])['Productio'].transform(lambda x: x.rolling(win).mean())

df['harvested'] = df['harvested'].fillna(0) if 'harvested' in df.columns else 0
df['yield'] = df['yield'].fillna(0) if 'yield' in df.columns else 0

df = df.dropna(subset=['Productio']).sort_values('date')

feature_cols = ['year', 'month_sin', 'month_cos',
                'Productio_lag_1', 'Productio_lag_2', 'Productio_lag_3',
                'Productio_lag_6', 'Productio_lag_12',
                'Productio_rolling_mean_3', 'Productio_rolling_mean_6', 'Productio_rolling_mean_12',
                'harvested', 'yield']

le_crop, le_dist = LabelEncoder(), LabelEncoder()
df['Crop_enc'] = le_crop.fit_transform(df['Crop_nam'])
df['Dist_enc'] = le_dist.fit_transform(df['Location_district'])

feature_cols_encoded = feature_cols + ['Crop_enc', 'Dist_enc']

train_df = df[df['date'] < '2020-01-01']
val_df = df[(df['date'] >= '2020-01-01') & (df['date'] < '2022-01-01')]
test_df = df[df['date'] >= '2022-01-01']

X_train, y_train = train_df[feature_cols_encoded].fillna(0), train_df['Productio']
X_val, y_val = val_df[feature_cols_encoded].fillna(0), val_df['Productio']
X_test, y_test = test_df[feature_cols_encoded].fillna(0), test_df['Productio']

scaler = StandardScaler()
numeric_cols = ['year', 'month_sin', 'month_cos', 'harvested', 'yield']
X_train.loc[:, numeric_cols] = scaler.fit_transform(X_train[numeric_cols])
X_val.loc[:, numeric_cols] = scaler.transform(X_val[numeric_cols])
X_test.loc[:, numeric_cols] = scaler.transform(X_test[numeric_cols])

def get_metrics(y_tr, y_te, pred_tr, pred_te):
    return {
        'Train MAE': mean_absolute_error(y_tr, pred_tr),
        'Test MAE': mean_absolute_error(y_te, pred_te),
        'Test RMSE': np.sqrt(mean_squared_error(y_te, pred_te)),
        'Test MAPE': mean_absolute_percentage_error(y_te, pred_te),
        'Test MASE': mase(y_tr, y_te, pred_te)
    }

metrics = {}

# 1. Linear Regression
lr = LinearRegression().fit(X_train, y_train)
metrics['Linear Regression'] = get_metrics(y_train, y_test, lr.predict(X_train), lr.predict(X_test))

# 2. Random Forest (Regularized)
rf = RandomForestRegressor(n_estimators=100, max_depth=5, min_samples_split=10, min_samples_leaf=5, max_features='sqrt', random_state=42, n_jobs=-1).fit(X_train, y_train)
metrics['Random Forest'] = get_metrics(y_train, y_test, rf.predict(X_train), rf.predict(X_test))

# 3. XGBoost (Regularized)
xgb_model = xgb.XGBRegressor(n_estimators=1000, max_depth=3, learning_rate=0.01, gamma=5, reg_lambda=5.0, random_state=42)
xgb_model.fit(X_train, y_train, eval_set=[(X_val, y_val)], early_stopping_rounds=50, verbose=False)
metrics['XGBoost'] = get_metrics(y_train, y_test, xgb_model.predict(X_train), xgb_model.predict(X_test))

# 4. ARIMA / SARIMA (Local Models - Regularized)
ts_crop, ts_dist = 'Cabbage', 'Anuradhapura'
ts_df = df[(df['Crop_nam'] == ts_crop) & (df['Location_district'] == ts_dist)]
ts_df = ts_df.set_index('date').asfreq('MS')
ts_df['Productio'] = ts_df['Productio'].ffill()

ts_train = ts_df[ts_df.index < '2022-01-01']['Productio']
ts_test = ts_df[ts_df.index >= '2022-01-01']['Productio']

arima = ARIMA(ts_train, order=(1,1,1)).fit()
arima_pred_tr = arima.predict(start=ts_train.index[0], end=ts_train.index[-1])
arima_pred_te = arima.forecast(steps=len(ts_test))
metrics['ARIMA (Local)'] = get_metrics(ts_train, ts_test, arima_pred_tr, arima_pred_te)

sarima = SARIMAX(ts_train, order=(0, 1, 1), seasonal_order=(0, 1, 0, 12)).fit(disp=False)
sarima_pred_tr = sarima.predict(start=ts_train.index[0], end=ts_train.index[-1])
sarima_pred_te = sarima.forecast(steps=len(ts_test))
metrics['SARIMA (Local)'] = get_metrics(ts_train, ts_test, sarima_pred_tr, sarima_pred_te)

print("\n--- NEW Model Evaluation Metrics (Generalization Focused) ---")
for m, vals in metrics.items():
    print(f"{m}:")
    for k, v in vals.items():
        print(f"  {k}: {v:.2f}")

plt.figure(figsize=(14, 6))
hist = df[(df['Crop_nam'] == ts_crop) & (df['Location_district'] == ts_dist)]
plot_df = hist[hist['date'] >= '2018-01-01'].copy()
X_plot = plot_df[feature_cols_encoded].fillna(0)
X_plot.loc[:, numeric_cols] = scaler.transform(X_plot[numeric_cols])
plot_df['xgb_pred'] = xgb_model.predict(X_plot)

plt.plot(plot_df['date'], plot_df['Productio'], label='Actual Production', color='black', linewidth=2)
plt.plot(plot_df['date'], plot_df['xgb_pred'], label='XGBoost Optimized', color='blue', linestyle='--')
plt.axvline(pd.to_datetime('2020-01-01'), color='red', linestyle=':', label='Start Validation')
plt.axvline(pd.to_datetime('2022-01-01'), color='green', linestyle=':', label='Start Test')
plt.title('Optimized Generalization: Cabbage in Anuradhapura (Actual vs XGBoost)')
plt.legend()
plt.grid(True)

out_dir = "C:/Users/HP/.gemini/antigravity/brain/54961611-c90e-4c6e-84fc-36be0caf49fd/artifacts/"
os.makedirs(out_dir, exist_ok=True)
plt.savefig(os.path.join(out_dir, "actual_vs_predicted.png"))
print("New graph saved to artifacts.")
