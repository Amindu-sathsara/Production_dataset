import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import xgboost as xgb
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
import os

df = pd.read_csv("../data/processed/monthly_agricultural_data.csv")

MONTH_MAP = {
    'January': 1, 'February': 2, 'March': 3, 'April': 4, 'May': 5, 'June': 6,
    'July': 7, 'August': 8, 'September': 9, 'October': 10, 'November': 11, 'December': 12
}

if 'month_num' not in df.columns:
    df['month_num'] = df['month'].map(MONTH_MAP)

df['date'] = pd.to_datetime(df['year'].astype(str) + '-' + df['month_num'].astype(str) + '-01')
df['month_sin'] = np.sin(2 * np.pi * df['month_num']/12)
df['month_cos'] = np.cos(2 * np.pi * df['month_num']/12)

# Lags and rolling means
df = df.sort_values(by=['Crop_nam', 'Location_district', 'date'])
for lag in [1, 2, 3, 6, 12]:
    df[f'Productio_lag_{lag}'] = df.groupby(['Crop_nam', 'Location_district'])['Productio'].shift(lag)

for win in [3, 6, 12]:
    df[f'Productio_rolling_mean_{win}'] = df.groupby(['Crop_nam', 'Location_district'])['Productio'].transform(lambda x: x.rolling(win).mean())

# Drop NAs
df = df.dropna(subset=['Productio'])

feature_cols = ['year', 'month_sin', 'month_cos',
                'Productio_lag_1', 'Productio_lag_2', 'Productio_lag_3',
                'Productio_lag_6', 'Productio_lag_12',
                'Productio_rolling_mean_3', 'Productio_rolling_mean_6', 'Productio_rolling_mean_12',
                'harvested', 'yield',
                'Crop_nam', 'Location_district']

le_crop = LabelEncoder()
le_dist = LabelEncoder()
df['Crop_enc'] = le_crop.fit_transform(df['Crop_nam'])
df['Dist_enc'] = le_dist.fit_transform(df['Location_district'])

feature_cols_encoded = ['year', 'month_sin', 'month_cos',
                        'Productio_lag_1', 'Productio_lag_2', 'Productio_lag_3',
                        'Productio_lag_6', 'Productio_lag_12',
                        'Productio_rolling_mean_3', 'Productio_rolling_mean_6', 'Productio_rolling_mean_12',
                        'harvested', 'yield',
                        'Crop_enc', 'Dist_enc']

train_df = df[df['date'] < '2021-01-01']
val_df = df[(df['date'] >= '2021-01-01') & (df['date'] < '2023-01-01')]
test_df = df[df['date'] >= '2023-01-01']

X_train = train_df[feature_cols_encoded].fillna(0)
y_train = train_df['Productio']
X_val = val_df[feature_cols_encoded].fillna(0)
y_val = val_df['Productio']
X_test = test_df[feature_cols_encoded].fillna(0)
y_test = test_df['Productio']

scaler = StandardScaler()
numeric_cols = ['year', 'month_sin', 'month_cos', 'harvested', 'yield']
X_train[numeric_cols] = scaler.fit_transform(X_train[numeric_cols])
X_val[numeric_cols] = scaler.transform(X_val[numeric_cols])
X_test[numeric_cols] = scaler.transform(X_test[numeric_cols])

model = xgb.XGBRegressor(
    n_estimators=800,
    max_depth=7,
    learning_rate=0.03,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    early_stopping_rounds=30,
    eval_metric='mae'
)

model.fit(
    X_train, y_train,
    eval_set=[(X_train, y_train), (X_val, y_val), (X_test, y_test)],
    verbose=False
)

results = model.evals_result()
epochs = len(results['validation_0']['mae'])
x_axis = range(0, epochs)

plt.figure(figsize=(10, 6))
plt.plot(x_axis, results['validation_0']['mae'], label='Train')
plt.plot(x_axis, results['validation_1']['mae'], label='Validation')
plt.plot(x_axis, results['validation_2']['mae'], label='Test')
plt.legend()
plt.ylabel('MAE')
plt.xlabel('Epochs')
plt.title('XGBoost Learning Curve (MAE over Epochs)')
plt.grid(True)
out_dir = "C:/Users/HP/.gemini/antigravity/brain/54961611-c90e-4c6e-84fc-36be0caf49fd/artifacts/"
os.makedirs(out_dir, exist_ok=True)
plt.savefig(os.path.join(out_dir, "learning_curve.png"))

y_pred_train = model.predict(X_train)
y_pred_val = model.predict(X_val)
y_pred_test = model.predict(X_test)

print(f"Train MAE: {mean_absolute_error(y_train, y_pred_train):.2f}")
print(f"Val MAE: {mean_absolute_error(y_val, y_pred_val):.2f}")
print(f"Test MAE: {mean_absolute_error(y_test, y_pred_test):.2f}")
