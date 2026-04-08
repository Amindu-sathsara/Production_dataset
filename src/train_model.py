import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
import xgboost as xgb
from preprocess import load_and_preprocess

def train():
    # Load preprocessed data
    df = load_and_preprocess("../data/processed/monthly_agricultural_data.csv")
    
    # Features to use (excluding text, date, and target)
    feature_cols = ['year', 'month_sin', 'month_cos',
                    'Productio_lag_1', 'Productio_lag_2', 'Productio_lag_3',
                    'Productio_lag_6', 'Productio_lag_12',
                    'Productio_rolling_mean_3', 'Productio_rolling_mean_6', 'Productio_rolling_mean_12',
                    'harvested', 'yield',
                    'Crop_nam', 'Location_district']
    
    # Encode categoricals
    le_crop = LabelEncoder()
    le_dist = LabelEncoder()
    df['Crop_enc'] = le_crop.fit_transform(df['Crop_nam'])
    df['Dist_enc'] = le_dist.fit_transform(df['Location_district'])
    
    # Replace text columns with encoded ones
    feature_cols_encoded = ['year', 'month_sin', 'month_cos',
                            'Productio_lag_1', 'Productio_lag_2', 'Productio_lag_3',
                            'Productio_lag_6', 'Productio_lag_12',
                            'Productio_rolling_mean_3', 'Productio_rolling_mean_6', 'Productio_rolling_mean_12',
                            'harvested', 'yield',
                            'Crop_enc', 'Dist_enc']
    
    # Split chronologically (e.g., train before 2021, validate 2021-2022, test after)
    train_df = df[df['date'] < '2021-01-01']
    val_df = df[(df['date'] >= '2021-01-01') & (df['date'] < '2023-01-01')]
    test_df = df[df['date'] >= '2023-01-01']
    
    X_train = train_df[feature_cols_encoded]
    y_train = train_df['Productio']
    X_val = val_df[feature_cols_encoded]
    y_val = val_df['Productio']
    X_test = test_df[feature_cols_encoded]
    y_test = test_df['Productio']
    
    # Handle NaN in lag/rolling features (fill with 0)
    X_train = X_train.fillna(0)
    X_val = X_val.fillna(0)
    X_test = X_test.fillna(0)
    
    # Optional: scale numeric features (XGBoost doesn't need scaling, but can help)
    scaler = StandardScaler()
    numeric_cols = ['year', 'month_sin', 'month_cos', 'harvested', 'yield']
    X_train[numeric_cols] = scaler.fit_transform(X_train[numeric_cols])
    X_val[numeric_cols] = scaler.transform(X_val[numeric_cols])
    X_test[numeric_cols] = scaler.transform(X_test[numeric_cols])
    
    # Train XGBoost
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
        eval_set=[(X_val, y_val)],
        verbose=True
    )
    
    # Evaluate
    y_pred_val = model.predict(X_val)
    y_pred_test = model.predict(X_test)
    
    mae_val = mean_absolute_error(y_val, y_pred_val)
    rmse_val = np.sqrt(mean_squared_error(y_val, y_pred_val))
    mae_test = mean_absolute_error(y_test, y_pred_test)
    rmse_test = np.sqrt(mean_squared_error(y_test, y_pred_test))

    # MASE (Mean Absolute Scaled Error)
    # Scale factor: in-sample MAE of a naive "y_t = y_{t-1}" forecast on the training set
    if len(y_train) > 1:
        mae_naive_train = np.mean(np.abs(y_train.values[1:] - y_train.values[:-1]))
        mase_val = mae_val / mae_naive_train if mae_naive_train != 0 else np.nan
        mase_test = mae_test / mae_naive_train if mae_naive_train != 0 else np.nan
    else:
        mase_val = np.nan
        mase_test = np.nan
    
    print(f"Validation MAE: {mae_val:.2f}, RMSE: {rmse_val:.2f}, MASE: {mase_val:.3f}")
    print(f"Test MAE: {mae_test:.2f}, RMSE: {rmse_test:.2f}, MASE: {mase_test:.3f}")
    
    # Save model, encoders, scaler
    import os
    os.makedirs("../models", exist_ok=True)
    joblib.dump(model, "../models/production_model.pkl")
    joblib.dump(le_crop, "../models/le_crop.pkl")
    joblib.dump(le_dist, "../models/le_dist.pkl")
    joblib.dump(scaler, "../models/scaler.pkl")
    
    # Save feature list for later
    with open("../models/feature_cols.txt", "w") as f:
        for col in feature_cols_encoded:
            f.write(col + "\n")
    
    print("Model and encoders saved to ../models/")

if __name__ == "__main__":
    train()