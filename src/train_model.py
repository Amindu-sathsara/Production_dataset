import pandas as pd
import numpy as np
import joblib
import os
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
import xgboost as xgb
from preprocess import load_and_preprocess

def train():
    print("Loading and preprocessing dataset...")
    df = load_and_preprocess("../data/processed/monthly_agricultural_data.csv")
    
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
    
    print("Splitting dataset chronologically...")
    train_df = df[df['date'] < '2020-01-01']
    val_df = df[(df['date'] >= '2020-01-01') & (df['date'] < '2022-01-01')]
    test_df = df[df['date'] >= '2022-01-01']
    
    X_train = train_df[feature_cols_encoded].fillna(0)
    y_train = train_df['Productio']
    X_val = val_df[feature_cols_encoded].fillna(0)
    y_val = val_df['Productio']
    X_test = test_df[feature_cols_encoded].fillna(0)
    y_test = test_df['Productio']
    
    print("Scaling features...")
    scaler = StandardScaler()
    numeric_cols = ['year', 'month_sin', 'month_cos', 'harvested', 'yield']
    X_train.loc[:, numeric_cols] = scaler.fit_transform(X_train[numeric_cols])
    X_val.loc[:, numeric_cols] = scaler.transform(X_val[numeric_cols])
    X_test.loc[:, numeric_cols] = scaler.transform(X_test[numeric_cols])
    
    os.makedirs("../models", exist_ok=True)
    results = {}

    print("\n--- Training Linear Regression ---")
    lr_model = LinearRegression()
    lr_model.fit(X_train, y_train)
    lr_mae = mean_absolute_error(y_test, lr_model.predict(X_test))
    results['Linear'] = lr_mae
    joblib.dump(lr_model, "../models/linear_model.pkl")
    print(f"Linear Regression Test MAE: {lr_mae:.2f}")

    print("\n--- Training Random Forest (Regularized) ---")
    rf_model = RandomForestRegressor(
        n_estimators=100, 
        max_depth=5, 
        min_samples_split=10, 
        min_samples_leaf=5, 
        max_features='sqrt', 
        random_state=42, 
        n_jobs=-1
    )
    rf_model.fit(X_train, y_train)
    rf_mae = mean_absolute_error(y_test, rf_model.predict(X_test))
    results['Random Forest'] = rf_mae
    joblib.dump(rf_model, "../models/rf_model.pkl")
    print(f"Random Forest Test MAE: {rf_mae:.2f}")

    print("\n--- Training XGBoost (Regularized) ---")
    xgb_model = xgb.XGBRegressor(
        n_estimators=1000, 
        max_depth=3, 
        learning_rate=0.01, 
        gamma=5, 
        reg_lambda=5.0, 
        random_state=42,
        early_stopping_rounds=50,
        eval_metric='mae'
    )
    xgb_model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
    xgb_mae = mean_absolute_error(y_test, xgb_model.predict(X_test))
    results['XGBoost'] = xgb_mae
    joblib.dump(xgb_model, "../models/production_model.pkl")
    print(f"XGBoost Test MAE: {xgb_mae:.2f}")
    
    joblib.dump(le_crop, "../models/le_crop.pkl")
    joblib.dump(le_dist, "../models/le_dist.pkl")
    joblib.dump(scaler, "../models/scaler.pkl")
    with open("../models/feature_cols.txt", "w") as f:
        for col in feature_cols_encoded:
            f.write(col + "\n")
            
    print("\nAll models and encoders saved to ../models/")

    plt.figure(figsize=(8, 5))
    bars = plt.bar(results.keys(), results.values(), color=['skyblue', 'lightgreen', 'salmon'])
    plt.ylabel('Test MAE')
    plt.title('Global Model Performance (Optimized for Generalization)')
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2, yval + 0.5, f"{yval:.2f}", ha='center', va='bottom')
    
    os.makedirs("../output", exist_ok=True)
    plot_path = os.path.abspath("../output/model_comparison.png")
    plt.savefig(plot_path)
    print(f"Comparison graph saved to {plot_path}")

if __name__ == "__main__":
    train()