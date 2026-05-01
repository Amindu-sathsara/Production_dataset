import pandas as pd
import numpy as np
import joblib
import sys
import os
import json
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
import warnings
warnings.filterwarnings('ignore')

from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX

MONTH_MAP = {
    'January': 1, 'February': 2, 'March': 3, 'April': 4, 'May': 5, 'June': 6,
    'July': 7, 'August': 8, 'September': 9, 'October': 10, 'November': 11, 'December': 12
}

def mean_absolute_percentage_error(y_true, y_pred): 
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    non_zero = y_true != 0
    if not np.any(non_zero): return 0.0
    return np.mean(np.abs((y_true[non_zero] - y_pred[non_zero]) / y_true[non_zero])) * 100

def evaluate_model(model_name, crop_name, district_name):
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_path = os.path.join(base_dir, "data", "processed", "monthly_agricultural_data.csv")
    models_dir = os.path.join(base_dir, "models")

    df = pd.read_csv(data_path)
    
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
    
    target_df = df[(df['Crop_nam'] == crop_name) & (df['Location_district'] == district_name)].copy()
    if target_df.empty:
        print(f"Error: No data found for {crop_name} in {district_name}")
        return

    # Split
    test_df = target_df[target_df['date'] >= '2022-01-01']
    if test_df.empty:
        print("Error: No test data available (>= 2022) to evaluate.")
        return
        
    y_test = test_df['Productio']

    print(f"--- Evaluating {model_name.title()} for {crop_name} in {district_name} ---\n")

    if any(m in model_name.lower() for m in ['xgboost', 'linear', 'random forest']):
        le_crop = joblib.load(os.path.join(models_dir, "le_crop.pkl"))
        le_dist = joblib.load(os.path.join(models_dir, "le_dist.pkl"))
        scaler = joblib.load(os.path.join(models_dir, "scaler.pkl"))
        
        with open(os.path.join(models_dir, "feature_cols.txt"), "r") as f:
            feature_cols_encoded = [line.strip() for line in f.readlines()]
            
        test_df['Crop_enc'] = le_crop.transform(test_df['Crop_nam'])
        test_df['Dist_enc'] = le_dist.transform(test_df['Location_district'])
        
        X_test = test_df[feature_cols_encoded].fillna(0)
        numeric_cols = ['year', 'month_sin', 'month_cos', 'harvested', 'yield']
        X_test.loc[:, numeric_cols] = scaler.transform(X_test[numeric_cols])

        if 'xgboost' in model_name.lower():
            model = joblib.load(os.path.join(models_dir, "production_model.pkl"))
        elif 'linear' in model_name.lower():
            model = joblib.load(os.path.join(models_dir, "linear_model.pkl"))
        elif 'random forest' in model_name.lower():
            model = joblib.load(os.path.join(models_dir, "rf_model.pkl"))

        y_pred = model.predict(X_test)
        
    elif model_name.lower() in ['arima', 'sarima']:
        ts_df = target_df.set_index('date').asfreq('MS')
        ts_df['Productio'] = ts_df['Productio'].ffill()
        
        ts_train = ts_df[ts_df.index < '2022-01-01']['Productio']
        y_test = ts_df[ts_df.index >= '2022-01-01']['Productio']
        
        print("Training Time Series model for evaluation...")
        if model_name.lower() == 'arima':
            model = ARIMA(ts_train, order=(5,1,0)).fit()
        else:
            model = SARIMAX(ts_train, order=(1, 1, 1), seasonal_order=(1, 1, 1, 12)).fit(disp=False)
            
        y_pred = model.forecast(steps=len(y_test))
    else:
        print("Model must be 'XGBoost', 'Linear', 'Random Forest', 'ARIMA', or 'SARIMA'")
        return

    # Metrics Calculation
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mape = mean_absolute_percentage_error(y_test, y_pred)
    
    # Robust Accuracy Logic for Low-Volume Districts
    avg_actual = np.mean(y_test)
    if avg_actual < 10:
        # For low-volume districts, we evaluate accuracy based on how close the MAE is to the actual volume
        # If the error is small (e.g. 2 tons), we consider it high accuracy
        accuracy = max(0, 100 * (1 - (mae / (avg_actual + 5)))) # +5 as a smoothing constant
    else:
        # Standard MAPE-based accuracy for high-volume districts
        accuracy = max(0, 100 - mape)

    print(f"Mean Absolute Error (MAE):     {mae:.2f} Metric Tons")
    print(f"Root Mean Squared Error (RMSE): {rmse:.2f} Metric Tons")
    if avg_actual < 10:
        print(f"District Volume Status:        Low Production Volume (<10 MT)")
    else:
        print(f"Mean Abs Percentage Err (MAPE): {mape:.2f}%")
    print(f"---------------------------------------------------")
    print(f"Estimated Accuracy Percentage:  {accuracy:.2f}%\n")

if __name__ == "__main__":
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    mapping_path = os.path.join(base_dir, "models", "best_model_mapping.json")

    if len(sys.argv) == 3:
        # Smart Mode: Only Crop and District provided
        crop_arg = sys.argv[1]
        district_arg = sys.argv[2]
        
        if os.path.exists(mapping_path):
            with open(mapping_path, "r") as f:
                mapping = json.load(f)
            key = f"{crop_arg}_{district_arg}"
            if key in mapping:
                model_arg = mapping[key]['best_model']
                print(f"--- Smart Mode: Automatically selecting best model '{model_arg}' ---")
            else:
                print(f"Error: No mapping found for {crop_arg} in {district_arg}. Please specify a model.")
                sys.exit(1)
        else:
            print("Error: best_model_mapping.json not found. Please run generate_best_models.py first.")
            sys.exit(1)
            
    elif len(sys.argv) == 4:
        # Manual Mode: Model, Crop, and District provided
        model_arg = sys.argv[1]
        crop_arg = sys.argv[2]
        district_arg = sys.argv[3]
    else:
        print("Usage (Smart):  python src/evaluate_results.py <Crop> <District>")
        print("Usage (Manual): python src/evaluate_results.py <Model> <Crop> <District>")
        print("Models: 'XGBoost', 'Linear', 'Random Forest', 'ARIMA', 'SARIMA'")
        sys.exit(1)

    evaluate_model(model_arg, crop_arg, district_arg)
