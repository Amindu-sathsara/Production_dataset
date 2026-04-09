import pandas as pd
import numpy as np
import joblib
import os
import sys
import warnings
warnings.filterwarnings('ignore')

# Add the parent directory to path so we can import utils
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from utils import load_series, evaluate_forecast

from pmdarima import auto_arima
from prophet import Prophet

def train_arima(train_series):
    try:
        model = auto_arima(train_series, seasonal=False, trace=False,
                           error_action='ignore', suppress_warnings=True)
        return model, 'ARIMA'
    except:
        return None, None

def train_sarima(train_series):
    try:
        model = auto_arima(train_series, seasonal=True, m=12, trace=False,
                           error_action='ignore', suppress_warnings=True)
        return model, 'SARIMA'
    except:
        return None, None

def train_prophet(train_series, train_dates):
    try:
        df = pd.DataFrame({'ds': train_dates, 'y': train_series.values})
        model = Prophet(yearly_seasonality=True, weekly_seasonality=False, daily_seasonality=False)
        model.fit(df)
        return model, 'Prophet'
    except:
        return None, None

def forecast_model(model, model_type, steps):
    if model_type in ['ARIMA', 'SARIMA']:
        pred = model.predict(n_periods=steps)
        return pred
    elif model_type == 'Prophet':
        future = model.make_future_dataframe(periods=steps, freq='MS')
        forecast = model.predict(future)
        return forecast['yhat'].iloc[-steps:].values
    else:
        return None

def main():
    # Resolve project root relative to this file so paths are stable
    this_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(os.path.dirname(this_dir))  # .../production
    data_path = os.path.join(project_root, "data", "processed", "monthly_agricultural_data.csv")
    model_dir = os.path.join(project_root, "models", "ts_models")
    os.makedirs(model_dir, exist_ok=True)
    
    df_all = pd.read_csv(data_path)
    pairs = df_all.groupby(['Crop_nam', 'Location_district']).size().reset_index()
    pairs.columns = ['crop', 'district', 'count']
    pairs = pairs[pairs['count'] >= 24]
    print(f"Found {len(pairs)} crop-district pairs with >=24 months.")
    
    results = []
    for idx, row in pairs.iterrows():
        crop = row['crop']
        district = row['district']
        print(f"\nProcessing {crop} - {district} ({idx+1}/{len(pairs)})")
        
        series = load_series(crop, district, data_path)
        if series is None or len(series) < 24:
            continue
        
        test_months = 12
        train = series.iloc[:-test_months]
        test = series.iloc[-test_months:]
        train_dates = train.index
        
        best_model = None
        best_type = None
        best_mae = float('inf')
        best_metrics = None
        
        # ARIMA
        model_ar, type_ar = train_arima(train)
        if model_ar:
            pred_ar = forecast_model(model_ar, type_ar, test_months)
            if pred_ar is not None and len(pred_ar) == len(test):
                metrics = evaluate_forecast(test.values, pred_ar)
                metrics['model'] = type_ar
                print(f"  ARIMA  -> MAE={metrics['MAE']:.2f}, RMSE={metrics['RMSE']:.2f}, MAPE={metrics['MAPE']:.1f}%, MASE={metrics['MASE']:.2f}")
                if metrics['MAE'] < best_mae:
                    best_mae = metrics['MAE']
                    best_model = model_ar
                    best_type = type_ar
                    best_metrics = metrics
        
        # SARIMA
        model_sar, type_sar = train_sarima(train)
        if model_sar:
            pred_sar = forecast_model(model_sar, type_sar, test_months)
            if pred_sar is not None and len(pred_sar) == len(test):
                metrics = evaluate_forecast(test.values, pred_sar)
                metrics['model'] = type_sar
                print(f"  SARIMA -> MAE={metrics['MAE']:.2f}, RMSE={metrics['RMSE']:.2f}, MAPE={metrics['MAPE']:.1f}%, MASE={metrics['MASE']:.2f}")
                if metrics['MAE'] < best_mae:
                    best_mae = metrics['MAE']
                    best_model = model_sar
                    best_type = type_sar
                    best_metrics = metrics
        
        # Prophet
        model_pro, type_pro = train_prophet(train, train_dates)
        if model_pro:
            pred_pro = forecast_model(model_pro, type_pro, test_months)
            if pred_pro is not None and len(pred_pro) == len(test):
                metrics = evaluate_forecast(test.values, pred_pro)
                metrics['model'] = type_pro
                print(f"  Prophet -> MAE={metrics['MAE']:.2f}, RMSE={metrics['RMSE']:.2f}, MAPE={metrics['MAPE']:.1f}%, MASE={metrics['MASE']:.2f}")
                if metrics['MAE'] < best_mae:
                    best_mae = metrics['MAE']
                    best_model = model_pro
                    best_type = type_pro
                    best_metrics = metrics
        
        if best_model:
            model_file = f"{model_dir}/{crop}_{district}.pkl"
            type_file = f"{model_dir}/{crop}_{district}_type.txt"
            joblib.dump(best_model, model_file)
            with open(type_file, 'w') as f:
                f.write(best_type)
            print(f"  => Best: {best_type} (MAE={best_metrics['MAE']:.2f})")
            results.append({
                'crop': crop,
                'district': district,
                'best_model': best_type,
                'MAE': best_metrics['MAE'],
                'RMSE': best_metrics['RMSE'],
                'MAPE': best_metrics['MAPE'],
                'MASE': best_metrics['MASE']
            })
        else:
            print("  No model succeeded.")
    
    results_df = pd.DataFrame(results)
    results_df.to_csv("../../models/ts_model_performance.csv", index=False)
    print("\n=== Summary ===")
    print(results_df.groupby('best_model').size())
    print("\nSaved evaluation results to ../../models/ts_model_performance.csv")

if __name__ == "__main__":
    main()