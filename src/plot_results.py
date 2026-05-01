import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib
import sys
import os
from sklearn.preprocessing import LabelEncoder, StandardScaler
import warnings
warnings.filterwarnings('ignore')

from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX

MONTH_MAP = {
    'January': 1, 'February': 2, 'March': 3, 'April': 4, 'May': 5, 'June': 6,
    'July': 7, 'August': 8, 'September': 9, 'October': 10, 'November': 11, 'December': 12
}

def generate_graph(model_name, crop_name, district_name):
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_path = os.path.join(base_dir, "data", "processed", "monthly_agricultural_data.csv")
    models_dir = os.path.join(base_dir, "models")
    output_dir = os.path.join(base_dir, "output")
    os.makedirs(output_dir, exist_ok=True)

    print(f"Loading data for {crop_name} in {district_name}...")
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

    # If it's a Global ML Model
    if model_name.lower() in ['xgboost', 'linear', 'random forest']:
        le_crop = joblib.load(os.path.join(models_dir, "le_crop.pkl"))
        le_dist = joblib.load(os.path.join(models_dir, "le_dist.pkl"))
        scaler = joblib.load(os.path.join(models_dir, "scaler.pkl"))
        
        with open(os.path.join(models_dir, "feature_cols.txt"), "r") as f:
            feature_cols_encoded = [line.strip() for line in f.readlines()]
            
        target_df['Crop_enc'] = le_crop.transform(target_df['Crop_nam'])
        target_df['Dist_enc'] = le_dist.transform(target_df['Location_district'])
        
        X_plot = target_df[feature_cols_encoded].fillna(0)
        numeric_cols = ['year', 'month_sin', 'month_cos', 'harvested', 'yield']
        X_plot.loc[:, numeric_cols] = scaler.transform(X_plot[numeric_cols])

        if model_name.lower() == 'xgboost':
            model = joblib.load(os.path.join(models_dir, "production_model.pkl"))
        elif model_name.lower() == 'linear':
            model = joblib.load(os.path.join(models_dir, "linear_model.pkl"))
        elif model_name.lower() == 'random forest':
            model = joblib.load(os.path.join(models_dir, "rf_model.pkl"))

        target_df['Predicted'] = model.predict(X_plot)
        
    # If it's a Local Time Series Model
    elif model_name.lower() in ['arima', 'sarima']:
        ts_df = target_df.set_index('date').asfreq('MS')
        ts_df['Productio'] = ts_df['Productio'].ffill()
        
        # Chronological Split
        ts_train = ts_df[ts_df.index < '2022-01-01']['Productio']
        ts_test = ts_df[ts_df.index >= '2022-01-01']['Productio']
        
        print(f"Training Local {model_name.upper()} model for graph generation (this takes a moment)...")
        if model_name.lower() == 'arima':
            model = ARIMA(ts_train, order=(5,1,0)).fit()
        else:
            model = SARIMAX(ts_train, order=(1, 1, 1), seasonal_order=(1, 1, 1, 12)).fit(disp=False)
            
        pred_train = model.predict(start=ts_train.index[0], end=ts_train.index[-1])
        pred_test = model.forecast(steps=len(ts_test))
        
        target_df['Predicted'] = np.nan
        target_df.set_index('date', inplace=True)
        target_df.loc[pred_train.index, 'Predicted'] = pred_train.values
        target_df.loc[pred_test.index, 'Predicted'] = pred_test.values
        target_df = target_df.reset_index()
    else:
        print("Model must be 'XGBoost', 'Linear', 'Random Forest', 'ARIMA', or 'SARIMA'")
        return

    # Plotting
    plt.figure(figsize=(14, 6))
    plt.plot(target_df['date'], target_df['Productio'], label='Actual Production', color='black', linewidth=2)
    plt.plot(target_df['date'], target_df['Predicted'], label=f'{model_name.title()} Prediction', color='blue', linestyle='--')

    plt.axvline(pd.to_datetime('2020-01-01'), color='red', linestyle=':', label='Start Validation (2020)')
    plt.axvline(pd.to_datetime('2022-01-01'), color='green', linestyle=':', label='Start Test (2022)')

    plt.title(f'Train/Val/Test Timeline: {crop_name} in {district_name} using {model_name.title()}')
    plt.xlabel('Date')
    plt.ylabel('Production (Metric Tons)')
    plt.legend()
    plt.grid(True)

    filename = f"graph_{model_name.replace(' ', '_')}_{crop_name}_{district_name}.png"
    filepath = os.path.join(output_dir, filename)
    plt.savefig(filepath)
    print(f"\nSUCCESS: Graph generated and saved to:")
    print(filepath)

if __name__ == "__main__":
    if len(sys.argv) < 4:
        print("Usage: python src/plot_results.py <Model> <Crop> <District>")
        print("Models: 'XGBoost', 'Linear', 'Random Forest', 'ARIMA', 'SARIMA'")
        print("Example: python src/plot_results.py XGBoost Cabbage Anuradhapura")
        print("Example: python src/plot_results.py \"Random Forest\" Beans Colombo")
        sys.exit(1)

    model_arg = sys.argv[1]
    crop_arg = sys.argv[2]
    district_arg = sys.argv[3]
    generate_graph(model_arg, crop_arg, district_arg)
