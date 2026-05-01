import pandas as pd
import numpy as np
import joblib
import os
import json
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import LabelEncoder, StandardScaler
import warnings
warnings.filterwarnings('ignore')

MONTH_MAP = {
    'January': 1, 'February': 2, 'March': 3, 'April': 4, 'May': 5, 'June': 6,
    'July': 7, 'August': 8, 'September': 9, 'October': 10, 'November': 11, 'December': 12
}

def mean_absolute_percentage_error(y_true, y_pred): 
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    non_zero = y_true != 0
    if not np.any(non_zero): return 0.0
    return np.mean(np.abs((y_true[non_zero] - y_pred[non_zero]) / y_true[non_zero])) * 100

def generate_mapping():
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
    df['harvested'] = df['harvested'].fillna(0)
    df['yield'] = df['yield'].fillna(0)
    df = df.dropna(subset=['Productio']).sort_values('date')

    lr_model = joblib.load(os.path.join(models_dir, "linear_model.pkl"))
    rf_model = joblib.load(os.path.join(models_dir, "rf_model.pkl"))
    xgb_model = joblib.load(os.path.join(models_dir, "production_model.pkl"))
    le_crop = joblib.load(os.path.join(models_dir, "le_crop.pkl"))
    le_dist = joblib.load(os.path.join(models_dir, "le_dist.pkl"))
    scaler = joblib.load(os.path.join(models_dir, "scaler.pkl"))
    with open(os.path.join(models_dir, "feature_cols.txt"), "r") as f:
        feature_cols = [line.strip() for line in f.readlines()]

    mapping = {}
    crops = df['Crop_nam'].unique()
    districts = df['Location_district'].unique()

    for crop in crops:
        for dist in districts:
            target_df = df[(df['Crop_nam'] == crop) & (df['Location_district'] == dist)]
            test_df = target_df[target_df['date'] >= '2022-01-01'].copy()
            if test_df.empty: continue
                
            test_df['Crop_enc'] = le_crop.transform(test_df['Crop_nam'])
            test_df['Dist_enc'] = le_dist.transform(test_df['Location_district'])
            X_test = test_df[feature_cols].fillna(0)
            numeric_cols = ['year', 'month_sin', 'month_cos', 'harvested', 'yield']
            X_test.loc[:, numeric_cols] = scaler.transform(X_test[numeric_cols])
            y_true = test_df['Productio']

            # Evaluate each model
            evals = {}
            for name, model in [("Linear Regression", lr_model), ("Random Forest", rf_model), ("XGBoost", xgb_model)]:
                preds = model.predict(X_test)
                mae = mean_absolute_error(y_true, preds)
                mape = mean_absolute_percentage_error(y_true, preds)
                evals[name] = {"mae": mae, "mape": mape}

            # Find best by MAE
            best_model_name = min(evals, key=lambda x: evals[x]['mae'])
            winner = evals[best_model_name]
            accuracy = max(0, 100 - winner['mape'])

            key = f"{crop}_{dist}"
            mapping[key] = {
                "best_model": best_model_name,
                "mae": float(winner['mae']),
                "accuracy": float(accuracy)
            }

    with open(os.path.join(models_dir, "best_model_mapping.json"), "w") as f:
        json.dump(mapping, f, indent=4)
        
    print(f"SUCCESS: Updated Mapping with Accuracy data saved.")

if __name__ == "__main__":
    generate_mapping()
