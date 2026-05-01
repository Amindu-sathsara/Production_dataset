import pandas as pd
import numpy as np
import joblib
import sys
import os
import json
from utils import MONTH_MAP

class ProductionPredictor:
    def __init__(self):
        self.base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        self.models_dir = os.path.join(self.base_dir, "models")
        self.data_path = os.path.join(self.base_dir, "data", "processed", "monthly_agricultural_data.csv")
        
        self.le_crop = joblib.load(os.path.join(self.models_dir, "le_crop.pkl"))
        self.le_dist = joblib.load(os.path.join(self.models_dir, "le_dist.pkl"))
        self.scaler = joblib.load(os.path.join(self.models_dir, "scaler.pkl"))
        
        with open(os.path.join(self.models_dir, "feature_cols.txt"), "r") as f:
            self.feature_cols = [line.strip() for line in f.readlines()]
            
        with open(os.path.join(self.models_dir, "best_model_mapping.json"), "r") as f:
            self.best_model_mapping = json.load(f)
            
        self.full_df = pd.read_csv(self.data_path)
        if 'month' in self.full_df.columns and self.full_df['month'].dtype == object:
            self.full_df['month_num'] = self.full_df['month'].map(MONTH_MAP)
        else:
            self.full_df['month_num'] = self.full_df['month']
        self.full_df['date'] = pd.to_datetime(self.full_df['year'].astype(str) + '-' + self.full_df['month_num'].astype(str) + '-01')

    def get_model_info(self, crop, district):
        key = f"{crop}_{district}"
        if key in self.best_model_mapping:
            return self.best_model_mapping[key]
        return {"best_model": "XGBoost", "accuracy": 0.0, "mae": 0.0}

    def load_model(self, model_name):
        if model_name == "Linear Regression":
            return joblib.load(os.path.join(self.models_dir, "linear_model.pkl"))
        elif model_name == "Random Forest":
            return joblib.load(os.path.join(self.models_dir, "rf_model.pkl"))
        else:
            return joblib.load(os.path.join(self.models_dir, "production_model.pkl"))

    def predict(self, year, month, crop, district):
        # 1. Determine best model and its accuracy
        info = self.get_model_info(crop, district)
        model_name = info['best_model']
        accuracy = info['accuracy']
        
        print(f"--- Smart Selection: Using {model_name} for {crop} in {district} ---")
        print(f"--- Reliability: {accuracy:.2f}% Expected Accuracy (MAE: {info['mae']:.2f}) ---")
        
        model = self.load_model(model_name)
        
        # 2. Prepare Features
        month_num = MONTH_MAP.get(month, 1)
        date = pd.to_datetime(f"{year}-{month_num}-01")
        
        hist = self.full_df[(self.full_df['Crop_nam'] == crop) & (self.full_df['Location_district'] == district)]
        if hist.empty:
            return f"Error: No historical data for {crop} in {district}"
            
        hist = hist.sort_values('date')
        
        lags = {f'Productio_lag_{i}': hist['Productio'].iloc[-i] if len(hist)>=i else 0 for i in [1, 2, 3, 6, 12]}
        rolling = {f'Productio_rolling_mean_{w}': hist['Productio'].iloc[-w:].mean() if len(hist)>=w else 0 for w in [3, 6, 12]}
        
        input_data = {
            'year': year,
            'month_sin': np.sin(2 * np.pi * month_num / 12),
            'month_cos': np.cos(2 * np.pi * month_num / 12),
            **lags,
            **rolling,
            'harvested': hist['harvested'].mean(),
            'yield': hist['yield'].mean(),
            'Crop_enc': self.le_crop.transform([crop])[0],
            'Dist_enc': self.le_dist.transform([district])[0]
        }
        
        input_df = pd.DataFrame([input_data])[self.feature_cols]
        numeric_cols = ['year', 'month_sin', 'month_cos', 'harvested', 'yield']
        input_df.loc[:, numeric_cols] = self.scaler.transform(input_df[numeric_cols])
        
        prediction = model.predict(input_df)[0]
        return max(0, prediction)

if __name__ == "__main__":
    if len(sys.argv) < 5:
        print("Usage: python src/predict.py <year> <month> <crop> <district>")
        sys.exit(1)
        
    year = int(sys.argv[1])
    month = sys.argv[2]
    crop = sys.argv[3]
    district = sys.argv[4]
    
    predictor = ProductionPredictor()
    result = predictor.predict(year, month, crop, district)
    
    if isinstance(result, str):
        print(result)
    else:
        print(f"Predicted production for {crop} in {district}, {month} {year}: {result:.2f} Metric Tons")