import pandas as pd
import numpy as np
import joblib
import sys
import os
from utils import MONTH_MAP

class ProductionPredictor:
    def __init__(self, models_dir="../models", data_path="../data/processed/monthly_agricultural_data.csv"):
        self.model = joblib.load(f"{models_dir}/production_model.pkl")
        self.le_crop = joblib.load(f"{models_dir}/le_crop.pkl")
        self.le_dist = joblib.load(f"{models_dir}/le_dist.pkl")
        self.scaler = joblib.load(f"{models_dir}/scaler.pkl")
        with open(f"{models_dir}/feature_cols.txt", "r") as f:
            self.feature_cols = [line.strip() for line in f.readlines()]
        self.data_path = data_path
        
        # Load full dataset to get last known values
        self.full_df = pd.read_csv(data_path)
        # Ensure month names are mapped to numbers
        if 'month' in self.full_df.columns and self.full_df['month'].dtype == object:
            self.full_df['month_num'] = self.full_df['month'].map(MONTH_MAP)
        else:
            self.full_df['month_num'] = self.full_df['month']
        self.full_df['date'] = pd.to_datetime(
            self.full_df['year'].astype(str) + '-' + 
            self.full_df['month_num'].astype(str) + '-01'
        )
    
    def get_last_known_sequence(self, crop_name, district_name, n_lags=12):
        """Return last n actual production values for the crop-district."""
        sub = self.full_df[(self.full_df['Crop_nam'] == crop_name) & 
                           (self.full_df['Location_district'] == district_name)]
        sub = sub.sort_values('date')
        if len(sub) == 0:
            return None, None, None
        
        productions = sub['Productio'].values[-n_lags:]
        # Build lag dict: most recent -> lag_1, second most recent -> lag_2, etc.
        lags = {}
        for i, val in enumerate(reversed(productions)):
            if i == 0:
                lags['Productio_lag_1'] = val
            elif i == 1:
                lags['Productio_lag_2'] = val
            elif i == 2:
                lags['Productio_lag_3'] = val
            elif i == 5:
                lags['Productio_lag_6'] = val
            elif i == 11:
                lags['Productio_lag_12'] = val
        # Fill missing lags with 0 (e.g., if not enough history)
        for lag in ['Productio_lag_1','Productio_lag_2','Productio_lag_3','Productio_lag_6','Productio_lag_12']:
            if lag not in lags:
                lags[lag] = 0
        
        # Rolling means from the same sequence
        prod_list = productions.tolist()
        roll3 = np.mean(prod_list[-3:]) if len(prod_list) >= 3 else np.mean(prod_list) if prod_list else 0
        roll6 = np.mean(prod_list[-6:]) if len(prod_list) >= 6 else np.mean(prod_list) if prod_list else 0
        roll12 = np.mean(prod_list[-12:]) if len(prod_list) >= 12 else np.mean(prod_list) if prod_list else 0
        rolling = {
            'Productio_rolling_mean_3': roll3,
            'Productio_rolling_mean_6': roll6,
            'Productio_rolling_mean_12': roll12
        }
        last_date = sub['date'].max()
        return lags, rolling, last_date
    
    def predict(self, year, month_name, crop_name, district_name, use_last_known=True):
        month_num = MONTH_MAP[month_name]
        month_sin = np.sin(2 * np.pi * month_num / 12)
        month_cos = np.cos(2 * np.pi * month_num / 12)
        
        # Encode crop and district
        try:
            crop_enc = self.le_crop.transform([crop_name])[0]
            dist_enc = self.le_dist.transform([district_name])[0]
        except ValueError as e:
            raise ValueError(f"Crop or district not seen in training: {e}")
        
        # Default values (all zeros)
        lag_values = {f'Productio_lag_{i}': 0 for i in [1,2,3,6,12]}
        rolling_values = {f'Productio_rolling_mean_{w}': 0 for w in [3,6,12]}
        harvested = 0
        yield_val = 0
        
        if use_last_known:
            last_lags, last_rollings, last_date = self.get_last_known_sequence(crop_name, district_name)
            if last_lags is not None:
                lag_values.update(last_lags)
                rolling_values.update(last_rollings)
                # Use average harvested area and yield for this crop-district
                sub = self.full_df[(self.full_df['Crop_nam'] == crop_name) & 
                                   (self.full_df['Location_district'] == district_name)]
                if not sub.empty:
                    harvested = sub['harvested'].mean()
                    yield_val = sub['yield'].mean()
            else:
                print(f"Warning: No historical data for {crop_name} in {district_name}. Using zeros.")
        
        # Build input dictionary in the exact order of feature_cols
        input_dict = {
            'year': year,
            'month_sin': month_sin,
            'month_cos': month_cos,
            **lag_values,
            **rolling_values,
            'harvested': harvested,
            'yield': yield_val,
            'Crop_enc': crop_enc,
            'Dist_enc': dist_enc
        }
        
        # Create DataFrame and ensure column order matches training
        input_df = pd.DataFrame([input_dict])[self.feature_cols]
        
        # Scale numeric columns (same as training)
        numeric_cols = ['year', 'month_sin', 'month_cos', 'harvested', 'yield']
        input_df[numeric_cols] = self.scaler.transform(input_df[numeric_cols])
        
        pred = self.model.predict(input_df)[0]
        return max(0, pred)  # production cannot be negative

if __name__ == "__main__":
    if len(sys.argv) != 5:
        print("Usage: python predict.py <year> <month> <crop> <district>")
        print("Example: python predict.py 2026 June Cabbage Badulla")
        sys.exit(1)
    
    year = int(sys.argv[1])
    month = sys.argv[2]
    crop = sys.argv[3]
    district = sys.argv[4]
    
    predictor = ProductionPredictor()
    pred = predictor.predict(year, month, crop, district, use_last_known=True)
    print(f"Predicted production for {crop} in {district}, {month} {year}: {pred:.2f}")