import pandas as pd
import numpy as np
import joblib
import sys
from datetime import datetime
from dateutil.relativedelta import relativedelta
from utils import MONTH_MAP, MONTH_LIST

class RecursiveProductionPredictor:
    def __init__(self, models_dir="../models", data_path="../data/processed/monthly_agricultural_data.csv"):
        self.model = joblib.load(f"{models_dir}/production_model.pkl")
        self.le_crop = joblib.load(f"{models_dir}/le_crop.pkl")
        self.le_dist = joblib.load(f"{models_dir}/le_dist.pkl")
        self.scaler = joblib.load(f"{models_dir}/scaler.pkl")
        with open(f"{models_dir}/feature_cols.txt", "r") as f:
            self.feature_cols = [line.strip() for line in f.readlines()]
        
        # Load historical data
        self.full_df = pd.read_csv(data_path)
        self.full_df['month_num'] = self.full_df['month'].map(MONTH_MAP)
        self.full_df['date'] = pd.to_datetime(
            self.full_df['year'].astype(str) + '-' + 
            self.full_df['month_num'].astype(str) + '-01'
        )
    
    def get_last_actual_series(self, crop, district, max_lag=12):
        """Return the last 'max_lag' actual production values as a list (oldest first)."""
        sub = self.full_df[(self.full_df['Crop_nam']==crop) & (self.full_df['Location_district']==district)]
        sub = sub.sort_values('date')
        if len(sub) == 0:
            return None
        productions = sub['Productio'].values[-max_lag:]
        return productions.tolist()
    
    def predict_single(self, year, month_num, crop_enc, dist_enc, lag_values, rolling_values, harvested, yield_val):
        """Predict one month given lag/rolling values."""
        month_sin = np.sin(2 * np.pi * month_num / 12)
        month_cos = np.cos(2 * np.pi * month_num / 12)
        
        input_dict = {
            'year': year,
            'month_sin': month_sin,
            'month_cos': month_cos,
            **{f'Productio_lag_{i}': lag_values.get(f'Productio_lag_{i}', 0) for i in [1,2,3,6,12]},
            **{f'Productio_rolling_mean_{w}': rolling_values.get(f'Productio_rolling_mean_{w}', 0) for w in [3,6,12]},
            'harvested': harvested,
            'yield': yield_val,
            'Crop_enc': crop_enc,
            'Dist_enc': dist_enc
        }
        input_df = pd.DataFrame([input_dict])[self.feature_cols]
        numeric_cols = ['year', 'month_sin', 'month_cos', 'harvested', 'yield']
        input_df[numeric_cols] = self.scaler.transform(input_df[numeric_cols])
        pred = self.model.predict(input_df)[0]
        return max(0, pred)
    
    def forecast_until(self, target_year, target_month_name, crop_name, district_name):
        """Recursively forecast from last known date up to target date."""
        # Encode crop and district
        crop_enc = self.le_crop.transform([crop_name])[0]
        dist_enc = self.le_dist.transform([district_name])[0]
        
        # Get last actual production series (up to 12 months)
        hist = self.get_last_actual_series(crop_name, district_name, max_lag=12)
        if hist is None:
            raise ValueError(f"No historical data for {crop_name} in {district_name}")
        
        # Determine last date in history
        sub = self.full_df[(self.full_df['Crop_nam']==crop_name) & (self.full_df['Location_district']==district_name)]
        last_date = sub['date'].max()
        last_year = last_date.year
        last_month = last_date.month
        
        # Build initial lag dictionary from hist (most recent = lag_1, etc.)
        # hist is oldest to newest? Need to reverse.
        hist_rev = hist[::-1]  # newest first
        lag_values = {}
        for i, val in enumerate(hist_rev):
            if i == 0:
                lag_values['Productio_lag_1'] = val
            elif i == 1:
                lag_values['Productio_lag_2'] = val
            elif i == 2:
                lag_values['Productio_lag_3'] = val
            elif i == 5:
                lag_values['Productio_lag_6'] = val
            elif i == 11:
                lag_values['Productio_lag_12'] = val
        # Fill missing
        for lag in [1,2,3,6,12]:
            key = f'Productio_lag_{lag}'
            if key not in lag_values:
                lag_values[key] = 0
        
        # Compute initial rolling means from hist
        prod_list = hist  # oldest to newest
        rolling_values = {
            'Productio_rolling_mean_3': np.mean(prod_list[-3:]) if len(prod_list)>=3 else np.mean(prod_list),
            'Productio_rolling_mean_6': np.mean(prod_list[-6:]) if len(prod_list)>=6 else np.mean(prod_list),
            'Productio_rolling_mean_12': np.mean(prod_list[-12:]) if len(prod_list)>=12 else np.mean(prod_list)
        }
        
        # Use average harvested and yield from history
        harvested = sub['harvested'].mean()
        yield_val = sub['yield'].mean()
        
        # Prepare to step forward month by month
        current_date = last_date
        target_month_num = MONTH_MAP[target_month_name]
        target_date = datetime(target_year, target_month_num, 1)
        
        # Store predictions (we need them to update lags)
        pred_series = []  # list of (date, pred)
        
        # If target is before or equal to last actual date, just return actual
        if target_date <= last_date:
            actual = sub[sub['date'] == target_date]['Productio'].values
            if len(actual) > 0:
                return actual[0]
            else:
                # fallback: nearest actual
                return hist[-1]
        
        # Recursive loop
        current = current_date
        while current < target_date:
            # Move to next month
            next_date = current + relativedelta(months=1)
            next_year = next_date.year
            next_month = next_date.month
            
            # Predict for next_date
            pred = self.predict_single(
                next_year, next_month, crop_enc, dist_enc,
                lag_values, rolling_values, harvested, yield_val
            )
            pred_series.append((next_date, pred))
            
            # Update lag_values with this prediction
            # Shift: old lag_1 becomes lag_2, new pred becomes lag_1
            new_lag = {}
            new_lag['Productio_lag_1'] = pred
            new_lag['Productio_lag_2'] = lag_values.get('Productio_lag_1', 0)
            new_lag['Productio_lag_3'] = lag_values.get('Productio_lag_2', 0)
            new_lag['Productio_lag_6'] = lag_values.get('Productio_lag_5', 0)  # we don't have lag_5, so approximate
            new_lag['Productio_lag_12'] = lag_values.get('Productio_lag_11', 0)
            # Simpler: just shift all lags (for simplicity, we only maintain lag_1,2,3,6,12 exactly)
            # Better: maintain a deque of last 12 predictions
            # For simplicity, we'll just update lag_1 and recompute rolling means from the extended series
            
            # Update rolling means: we need last 3,6,12 values including actuals + predictions
            # Let's rebuild a complete list of actuals + predictions so far
            all_vals = hist.copy() + [p for (_, p) in pred_series]
            rolling_values['Productio_rolling_mean_3'] = np.mean(all_vals[-3:]) if len(all_vals)>=3 else np.mean(all_vals)
            rolling_values['Productio_rolling_mean_6'] = np.mean(all_vals[-6:]) if len(all_vals)>=6 else np.mean(all_vals)
            rolling_values['Productio_rolling_mean_12'] = np.mean(all_vals[-12:]) if len(all_vals)>=12 else np.mean(all_vals)
            
            # Update lag_values with the new prediction (most recent)
            # Rebuild lag dict from all_vals (last 12)
            rev_vals = all_vals[::-1]
            lag_values = {}
            for i, val in enumerate(rev_vals):
                if i == 0:
                    lag_values['Productio_lag_1'] = val
                elif i == 1:
                    lag_values['Productio_lag_2'] = val
                elif i == 2:
                    lag_values['Productio_lag_3'] = val
                elif i == 5:
                    lag_values['Productio_lag_6'] = val
                elif i == 11:
                    lag_values['Productio_lag_12'] = val
            for lag in [1,2,3,6,12]:
                key = f'Productio_lag_{lag}'
                if key not in lag_values:
                    lag_values[key] = 0
            
            current = next_date
        
        # Return prediction for target date
        for date, pred in pred_series:
            if date == target_date:
                return pred
        return pred_series[-1][1] if pred_series else 0

if __name__ == "__main__":
    if len(sys.argv) != 5:
        print("Usage: python predict_recursive.py <year> <month> <crop> <district>")
        sys.exit(1)
    year = int(sys.argv[1])
    month = sys.argv[2]
    crop = sys.argv[3]
    district = sys.argv[4]
    
    predictor = RecursiveProductionPredictor()
    pred = predictor.forecast_until(year, month, crop, district)
    print(f"Predicted production for {crop} in {district}, {month} {year}: {pred:.2f}")