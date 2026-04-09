import sys
import os
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from utils import load_series, MONTH_MAP
import joblib

class TSForecaster:
    def __init__(self, model_dir=None, data_path=None):
        # Resolve project root relative to this file so paths do not depend on cwd
        this_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.dirname(os.path.dirname(this_dir))  # .../production
        self.model_dir = model_dir or os.path.join(project_root, "models", "ts_models")
        self.data_path = data_path or os.path.join(project_root, "data", "processed", "monthly_agricultural_data.csv")
        self.cache = {}
    
    def get_model(self, crop, district):
        key = f"{crop}_{district}"
        if key in self.cache:
            return self.cache[key]
        model_file = f"{self.model_dir}/{crop}_{district}.pkl"
        type_file = f"{self.model_dir}/{crop}_{district}_type.txt"
        if not os.path.exists(model_file) or not os.path.exists(type_file):
            print(f"No pre-trained model for {crop} in {district}. Run train_evaluate.py first.")
            return None, None
        with open(type_file, 'r') as f:
            model_type = f.read().strip()
        model = joblib.load(model_file)
        self.cache[key] = (model, model_type)
        return model, model_type
    
    def forecast(self, year, month_name, crop, district):
        model, model_type = self.get_model(crop, district)
        if model is None:
            return None
        series = load_series(crop, district, self.data_path)
        if series is None:
            return None
        last_date = series.index[-1]
        target_date = datetime(year, MONTH_MAP[month_name], 1)
        if target_date <= last_date:
            if target_date in series.index:
                return series.loc[target_date]
            else:
                prev = series[series.index <= target_date]
                if not prev.empty:
                    return prev.iloc[-1]
                return None
        steps = (target_date.year - last_date.year) * 12 + (target_date.month - last_date.month)
        if steps <= 0:
            return None
        if model_type in ['ARIMA', 'SARIMA']:
            forecast = model.predict(n_periods=steps)
            return forecast.iloc[-1]
        elif model_type == 'Prophet':
            future = model.make_future_dataframe(periods=steps, freq='MS')
            forecast = model.predict(future)
            return forecast['yhat'].iloc[-1]
        else:
            return None

if __name__ == "__main__":
    if len(sys.argv) != 5:
        print("Usage: python predict.py <year> <month> <crop> <district>")
        sys.exit(1)
    year = int(sys.argv[1])
    month = sys.argv[2]
    crop = sys.argv[3]
    district = sys.argv[4]
    forecaster = TSForecaster()
    pred = forecaster.forecast(year, month, crop, district)
    if pred is None:
        print("Forecast failed.")
    else:
        print(f"Forecasted production for {crop} in {district}, {month} {year}: {pred:.2f} MT") 