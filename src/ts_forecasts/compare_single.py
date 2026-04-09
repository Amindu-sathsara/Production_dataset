import sys
from .train_evaluate import train_arima, train_sarima, train_prophet, forecast_model
from .utils import load_series, evaluate_forecast

def compare_for(crop, district, test_months=12):
    series = load_series(crop, district)
    if series is None or len(series) < 24:
        print("Not enough data")
        return
    train = series.iloc[:-test_months]
    test = series.iloc[-test_months:]
    train_dates = train.index
    
    models = [
        train_arima(train),
        train_sarima(train),
        train_prophet(train, train_dates)
    ]
    
    print(f"\nResults for {crop} in {district} (test months = {test_months}):")
    for model, mtype in models:
        if model is None:
            continue
        pred = forecast_model(model, mtype, test_months)
        if pred is not None:
            metrics = evaluate_forecast(test.values, pred)
            print(f"{mtype:8} MAE={metrics['MAE']:.2f} RMSE={metrics['RMSE']:.2f} MAPE={metrics['MAPE']:.1f}% MASE={metrics['MASE']:.2f}")

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python -m ts_forecast.compare_single <crop> <district>")
        sys.exit(1)
    crop = sys.argv[1]
    district = sys.argv[2]
    compare_for(crop, district)
    