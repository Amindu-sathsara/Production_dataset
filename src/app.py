from flask import Flask, request, jsonify
from predict import ProductionPredictor

app = Flask(__name__)
predictor = ProductionPredictor()

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    required = ['year', 'month', 'crop', 'district']
    if not all(k in data for k in required):
        return jsonify({'error': 'Missing fields. Required: year, month, crop, district'}), 400
    
    year = int(data['year'])
    month = data['month']
    crop = data['crop']
    district = data['district']
    
    # Optional historical lags (for advanced users)
    lag_values = data.get('lag_values', {})
    rolling_values = data.get('rolling_values', {})
    
    try:
        pred = predictor.predict(year, month, crop, district, lag_values, rolling_values)
        return jsonify({
            'predicted_production': round(pred, 2),
            'year': year,
            'month': month,
            'crop': crop,
            'district': district
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/health', methods=['GET'])
def health():
    return jsonify({'status': 'ok'})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=False)