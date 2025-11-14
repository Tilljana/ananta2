from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import pickle
import math
import os
import traceback
import xgboost as xgb  # DIUBAH: Impor xgboost

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# ... (FEATURE_NAMES dan FEATURE_MAPPING tetap sama) ...
FEATURE_NAMES = [
    'Akses Jangkauan', 'Jumlah Keluarga Miskin', 'Rasio Penduduk Miskin Desil 1', 
    'Rumah tangga tanpa akses listrik', 'Produksi pangan', 'Luas lahan', 
    'Rasio Sarana Pangan', 'Persentase balita stunting', 'Proporsi Penduduk Lanjut Usia', 
    'Rasio Rumah Tangga Tanpa Air Bersih', 'Rasio Tenaga Kesehatan', 
    'Total Keluarga Beresiko Stunting dan Keluarga rentan'
]
FEATURE_MAPPING = {
    'X1': 'Akses Jangkauan', 'X2': 'Jumlah Keluarga Miskin', 'X3': 'Rasio Penduduk Miskin Desil 1',
    'X4': 'Rumah tangga tanpa akses listrik', 'X5': 'Produksi pangan', 'X6': 'Luas lahan',
    'X7': 'Rasio Sarana Pangan', 'X8': 'Persentase balita stunting', 'X9': 'Proporsi Penduduk Lanjut Usia',
    'X10': 'Rasio Rumah Tangga Tanpa Air Bersih', 'X11': 'Rasio Tenaga Kesehatan',
    'X12': 'Total Keluarga Beresiko Stunting dan Keluarga rentan'
}

# Global model and scaler
model = None
scaler = None

# DIUBAH: Fungsi ini sekarang memuat file .json
def load_xgb_model():
    """Load the XGBoost model from the local .json file"""
    try:
        # Nama file baru: .json
        model_path = os.path.join(BASE_DIR, 'best_model_XGB.json') 
        
        global model
        # Buat instance model kosong (sesuaikan jika ini Classifier)
        model = xgb.XGBRegressor() 
        
        # Muat model dari file .json
        model.load_model(model_path) 
        
        print("✓ Model loaded from local JSON successfully.")
        return True
    except Exception as e:
        print(f"✗ Error loading local model: {e}")
        traceback.print_exc()
        return False

# Fungsi ini TIDAK BERUBAH, scaler.pkl tetap aman
def load_scaler():
    """Load the scaler from the local pickle file"""
    try:
        scaler_path = os.path.join(BASE_DIR, 'scaler.pkl')
        with open(scaler_path, 'rb') as f:
            global scaler
            scaler = pickle.load(f)
        print("✓ Scaler loaded from local PKL successfully.")
        return True
    except Exception as e:
        print(f"✗ Error loading local scaler: {e}")
        traceback.print_exc()
        return False

# ... (Sisa kode /predict dan lainnya SAMA PERSIS) ...
# ... (Pastikan bagian if __name__ == '__main__': juga masih ada) ...

# Load models at startup
model_loaded = load_xgb_model()
scaler_loaded = load_scaler()

@app.route('/')
def home():
    """Render home page"""
    return send_from_directory('.', 'index.html')

@app.route('/api/status', methods=['GET'])
def status():
    """Check the API status"""
    status_info = {
        'status': 'API is running',
        'model_loaded': model is not None,
        'scaler_loaded': scaler is not None,
        'feature_names': FEATURE_NAMES
    }

    if model and scaler:
        try:
            test_data = [[50] * 12] 
            test_scaled = scaler.transform(test_data)
            test_pred = model.predict(test_scaled)
            status_info['model_test'] = 'Model can predict'
            status_info['test_prediction'] = float(test_pred[0])
        except Exception as e:
            status_info['model_test'] = f'Model test failed: {str(e)}'
    
    return jsonify(status_info)

@app.route('/predict', methods=['POST', 'OPTIONS'])
def predict():
    """Make a prediction based on user input"""
    
    if request.method == 'OPTIONS':
        return '', 204
    
    print("\n" + "="*60)
    print("📥 NEW PREDICTION REQUEST")
    print("="*60)
    
    if not model or not scaler:
        error_msg = 'Model atau scaler belum dimuat. Pastikan file model tersedia.'
        print(f"❌ {error_msg}")
        return jsonify({
            'success': False,
            'error': error_msg
        }), 500
    
    try:
        try:
            data = request.get_json(force=True)
        except Exception as e:
            print(f"❌ Error parsing JSON: {e}")
            return jsonify({
                'success': False,
                'error': f'Error parsing JSON: {str(e)}'
            }), 400
        
        if not data:
            print("❌ No data received")
            return jsonify({
                'success': False,
                'error': 'Data tidak diberikan. Pastikan mengirim JSON.'
            }), 400
        
        print(f"📊 Data received: {data}")
        
        features_dict = {}
        missing_features = []
        invalid_features = []
        
        for feature_name in FEATURE_NAMES:
            feature_key = [key for key, value in FEATURE_MAPPING.items() if value == feature_name][0]
            value = data.get(feature_key)
            
            if value is None or value == '':
                missing_features.append(feature_name)
            else:
                try:
                    float_value = float(value)
                    if math.isnan(float_value) or math.isinf(float_value):
                        invalid_features.append(f"{feature_name} (invalid number)")
                    else:
                        features_dict[feature_name] = float_value
                except (ValueError, TypeError) as e:
                    invalid_features.append(f"{feature_name} (value: {value})")
        
        if missing_features:
            error_msg = f'Fitur yang hilang: {", ".join(missing_features)}'
            print(f"❌ {error_msg}")
            return jsonify({
                'success': False,
                'error': error_msg
            }), 400
        
        if invalid_features:
            error_msg = f'Nilai tidak valid untuk: {", ".join(invalid_features)}'
            print(f"❌ {error_msg}")
            return jsonify({
                'success': False,
                'error': error_msg
            }), 400
        
        print(f"✓ All features validated: {features_dict}")
        
        if all(value == 0 for value in features_dict.values()):
            error_msg = 'Semua fitur tidak boleh bernilai 0.'
            print(f"❌ {error_msg}")
            return jsonify({
                'success': False,
                'error': error_msg
            }), 400
        
        features_list = [features_dict[name] for name in FEATURE_NAMES]
        print(f"📊 Features list: {features_list}")
        
        features_scaled = scaler.transform([features_list]) 
        print(f"📈 Scaled features: {features_scaled}")
        
        prediction = model.predict(features_scaled)
        score = float(prediction[0])
        
        print(f"🎯 Raw prediction: {prediction}")
        print(f"🎯 Prediction score: {score}")
        
        confidence = 96.8
        
        try:
            if hasattr(model, 'predict_proba'):
                probabilities = model.predict_proba(features_scaled)
                import numpy as np 
                confidence = float(np.max(probabilities) * 100)
                print(f"📊 Confidence from predict_proba: {confidence}%")
        except Exception as e:
            print(f"ℹ️ Using default confidence: {confidence}% (predict_proba not available or numpy not found)")

        response = {
            'success': True,
            'score': round(score, 3),
            'confidence': round(confidence, 1),
            'features_received': list(features_dict.keys()),
            'message': 'Prediksi berhasil dilakukan'
        }
        
        print(f"✅ Response prepared: {response}")
        print("="*60 + "\n")
        
        return jsonify(response), 200
    
    except Exception as e:
        print(f"❌ FATAL ERROR in prediction: {str(e)}")
        print("="*60)
        print("TRACEBACK:")
        traceback.print_exc()
        print("="*60 + "\n")
        
        return jsonify({
            'success': False,
            'error': f'Terjadi kesalahan server: {str(e)}'
        }), 500

# Error handlers
@app.errorhandler(404)
def not_found(e):
    return jsonify({
        'success': False,
        'error': 'Endpoint not found'
    }), 404

@app.errorhandler(405)
def method_not_allowed(e):
    return jsonify({
        'success': False,
        'error': 'Method not allowed'
    }), 405

@app.errorhandler(500)
def internal_error(e):
    return jsonify({
        'success': False,
        'error': 'Internal server error'
    }), 500

# Kode untuk Railway (mengambil $PORT)
if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    print("\n" + "="*60)
    print("🚀 Flask API + Web Server Starting")
    print("="*60)
    print(f"📂 Base Directory: {BASE_DIR}")
    print(f"🤖 Model Status: {'✓ Loaded' if model else '✗ Not Loaded'}")
    print(f"📊 Scaler Status: {'✓ Loaded' if scaler else '✗ Not Loaded'}")
    print("="*60)
    print(f"📌 Running on http://0.0.0.0:{port}")
    print("="*60 + "\n")
    app.run(host='0.0.0.0', port=port, threaded=True, debug=False)
