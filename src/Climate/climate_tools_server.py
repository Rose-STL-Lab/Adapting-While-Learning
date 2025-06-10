# climate_tools_server.py
from flask import Flask, request, jsonify
import sys
import os
import json
import numpy as np

# Add the necessary paths
src_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../"))
sys.path.append(src_dir)
sys.path.append(os.path.join(os.path.dirname(__file__), "tools/Climate_online"))

# Import the climate tools
from tools.emulators import *

app = Flask(__name__)

# Custom JSON encoder to handle NumPy types
class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NumpyEncoder, self).default(obj)

# Configure Flask to use our custom encoder
app.json_encoder = NumpyEncoder

# Helper function to convert NumPy types in dictionaries
def convert_numpy_types(obj):
    if isinstance(obj, dict):
        return {k: convert_numpy_types(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_types(item) for item in obj]
    elif isinstance(obj, (np.integer, np.int32, np.int64)):
        return int(obj)
    elif isinstance(obj, (np.floating, np.float32, np.float64)):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    else:
        return obj

@app.route('/location_summary', methods=['POST'])
def api_location_summary():
    data = request.json
    longitude = data.get('longitude')
    latitude = data.get('latitude')
    try:
        result, text = location_summary(float(longitude), float(latitude))
        # Convert numpy types to Python native types
        result = convert_numpy_types(result)
        return jsonify({"result": result, "text": text})
    except Exception as e:
        import traceback
        print(traceback.format_exc())
        return jsonify({"result": None, "text": f"Error: {str(e)}"})

@app.route('/history_temperature', methods=['POST'])
def api_history_temperature():
    data = request.json
    longitude = data.get('longitude')
    latitude = data.get('latitude')
    year = data.get('year')
    try:
        result, text = history_temperature(float(longitude), float(latitude), int(year))
        # Convert numpy types to Python native types
        if isinstance(result, (np.integer, np.floating, np.ndarray)):
            result = float(result)
        return jsonify({"result": result, "text": text})
    except Exception as e:
        import traceback
        print(traceback.format_exc())
        return jsonify({"result": None, "text": f"Error: {str(e)}"})

@app.route('/future_temperature', methods=['POST'])
def api_future_temperature():
    data = request.json
    longitude = data.get('longitude')
    latitude = data.get('latitude')
    setting = data.get('setting')
    year = data.get('year')
    try:
        result, text = future_temperature(float(longitude), float(latitude), setting, int(year))
        # Convert numpy types to Python native types
        if isinstance(result, (np.integer, np.floating, np.ndarray)):
            result = float(result)
        return jsonify({"result": result, "text": text})
    except Exception as e:
        import traceback
        print(traceback.format_exc())
        return jsonify({"result": None, "text": f"Error: {str(e)}"})

@app.route('/query_lat_and_lon', methods=['POST'])
def api_query_lat_and_lon():
    data = request.json
    city_name = data.get('city_name')
    try:
        result, text = query_lat_and_lon(city_name)
        return jsonify({"result": result, "text": text})
    except Exception as e:
        import traceback
        print(traceback.format_exc())
        return jsonify({"result": None, "text": f"Error: {str(e)}"})

@app.route('/diy_greenhouse', methods=['POST'])
def api_diy_greenhouse():
    data = request.json
    longitude = data.get('longitude')
    latitude = data.get('latitude')
    setting = data.get('setting')
    year = data.get('year')
    delta_CO2 = data.get('delta_CO2', 0)
    delta_CH4 = data.get('delta_CH4', 0)
    try:
        result, text = diy_greenhouse(
            float(longitude), 
            float(latitude), 
            setting, 
            int(year), 
            float(delta_CO2), 
            float(delta_CH4)
        )
        
        # Convert numpy types to Python native types
        if isinstance(result, (np.integer, np.floating, np.ndarray)):
            result = float(result)
        
        return jsonify({"result": result, "text": text})
    except Exception as e:
        import traceback
        print(traceback.format_exc())
        return jsonify({"result": None, "text": f"Error: {str(e)}"})

# Health check endpoint
@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({"status": "ok"})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)