from flask import Flask, request, jsonify, render_template
import google.generativeai as genai
from flask_cors import CORS
import os
import traceback
from datetime import datetime
import joblib
import numpy as np
import json

app = Flask(__name__)
CORS(app)

# Configure Gemini AI with API key
API_KEY = "AIzaSyDqgRQe8nu1lJry7NI0MgF21WSdRSOLEmw"
genai.configure(api_key=API_KEY)

# Load crop recommendation model
try:
    crop_model = joblib.load("models/crop_recommender.pkl")
    print("Crop recommendation model loaded successfully!")
except Exception as e:
    print(f"Error loading crop model: {str(e)}")
    crop_model = None

# Multi-language support dictionary
LANGUAGES = {
    "English": {
        "land_types": ["Clay Soil", "Sandy Soil", "Loamy Soil", "Silt Soil", "Black Soil"],
        "seasons": ["Kharif (Monsoon)", "Rabi (Winter)", "Zaid (Summer)"],
        "crops": ["Rice", "Wheat", "Cotton", "Sugarcane", "Pulses", "Vegetables", "Fruits", "Oil Seeds"]
    },
    "Hindi": {
        "land_types": ["मिट्टी की मिट्टी", "रेतीली मिट्टी", "दोमट मिट्टी", "गाद मिट्टी", "काली मिट्टी"],
        "seasons": ["खरीफ (मानसून)", "रबी (सर्दी)", "जायद (गर्मी)"],
        "crops": ["चावल", "गेहूं", "कपास", "गन्ना", "दालें", "सब्जियां", "फल", "तिलहन"]
    },
    "Telugu": {
        "land_types": ["బంక మట్టి", "ఇసుక నేల", "లోమి నేల", "బురద నేల", "నల్ల నేల"],
        "seasons": ["ఖరీఫ్ (వర్షాకాలం)", "రబీ (శీతాకాలం)", "జైద్ (వేసవి)"],
        "crops": ["వరి", "గోధుమ", "పత్తి", "చెరకు", "పప్పు ధాన్యాలు", "కూరగాయలు", "పండ్లు", "నూనె గింజలు"]
    }
}

# Create directories if they don't exist
os.makedirs('solutions', exist_ok=True)
os.makedirs('user_locations', exist_ok=True)
LOCATIONS_FILE = 'user_locations/locations.json'

def load_locations():
    """Load locations from JSON file with error handling"""
    try:
        if os.path.exists(LOCATIONS_FILE):
            with open(LOCATIONS_FILE, 'r') as f:
                return json.load(f)
        return []
    except Exception as e:
        print(f"Error loading locations: {str(e)}")
        return []

def save_locations(locations):
    """Save locations to JSON file with error handling"""
    try:
        with open(LOCATIONS_FILE, 'w') as f:
            json.dump(locations, f, indent=2)
        return True
    except Exception as e:
        print(f"Error saving locations: {str(e)}")
        return False

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/chatbot')
def chatbot():
    return render_template('chatbot.html')

@app.route('/crop-recommendation')
def crop_recommendation():
    return render_template('crop_recommend.html')

@app.route('/location-picker')
def location_picker():
    return render_template('location_picker.html')

@app.route('/api/get_options/<language>')
def get_options(language):
    if language in LANGUAGES:
        return jsonify(LANGUAGES[language])
    return jsonify({"error": "Language not supported"}), 400

@app.route('/api/generate_solution', methods=['POST'])
def generate_solution():
    try:
        if not request.is_json:
            return jsonify({"error": "Request must be JSON"}), 400
        
        data = request.json
        
        required_fields = ['land_type', 'season', 'crop_type', 'acres', 'problem']
        for field in required_fields:
            if field not in data:
                return jsonify({"error": f"Missing required field: {field}"}), 400
        
        language = data.get('language', 'English')
        model_name = 'models/gemini-1.5-pro'
        model = genai.GenerativeModel(model_name)
        
        prompt = f"""
        As an agricultural expert, provide a detailed solution in {language} for the following farming situation:
        
        Land Type: {data['land_type']}
        Season: {data['season']}
        Crop Type: {data['crop_type']}
        Land Area: {data['acres']} acres
        Problem Description: {data['problem']}
        
        Please provide:
        1. Problem analysis
        2. Recommended solutions
        3. Preventive measures for the future
        4. Additional tips specific to the land type, crop, and season
        """
        
        response = model.generate_content(prompt)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"solutions/farm_solution_{timestamp}.txt"
        
        with open(filename, "w", encoding="utf-8") as f:
            f.write("FARM PROBLEM DETAILS\n")
            f.write("-------------------\n\n")
            for key, value in data.items():
                f.write(f"{key.title()}: {value}\n")
            f.write("\nRECOMMENDED SOLUTION\n")
            f.write("-------------------\n\n")
            f.write(response.text)
        
        return jsonify({
            "solution": response.text,
            "filename": filename
        })
        
    except Exception as e:
        error_details = traceback.format_exc()
        print(f"Error in generate_solution: {str(e)}")
        return jsonify({"error": str(e), "traceback": error_details}), 500

@app.route('/api/recommend-crop', methods=['POST'])
def recommend_crop():
    try:
        if not request.is_json:
            return jsonify({"error": "Request must be JSON"}), 400
        
        data = request.json
        
        required_features = ['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall']
        for feature in required_features:
            if feature not in data:
                return jsonify({"error": f"Missing required feature: {feature}"}), 400

        # Handle location if provided
        location = data.get('location')
        if location:
            save_user_location(location)
        
        input_data = np.array([
            data['N'],
            data['P'],
            data['K'],
            data['temperature'],
            data['humidity'],
            data['ph'],
            data['rainfall']
        ]).reshape(1, -1)
        
        prediction = crop_model.predict(input_data)[0]
        probabilities = crop_model.predict_proba(input_data)[0]
        confidence = round(np.max(probabilities) * 100, 2)
        
        # Get top 3 crops
        top3_idx = np.argsort(probabilities)[-3:][::-1]
        top3_crops = crop_model.classes_[top3_idx]
        top3_conf = [round(probabilities[i]*100, 2) for i in top3_idx]
        
        return jsonify({
            "recommended_crop": prediction,
            "confidence": confidence,
            "top_recommendations": [
                {"crop": crop, "confidence": conf} 
                for crop, conf in zip(top3_crops, top3_conf)
            ]
        })
        
    except Exception as e:
        error_details = traceback.format_exc()
        print(f"Error in recommend_crop: {str(e)}")
        return jsonify({"error": str(e), "traceback": error_details}), 500

@app.route('/api/process-location', methods=['POST'])
def process_location():
    try:
        if not request.is_json:
            return jsonify({"error": "Request must be JSON"}), 400
        
        data = request.json
        
        # Validate required fields
        if not data or not data.get('location') or not data['location'].get('coordinates'):
            return jsonify({
                "status": "error",
                "error": "Missing required location data"
            }), 400
        
        # Load existing locations
        locations = load_locations()
        
        # Add new entry with server-side timestamp
        locations.append({
            "ip": request.remote_addr,
            "timestamp": datetime.now().isoformat(),
            "location": {
                "address": data['location'].get('address', "Unknown address"),
                "coordinates": data['location']['coordinates']
            }
        })
        
        # Save back to file
        if save_locations(locations):
            return jsonify({
                "status": "success",
                "message": "Location processed"
            })
        else:
            return jsonify({
                "status": "error",
                "error": "Failed to save location"
            }), 500
            
    except Exception as e:
        return jsonify({
            "status": "error",
            "error": str(e)
        }), 500

@app.route('/api/get-locations', methods=['GET'])
def get_locations():
    """Get all valid locations (filter out null entries)"""
    try:
        locations = load_locations()
        # Filter out invalid entries
        valid_locations = [
            loc for loc in locations 
            if loc.get('location') and loc['location'].get('coordinates')
        ]
        return jsonify({
            "status": "success",
            "locations": valid_locations,
            "count": len(valid_locations)
        })
    except Exception as e:
        return jsonify({
            "status": "error",
            "error": str(e),
            "locations": [],
            "count": 0
        }), 500

@app.route('/api/replace-locations', methods=['POST'])
def replace_locations():
    """Replace all locations (used for cleanup)"""
    try:
        new_data = request.json.get('locations', [])
        
        # Validate all entries
        valid_locations = [
            loc for loc in new_data
            if loc.get('location') and loc['location'].get('coordinates')
        ]
        
        if save_locations(valid_locations):
            return jsonify({
                "status": "success",
                "count": len(valid_locations)
            })
        else:
            return jsonify({
                "status": "error",
                "error": "Failed to save locations"
            }), 500
            
    except Exception as e:
        return jsonify({
            "status": "error",
            "error": str(e)
        }), 500

@app.route('/api/clear-locations', methods=['POST'])
def clear_locations():
    """Clear all locations"""
    try:
        if save_locations([]):
            return jsonify({"status": "success"})
        else:
            return jsonify({
                "status": "error",
                "error": "Failed to clear locations"
            }), 500
    except Exception as e:
        return jsonify({
            "status": "error",
            "error": str(e)
        }), 500

def save_user_location(location_data):
    """Legacy function for crop recommendation endpoint"""
    try:
        locations = load_locations()
        locations.append({
            "ip": request.remote_addr,
            "timestamp": datetime.now().isoformat(),
            "location": {
                "address": location_data.get('address', "Unknown address"),
                "coordinates": location_data.get('coordinates', {})
            }
        })
        save_locations(locations)
    except Exception as e:
        print(f"Error saving location: {str(e)}")

if __name__ == '__main__':
    app.run(debug=True)