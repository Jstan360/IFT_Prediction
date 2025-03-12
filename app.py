import numpy as np
from flask import Flask, request, render_template
import os
import pickle
import logging

# Set up logging for debugging
logging.basicConfig(level=logging.DEBUG)

app = Flask(__name__)

# Load the model at startup
model_path = os.path.join("models", "model.pk1")
try:
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    logging.info("Model loaded successfully: {}".format(type(model)))  # Debug: Check model type
except FileNotFoundError:
    logging.error(f"Model file not found at {model_path}")
    raise
except Exception as e:
    logging.error(f"Error loading model: {str(e)}")
    raise

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Define expected feature names (for reference only)
        feature_names = [
            'Critical Micelle Concentration (CMC) (M)',
            'Hydrophilic - Lipophilic Balance (HLB)',
            'Solubility Ratio (SR)',
            'Molecular Packing Parameter (MPP)',
            'Density (g/mL)',
            'Molecular Weight (g/mol)'
        ]

        # Convert form data to floats
        input_features = [float(x) for x in request.form.values()]
        logging.debug(f"Input features: {input_features}")  # Debug: Log inputs

        # Validate input length
        if len(input_features) != len(feature_names):
            raise ValueError(f"Expected {len(feature_names)} features, got {len(input_features)}")

        # Convert to 2D NumPy array for prediction
        features_array = np.array([input_features])
        logging.debug(f"Features array: {features_array}")  # Debug: Log array

        # Make prediction
        prediction = model.predict(features_array)
        output = round(prediction[0], 2)

        return render_template('index.html', prediction_text=f'The crude oil/brine IFT is {output} mN/m'.format(output))

    except ValueError as ve:
        logging.error(f"Input error: {str(ve)}")
        return render_template('index.html', prediction_text=f"Error: {str(ve)}")
    except Exception as e:
        logging.error(f"Prediction error: {str(e)}")
        return render_template('index.html', prediction_text=f"Error: {str(e)}")

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
