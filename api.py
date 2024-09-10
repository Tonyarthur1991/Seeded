import os
from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_score
import logging

app = Flask(__name__)
CORS(app)  # This allows cross-origin requests, necessary for API calls from a static frontend

# Set up logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# Global variables to hold our models and transformers
model_coverage = None
model_number = None
preprocessor = None
data_means = None
data_stds = None
config_categories = None

def initialize_models():
    global model_coverage, model_number, preprocessor, data_means, data_stds, config_categories
    try:
        logger.info("Starting model initialization...")
        
        # Load the data
        data = pd.read_csv('Model_data.csv')
        logger.info(f"Data loaded. Shape: {data.shape}")
        logger.info(f"Columns: {data.columns}")

        # Convert Screw_Configuration to categorical
        data['Screw_Configuration'] = pd.Categorical(data['Screw_Configuration'])

        # Calculate means and standard deviations for normalization
        data_means = data[['Screw_speed', 'Liquid_content', 'Liquid_binder']].mean()
        data_stds = data[['Screw_speed', 'Liquid_content', 'Liquid_binder']].std()
        config_categories = data['Screw_Configuration'].cat.categories.tolist()

        # Define preprocessor
        numeric_features = ['Screw_speed', 'Liquid_content', 'Liquid_binder']
        categorical_features = ['Screw_Configuration']

        preprocessor = ColumnTransformer(
            transformers=[
                ('num', StandardScaler(), numeric_features),
                ('cat', OneHotEncoder(drop='first', sparse_output=False), categorical_features)
            ])

        # Prepare the data
        X = data[numeric_features + categorical_features]
        y_coverage = data['Seed_coverage']
        y_number = data['number_seeded']

        # Create pipelines
        model_coverage = Pipeline([
            ('preprocessor', preprocessor),
            ('regressor', LinearRegression())
        ])

        model_number = Pipeline([
            ('preprocessor', preprocessor),
            ('regressor', LinearRegression())
        ])

        # Fit models
        model_coverage.fit(X, y_coverage)
        model_number.fit(X, y_number)

        logger.info("Models fitted successfully")

        # Evaluate models
        coverage_scores = cross_val_score(model_coverage, X, y_coverage, cv=5)
        number_scores = cross_val_score(model_number, X, y_number, cv=5)

        logger.info(f"Coverage Model CV Scores: {coverage_scores}")
        logger.info(f"Number Model CV Scores: {number_scores}")

    except Exception as e:
        logger.error(f"Error during initialization: {str(e)}")
        raise

@app.route('/predict', methods=['POST'])
def predict():
    try:
        logger.info("Received prediction request")
        data = request.json
        logger.info(f"Input data: {data}")

        liquid_content = data['liquid_content']
        liquid_binder = data['liquid_binder']
        screw_speed = data['screw_speed']
        screw_config = data['screw_config']

        # Input validation
        if not 0 <= liquid_content <= 1:
            return jsonify({'error': 'Liquid content must be between 0 and 1'}), 400
        if not 0 <= liquid_binder <= 100:
            return jsonify({'error': 'Liquid binder must be between 0 and 100'}), 400
        if not 0 <= screw_speed <= 1000:
            return jsonify({'error': 'Invalid screw speed'}), 400
        if screw_config not in config_categories:
            return jsonify({'error': 'Invalid screw configuration'}), 400

        # Create input data for the model
        new_data = pd.DataFrame({
            'Screw_speed': [screw_speed],
            'Liquid_content': [liquid_content],
            'Liquid_binder': [liquid_binder],
            'Screw_Configuration': [screw_config]
        })

        # Make predictions
        predicted_coverage = model_coverage.predict(new_data)[0]
        predicted_number = model_number.predict(new_data)[0]

        # Ensure predictions are within reasonable ranges
        predicted_coverage = max(0, min(100, predicted_coverage))
        predicted_number = max(0, predicted_number)

        logger.info(f"Predictions - Coverage: {predicted_coverage}, Number: {predicted_number}")

        # Calculate probability
        expected_seeds = np.interp(screw_speed, [100, 300, 500], [18, 20, 13])
        prob_number = (predicted_number >= 0.85*expected_seeds) and (predicted_number <= 1.15*expected_seeds)
        prob_coverage = predicted_coverage >= 40
        probability = (float(prob_number) + float(prob_coverage)) / 2

        logger.info(f"Calculated probability: {probability}")

        return jsonify({
            'predicted_coverage': predicted_coverage,
            'predicted_number': predicted_number,
            'probability': probability
        })

    except Exception as e:
        logger.error(f"Error during prediction: {str(e)}")
        return jsonify({'error': 'An error occurred during prediction. Please check the server logs for details.'}), 500

# Initialize models when the app starts
initialize_models()

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)