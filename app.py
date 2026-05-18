from flask import Flask, render_template, request
from flask_cors import CORS

import pandas as pd
import joblib

# Create Flask app
app = Flask(__name__)

# Allow frontend connection
CORS(app)

# Load trained AI model
model = joblib.load('house_price_model.pkl')

# Home Route
@app.route('/')
def home():
    return render_template('index.html')

# Prediction Route
@app.route('/predict', methods=['POST'])
def predict():

    try:

        # Get form data
        data = {
            'OverallQual': int(request.form['OverallQual']),
            'GrLivArea': int(request.form['GrLivArea']),
            'GarageCars': int(request.form['GarageCars']),
            'TotalBsmtSF': int(request.form['TotalBsmtSF']),
            'FullBath': int(request.form['FullBath']),
            'YearBuilt': int(request.form['YearBuilt'])
        }

        # Convert to DataFrame
        input_df = pd.DataFrame([data])

        # Prediction
        prediction = model.predict(input_df)

        predicted_price = round(prediction[0], 2)

        # Return result page
        return render_template(
            'index.html',
            prediction_text=f"Predicted House Price: ${predicted_price}"
        )

    except Exception as e:
        return f"Error: {e}"

# Run Flask server
if __name__ == '__main__':
    app.run(
        host='0.0.0.0',
        port=5000,
        debug=True
    )