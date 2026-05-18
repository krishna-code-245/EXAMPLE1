from flask import Flask, render_template, request
from flask_cors import CORS
import pandas as pd
import joblib

# Create Flask App
app = Flask(__name__)

# Enable CORS
CORS(app)

# Load AI Model
model = joblib.load('house_price_model.pkl')

# Home Page
@app.route('/')
def home():
    return render_template('index.html', prediction_text="")

# Prediction Route
@app.route('/predict', methods=['POST'])
def predict():

    try:

        # Get user inputs
        overall_qual = int(request.form['OverallQual'])
        gr_liv_area = int(request.form['GrLivArea'])
        garage_cars = int(request.form['GarageCars'])
        total_bsmt_sf = int(request.form['TotalBsmtSF'])
        full_bath = int(request.form['FullBath'])
        year_built = int(request.form['YearBuilt'])

        # Create DataFrame
        input_data = pd.DataFrame([{
            'OverallQual': overall_qual,
            'GrLivArea': gr_liv_area,
            'GarageCars': garage_cars,
            'TotalBsmtSF': total_bsmt_sf,
            'FullBath': full_bath,
            'YearBuilt': year_built
        }])

        # Make Prediction
        prediction = model.predict(input_data)

        predicted_price = round(prediction[0], 2)

        # Format Prediction
        result = f"🏡 Estimated House Price: ${predicted_price}"

        # Return HTML page with result
        return render_template(
            'index.html',
            prediction_text=result
        )

    except Exception as e:

        return render_template(
            'index.html',
            prediction_text=f"❌ Error: {e}"
        )

# Run Flask App
if __name__ == '__main__':
    app.run(
        host='0.0.0.0',
        port=5000,
        debug=True
    )