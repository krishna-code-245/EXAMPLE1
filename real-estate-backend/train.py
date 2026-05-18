import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score

# Load dataset
df = pd.read_csv("train.csv")

# Features
X = df[['OverallQual',
        'GrLivArea',
        'GarageCars',
        'TotalBsmtSF',
        'FullBath',
        'YearBuilt']]

# Target
y = df['SalePrice']

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# Create better AI model
model = RandomForestRegressor(
    n_estimators=200,
    random_state=42
)

# Train model
model.fit(X_train, y_train)

# Test accuracy
predictions = model.predict(X_test)

score = r2_score(y_test, predictions)

print(f"Model Accuracy: {score}")

# Save trained model
joblib.dump(model, 'house_price_model.pkl')

print("Better AI model saved successfully")