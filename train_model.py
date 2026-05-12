import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestRegressor

# Load dataset

df = pd.read_csv(r"C:\Users\sreem\OneDrive\Desktop\New folder (2)\real-estate-backend\train.csv")

# Target column

y = df['SalePrice']

# Features

X = df.drop(['SalePrice', 'Id'], axis=1)

# Numerical columns

numerical_cols = X.select_dtypes(
    include=['int64', 'float64']
).columns

# Categorical columns

categorical_cols = X.select_dtypes(
    include=['object']
).columns

# Numerical preprocessing

numerical_transformer = Pipeline([
    ('imputer', SimpleImputer(strategy='median'))
])

# Categorical preprocessing

categorical_transformer = Pipeline([
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

# Combine preprocessing

preprocessor = ColumnTransformer([
    ('num', numerical_transformer, numerical_cols),
    ('cat', categorical_transformer, categorical_cols)
])

# Final AI model

model = Pipeline([
    ('preprocessor', preprocessor),

    ('regressor', RandomForestRegressor(
        n_estimators=200,
        random_state=42
    ))
])

# Split dataset

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# Train model

print('Training AI Model...')

model.fit(X_train, y_train)

print('Training Completed')

# Save trained model

joblib.dump(model, 'house_price_model.pkl')

print('Model Saved Successfully')