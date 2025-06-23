import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from lightgbm import LGBMClassifier
from sklearn.metrics import classification_report
import joblib
from flask import Flask, request, jsonify

# Load and preprocess data (mock data here)
data = pd.read_csv('ecommerce_data.csv')

# Sample features and targets
features = ['user_age', 'user_location', 'product_price', 'product_category', 'session_duration']
targets = ['click', 'cart', 'purchase']

# Encode categorical features if needed
data = pd.get_dummies(data, columns=['user_location', 'product_category'])

X = data.drop(columns=targets)
y_click = data['click']
y_cart = data['cart']
y_purchase = data['purchase']

# Split data
X_train, X_test, y_click_train, y_click_test = train_test_split(X, y_click, test_size=0.2, random_state=42)
_, _, y_cart_train, y_cart_test = train_test_split(X, y_cart, test_size=0.2, random_state=42)
_, _, y_purchase_train, y_purchase_test = train_test_split(X, y_purchase, test_size=0.2, random_state=42)

# Train models
click_model = LGBMClassifier().fit(X_train, y_click_train)
cart_model = LGBMClassifier().fit(X_train, y_cart_train)
purchase_model = LGBMClassifier().fit(X_train, y_purchase_train)

# Save models
joblib.dump(click_model, 'click_model.pkl')
joblib.dump(cart_model, 'cart_model.pkl')
joblib.dump(purchase_model, 'purchase_model.pkl')

# Test evaluation
print("CLICK PREDICTION:")
print(classification_report(y_click_test, click_model.predict(X_test)))

print("CART PREDICTION:")
print(classification_report(y_cart_test, cart_model.predict(X_test)))

print("PURCHASE PREDICTION:")
print(classification_report(y_purchase_test, purchase_model.predict(X_test)))

# Setup Flask API
app = Flask(__name__)

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    input_df = pd.DataFrame([data])
    input_df = pd.get_dummies(input_df)

    # Align with training columns
    for col in X.columns:
        if col not in input_df:
            input_df[col] = 0
    input_df = input_df[X.columns]  # Reorder columns

    click_model = joblib.load('click_model.pkl')
    cart_model = joblib.load('cart_model.pkl')
    purchase_model = joblib.load('purchase_model.pkl')

    predictions = {
        'click': int(click_model.predict(input_df)[0]),
        'cart': int(cart_model.predict(input_df)[0]),
        'purchase': int(purchase_model.predict(input_df)[0])
    }

    return jsonify(predictions)

if __name__ == '__main__':
    app.run(debug=True)
