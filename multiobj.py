import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from lightgbm import LGBMClassifier
from sklearn.metrics import classification_report
import joblib
import streamlit as st

st.set_page_config(page_title="E-commerce Multi-Objective Recommender")
st.title("🛒 E-commerce Multi-Objective Recommender")

# Sample/mock data
@st.cache_data
def load_data():
    return pd.DataFrame({
        'user_age': [25, 30, 22, 40, 35],
        'user_location': ['Delhi', 'Mumbai', 'Delhi', 'Bangalore', 'Chennai'],
        'product_price': [299, 999, 159, 499, 799],
        'product_category': ['Electronics', 'Clothing', 'Books', 'Electronics', 'Fashion'],
        'session_duration': [180, 300, 120, 240, 90],
        'click': [1, 1, 0, 1, 0],
        'cart': [0, 1, 0, 1, 1],
        'purchase': [0, 0, 0, 1, 1]
    })

data = load_data()

features = ['user_age', 'user_location', 'product_price', 'product_category', 'session_duration']
targets = ['click', 'cart', 'purchase']

# Preprocessing
encoded_data = pd.get_dummies(data, columns=['user_location', 'product_category'])
X = encoded_data.drop(columns=targets)
y_click = encoded_data['click']
y_cart = encoded_data['cart']
y_purchase = encoded_data['purchase']

# Split
X_train, X_test, y_click_train, y_click_test = train_test_split(X, y_click, test_size=0.2, random_state=42)
_, _, y_cart_train, y_cart_test = train_test_split(X, y_cart, test_size=0.2, random_state=42)
_, _, y_purchase_train, y_purchase_test = train_test_split(X, y_purchase, test_size=0.2, random_state=42)

# Train or Load Models
@st.cache_resource
def train_models():
    click_model = LGBMClassifier().fit(X_train, y_click_train)
    cart_model = LGBMClassifier().fit(X_train, y_cart_train)
    purchase_model = LGBMClassifier().fit(X_train, y_purchase_train)
    return click_model, cart_model, purchase_model

click_model, cart_model, purchase_model = train_models()

# Input UI
st.subheader("Enter User & Product Details")

user_age = st.slider("User Age", 18, 65, 25)
user_location = st.selectbox("User Location", data['user_location'].unique())
product_price = st.number_input("Product Price", value=300)
product_category = st.selectbox("Product Category", data['product_category'].unique())
session_duration = st.slider("Session Duration (seconds)", 30, 600, 180)

if st.button("Predict Actions"):
    input_df = pd.DataFrame([{
        'user_age': user_age,
        'user_location': user_location,
        'product_price': product_price,
        'product_category': product_category,
        'session_duration': session_duration
    }])

    input_df = pd.get_dummies(input_df)
    for col in X.columns:
        if col not in input_df:
            input_df[col] = 0
    input_df = input_df[X.columns]

    click_pred = click_model.predict(input_df)[0]
    cart_pred = cart_model.predict(input_df)[0]
    purchase_pred = purchase_model.predict(input_df)[0]

    st.success("Prediction Results:")
    st.write(f"🖱️ Click Likelihood: {'Yes' if click_pred else 'No'}")
    st.write(f"🛒 Cart Addition Likelihood: {'Yes' if cart_pred else 'No'}")
    st.write(f"💳 Purchase Likelihood: {'Yes' if purchase_pred else 'No'}")

