import streamlit as st
import numpy as np
import pandas as pd
import joblib

# Load model and features
model = joblib.load("bd_house_rent_model.pkl")
model_features = joblib.load("bd_model_features.pkl")

# App settings
st.set_page_config(page_title="Bangladesh House Rent Calculator", layout="centered")
st.title("🏘️ Bangladesh House Rent Prediction Calculator (BDT)")

st.markdown("""
Enter the details of a property in Bangladesh to estimate its **monthly rent** in **Bangladeshi Taka (৳)**.
""")

# Input fields
area = st.number_input("📏 Total Area (sq ft)", min_value=100, max_value=5000, value=1000, step=50)
bhk = st.selectbox("🛏️ Number of Bedrooms (BHK)", [1, 2, 3, 4])
bathroom = st.selectbox("🚿 Number of Bathrooms", [1, 2, 3, 4])
district = st.selectbox("🏙️ District", ["Dhaka", "Pabna", "Rajshahi", "Faridpur", "Cumilla"])
furnishing = st.selectbox("🪑 Furnishing Status", ["Furnished", "Semi-Furnished", "Unfurnished"])

# Prepare input dictionary
input_dict = {
    "Area": area,
    "BHK": bhk,
    "Bathroom": bathroom,
    "District_Cumilla": 0,
    "District_Dhaka": 0,
    "District_Faridpur": 0,
    "District_Pabna": 0,
    "District_Rajshahi": 0,
    "Furnishing_Semi-Furnished": 0,
    "Furnishing_Unfurnished": 0
}

# Set one-hot encoding flags
district_col = f"District_{district}"
if district_col in input_dict:
    input_dict[district_col] = 1

if furnishing != "Furnished":  # Furnished is the base class
    furn_col = f"Furnishing_{furnishing}"
    input_dict[furn_col] = 1

# Convert to DataFrame
input_data = pd.DataFrame([input_dict], columns=model_features).fillna(0)

# Predict
if st.button("Predict Monthly Rent"):
    predicted_rent = model.predict(input_data)[0]
    st.success(f"Estimated Monthly Rent: ৳ {predicted_rent:,.0f}")
