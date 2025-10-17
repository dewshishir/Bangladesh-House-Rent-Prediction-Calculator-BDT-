import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
import joblib

# Set random seed for reproducibility
np.random.seed(42)
n = 1000

# Generate synthetic data
area = np.random.randint(300, 2500, size=n)
bhk = np.random.randint(1, 5, size=n)
bathroom = bhk
zilla = np.random.choice(["Dhaka", "Pabna", "Rajshahi", "Faridpur", "Cumilla"], size=n)
furnishing = np.random.choice(["Furnished", "Semi-Furnished", "Unfurnished"], size=n)

# Simulate rent in BDT
base_rent = area * 35 + bhk * 2000 + np.random.randint(1000, 8000, size=n)

# Create DataFrame
df = pd.DataFrame({
    "Area": area,
    "BHK": bhk,
    "Bathroom": bathroom,
    "District": zilla,
    "Furnishing": furnishing,
    "Rent": base_rent
})

# Encode categorical variables
df = pd.get_dummies(df, columns=["District", "Furnishing"], drop_first=True)

# Train model
X = df.drop("Rent", axis=1)
y = df["Rent"]
model = RandomForestRegressor()
model.fit(X, y)

# Save model and feature list
joblib.dump(model, "bd_house_rent_model.pkl")
joblib.dump(X.columns.tolist(), "bd_model_features.pkl")

print("✅ Model trained and saved for Bangladesh districts.")
