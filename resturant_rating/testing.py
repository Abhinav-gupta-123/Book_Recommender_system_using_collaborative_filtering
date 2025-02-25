import joblib
import numpy as np
import pandas as pd

# Load the trained model and label encoders
model = joblib.load(r"C:\Users\abhin\Desktop\resturant_rating\trained_restaurant_rating_classifier.pkl")
label_encoders = joblib.load(r"C:\Users\abhin\Desktop\resturant_rating\label_encoders.pkl")

# Define user input keys based on features used in training
input_features = list(model.feature_names_in_)  # Get feature names from trained model

# Print expected features to debug mismatches
print("Expected Features:", input_features)

# Collect user input
user_input = {}

for feature in input_features:
    value = input(f"Enter value for {feature}: ").strip()
    user_input[feature] = value

# Manual mapping for binary features
binary_map = {"yes": "Yes", "no": "No", "NO": "No", "YES": "Yes"}
for feature in ["Has Table booking", "Has Online delivery"]:
    user_input[feature] = binary_map.get(user_input[feature], user_input[feature])

# Convert user input to DataFrame
input_df = pd.DataFrame([user_input])

# Apply label encoding to categorical features
for col in label_encoders:
    if user_input[col] in label_encoders[col].classes_:
        input_df[col] = label_encoders[col].transform([user_input[col]])
    else:
        print(f"⚠ Warning: '{user_input[col]}' is not recognized for {col}. Using default value.")
        input_df[col] = label_encoders[col].transform([label_encoders[col].classes_[0]])

# Ensure input features match model training order
input_df = input_df[input_features]

# Convert numerical values to appropriate types
input_df = input_df.astype(float)

# Predict the rating category
prediction = model.predict(input_df)[0]

# Map numeric category to rating level
rating_map = {0: "Low", 1: "Medium", 2: "High"}
predicted_rating = rating_map[prediction]

print(f"\n🔹 Predicted Restaurant Rating Category: {predicted_rating}")
