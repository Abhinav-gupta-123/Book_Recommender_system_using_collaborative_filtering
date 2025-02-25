import streamlit as st
import joblib
import pandas as pd

# Load trained model and label encoders
model = joblib.load(r"C:\Users\abhin\Desktop\resturant_rating\trained_restaurant_rating_classifier.pkl")
label_encoders = joblib.load(r"C:\Users\abhin\Desktop\resturant_rating\label_encoders.pkl")

# Get expected feature names from the trained model
input_features = list(model.feature_names_in_)

# Streamlit UI
st.title("🍽️ Restaurant Rating Prediction")
st.write("Enter restaurant details to predict its rating category.")

# User Inputs
has_table_booking = st.radio("Has Table Booking?", ["Yes", "No"])
has_online_delivery = st.radio("Has Online Delivery?", ["Yes", "No"])
average_cost = st.number_input("Average Cost for Two", min_value=0, step=50)
cuisines = st.text_input("Enter Cuisines (comma-separated)")
votes = st.number_input("Number of Votes", min_value=0, step=1)
price_range = st.slider("Price Range", min_value=1, max_value=4)

# Prepare user input dictionary
user_input = {
    "Has Table booking": has_table_booking,
    "Has Online delivery": has_online_delivery,
    "Average Cost for two": average_cost,
    "Cuisines": cuisines,
    "Votes": votes,
    "Price range": price_range,
}

# Convert user input to DataFrame
input_df = pd.DataFrame([user_input])

# Apply label encoding to categorical features
for col in label_encoders:
    if user_input[col] in label_encoders[col].classes_:
        input_df[col] = label_encoders[col].transform([user_input[col]])
    else:
        st.warning(f"⚠ '{user_input[col]}' is not recognized for {col}. Using default value.")
        input_df[col] = label_encoders[col].transform([label_encoders[col].classes_[0]])

# Ensure input features match model training order
input_df = input_df[input_features]

# Convert numerical values to appropriate types
input_df = input_df.astype(float)

# Predict the rating category
if st.button("Predict Rating"):
    prediction = model.predict(input_df)[0]

    # Mapping numeric category to rating level
    rating_map = {0: "Low", 1: "Medium", 2: "High"}
    predicted_rating = rating_map[prediction]

    st.success(f"✅ Predicted Restaurant Rating: **{predicted_rating}**")
