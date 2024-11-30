import streamlit as st
import pandas as pd
import joblib

# Title
st.title("Cross-sell Prediction App")

# File check
try:
    df = pd.read_csv('train.csv')
    model = joblib.load('promote_pipeline_model.pkl')
except FileNotFoundError as e:
    st.error(f"Error loading files: {e}")
    st.stop()

# Input Elements
gender = st.selectbox("Gender", df['Gender'].unique())
age = st.selectbox("Age", sorted(df['Age'].unique()))
driving_license = st.selectbox("Driving License", df['Driving_License'].unique())
region_code = st.selectbox("Region Code", sorted(df['Region_Code'].unique()))
previously_insured = st.selectbox("Previously Insured", df['Previously_Insured'].unique())
vehicle_damage = st.selectbox("Vehicle Damage", df['Vehicle_Damage'].unique())
annual_premium = st.number_input("Annual Premium", min_value=0.0, step=0.1)
policy_sales_channel = st.number_input("Policy Sales Channel", min_value=0, step=1)
vintage = st.number_input("Vintage", min_value=0, step=1)

# Add input for Vehicle Age
vehicle_age = st.selectbox("Vehicle Age", df['Vehicle_Age'].unique())

# Update the input dictionary
inputs = {
    'Gender': gender,
    'Age': age,
    'Driving_License': driving_license,
    'Region_Code': region_code,
    'Previously_Insured': previously_insured,
    'Vehicle_Age': vehicle_age,  # Added Vehicle_Age
    'Vehicle_Damage': vehicle_damage,
    'Annual_Premium': annual_premium,
    'Policy_Sales_Channel': policy_sales_channel,
    'Vintage': vintage,
}

# Create DataFrame from inputs
X_input = pd.DataFrame([inputs])

# Make prediction
if st.button('Predict'):
    prediction = model.predict(X_input)
    st.write("The predicted value is:")
    st.write(prediction)


# File Upload Prediction
uploaded_file = st.file_uploader("Upload a CSV file", type='csv')
if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.write(df.head())
    
    if st.button("Batch Predict"):
        df['is_promoted'] = model.predict(df)
        st.write(df.head())
        st.download_button("Download Predictions", data=df.to_csv(index=False), file_name="predictions.csv", mime="text/csv")
