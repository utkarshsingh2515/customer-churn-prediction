import streamlit as st
from src.predict import predict_churn

st.set_page_config(page_title="Customer Churn Prediction", layout="centered")

st.title("📊 Customer Churn Prediction")
st.write("Enter customer details to predict churn probability")

st.markdown("---")

credit_score = st.number_input("Credit Score", 300, 900, 650)
age = st.number_input("Age", 18, 100, 35)
tenure = st.number_input("Tenure (years)", 0, 10, 5)
balance = st.number_input("Account Balance", 0.0, value=50000.0)
num_products = st.selectbox("Number of Products", [1, 2, 3, 4])
has_card = st.selectbox("Has Credit Card?", ["Yes", "No"])
is_active = st.selectbox("Is Active Member?", ["Yes", "No"])
salary = st.number_input("Estimated Salary", 0.0, value=60000.0)
geography = st.selectbox("Geography", ["France", "Germany", "Spain"])
gender = st.selectbox("Gender", ["Female", "Male"])

# Encoding
has_card = 1 if has_card == "Yes" else 0
is_active = 1 if is_active == "Yes" else 0
gender_male = 1 if gender == "Male" else 0
geo_germany = 1 if geography == "Germany" else 0
geo_spain = 1 if geography == "Spain" else 0

features = [
    credit_score,
    age,
    tenure,
    balance,
    num_products,
    has_card,
    is_active,
    salary,
    geo_germany,
    geo_spain,
    gender_male
]

if st.button("Predict Churn"):
    prob = predict_churn(features)

    st.markdown("---")
    st.subheader("Prediction Result")
    st.write(f"Churn Probability: **{prob:.2f}**")

    if prob > 0.5:
        st.error("⚠️ Customer is likely to churn")
    else:
        st.success("✅ Customer is likely to stay")
