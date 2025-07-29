import streamlit as st
import pandas as pd
import joblib

# Load model and encoders
model = joblib.load("bank_account_model.pkl")
label_encoders = joblib.load("label_encoders.pkl")

st.set_page_config(page_title="Financial Inclusion Predictor")
st.title("🌍 Financial Inclusion in Africa")
st.subheader("Will this person likely have a bank account?")

# User input form
with st.form("prediction_form"):
    st.write("### 📥 Input Features")

    country = st.selectbox("Country", label_encoders["country"].classes_)
    year = st.selectbox("Year", [2016, 2017, 2018])
    location_type = st.selectbox("Location Type", label_encoders["location_type"].classes_)
    cellphone_access = st.selectbox("Cellphone Access", label_encoders["cellphone_access"].classes_)
    household_size = st.number_input("Household Size", min_value=1, max_value=20, value=3)
    age_of_respondent = st.number_input("Age", min_value=16, max_value=100, value=30)
    gender = st.selectbox("Gender", label_encoders["gender_of_respondent"].classes_)
    relationship = st.selectbox("Relationship with Head", label_encoders["relationship_with_head"].classes_)
    marital_status = st.selectbox("Marital Status", label_encoders["marital_status"].classes_)
    education = st.selectbox("Education Level", label_encoders["education_level"].classes_)
    job_type = st.selectbox("Job Type", label_encoders["job_type"].classes_)

    submitted = st.form_submit_button("🔍 Predict")

if submitted:
    # Encode inputs
    input_dict = {
        "country": label_encoders["country"].transform([country])[0],
        "year": year,
        "location_type": label_encoders["location_type"].transform([location_type])[0],
        "cellphone_access": label_encoders["cellphone_access"].transform([cellphone_access])[0],
        "household_size": household_size,
        "age_of_respondent": age_of_respondent,
        "gender_of_respondent": label_encoders["gender_of_respondent"].transform([gender])[0],
        "relationship_with_head": label_encoders["relationship_with_head"].transform([relationship])[0],
        "marital_status": label_encoders["marital_status"].transform([marital_status])[0],
        "education_level": label_encoders["education_level"].transform([education])[0],
        "job_type": label_encoders["job_type"].transform([job_type])[0],
    }

    input_df = pd.DataFrame([input_dict])

    # Predict
    prediction = model.predict(input_df)[0]

    # Result
    if prediction == 1:
        st.success(" This person is **likely to have a bank account**.")
    else:
        st.warning(" This person is **unlikely to have a bank account**.")
       
