import streamlit as st
import pandas as pd
import joblib

@st.cache_resource
def load_model():
    # Load the trained pipeline/model
    return joblib.load("model.pkl")  # Change filename if needed

st.title("Performance Rating Prediction App")

st.markdown("""
Enter the details below to predict the **Performance Rating**.
""")

# Load model
model = load_model()

# Form for input
with st.form("prediction_form"):
    st.subheader("Employee Information")

    job_level_middle = st.selectbox(
        "Job Level - Middle",
        options=[0, 1],
        format_func=lambda x: "Yes" if x == 1 else "No"
    )

    department_others = st.selectbox(
        "Department - Others",
        options=[0, 1],
        format_func=lambda x: "Yes" if x == 1 else "No"
    )

    location_asia = st.selectbox(
        "Location - Asia-Pacific",
        options=[0, 1],
        format_func=lambda x: "Yes" if x == 1 else "No"
    )

    location_europe = st.selectbox(
        "Location - Europe",
        options=[0, 1],
        format_func=lambda x: "Yes" if x == 1 else "No"
    )

    location_others = st.selectbox(
        "Location - Others",
        options=[0, 1],
        format_func=lambda x: "Yes" if x == 1 else "No"
    )

    billable_days = st.number_input(
        "Billable Days",
        min_value=0,
        max_value=365,
        value=0
    )

    layoff_risk = st.selectbox(
        "Layoff Risk",
        options=[0, 1],
        format_func=lambda x: "Yes" if x == 1 else "No"
    )

    submitted = st.form_submit_button("Predict")

if submitted:
    # DataFrame with columns in same order as training
    input_data = pd.DataFrame([[
        job_level_middle,
        department_others,
        location_asia,
        location_europe,
        location_others,
        billable_days,
        layoff_risk
    ]], columns=[
        'job_level_Middle',
        'department_Others',
        'location_Asia-Pacific',
        'location_Europe',
        'location_Others',
        'billable_days',
        'layoff_risk'
    ])

    prediction = model.predict(input_data)[0]
    st.success(f"Predicted Performance Rating: {prediction:.2f}")
