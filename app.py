import streamlit as st
import pandas as pd
import joblib

# --- Load Model and Define Columns ---
# Load the trained model. Ensure 'model.joblib' is in the same directory.
try:
    model = joblib.load('model.joblib')
except FileNotFoundError:
    st.error("Error: `model.joblib` not found. Please ensure the trained model file is in the correct directory.")
    st.stop()

# Define the exact feature columns the model was trained on.
# This list MUST match the features used for training your model.
model_columns = [
    'age', 'years_of_experience', 'billable_days', 'bench_time_days',
    'performance_rating', 'training_hours', 'project_utilization',
    'redeployment_attempts', 'redeployment_success', 'job_level_Middle',
    'job_level_Senior', 'department_Software Development', 'department_Testing',
    'location_North America', 'salary_band_Low'
]

# Load the raw dataset to get unique values for dropdowns
try:
    df_raw = pd.read_csv('tcs_layoff_dataset.csv')
except FileNotFoundError:
    st.error("Error: `tcs_layoff_dataset.csv` not found. Please ensure the dataset file is in the correct directory.")
    st.stop()


# --- Streamlit App Interface ---
st.set_page_config(page_title="TCS Layoff Risk Predictor", layout="wide")
st.title('🔮 TCS Layoff Risk Prediction')
st.markdown("""
This app predicts an employee's layoff risk score based on their professional details.
The prediction is made using a machine learning model trained on historical data.
""")


# --- Sidebar for User Input ---
st.sidebar.header('Employee Input Features')

def user_input_features():
    """
    Creates sidebar widgets to collect user input and returns a DataFrame.
    """
    # Numerical features
    age = st.sidebar.slider('Age', 20, 60, 35)
    years_of_experience = st.sidebar.slider('Years of Experience', 0.0, 40.0, 10.0, 0.5)
    billable_days = st.sidebar.slider('Billable Days (in a year)', 0, 365, 200)
    bench_time_days = st.sidebar.slider('Bench Time Days (in a year)', 0, 365, 60)
    performance_rating = st.sidebar.slider('Performance Rating', 1.0, 5.0, 3.5, 0.1)
    training_hours = st.sidebar.slider('Training Hours', 0, 100, 20)
    project_utilization = st.sidebar.slider('Project Utilization (%)', 0.0, 100.0, 80.0, 0.1)
    redeployment_attempts = st.sidebar.slider('Redeployment Attempts', 0, 5, 1)
    
    # Categorical and Binary features
    redeployment_success = st.sidebar.selectbox('Redeployment Success', options=[0, 1], format_func=lambda x: 'Yes' if x == 1 else 'No')
    job_level = st.sidebar.selectbox('Job Level', options=sorted(df_raw['job_level'].unique()))
    department = st.sidebar.selectbox('Department', options=sorted(df_raw['department'].unique()))
    location = st.sidebar.selectbox('Location', options=sorted(df_raw['location'].unique()))
    salary_band = st.sidebar.selectbox('Salary Band', options=sorted(df_raw['salary_band'].unique()))

    # Create a dictionary of the raw input data
    data = {
        'age': age,
        'years_of_experience': years_of_experience,
        'billable_days': billable_days,
        'bench_time_days': bench_time_days,
        'performance_rating': performance_rating,
        'training_hours': training_hours,
        'project_utilization': project_utilization,
        'redeployment_attempts': redeployment_attempts,
        'redeployment_success': redeployment_success,
        'job_level': job_level,
        'department': department,
        'location': location,
        'salary_band': salary_band
    }
    
    features = pd.DataFrame(data, index=[0])
    return features

# Get user input
input_df = user_input_features()

# Display the user's selected features in the main panel
st.subheader('User Input Parameters')
st.write(input_df)


# --- Preprocess User Input for Prediction ---
# Create a copy to avoid changing the displayed input_df
input_to_process = input_df.copy()

# One-hot encode the categorical features
input_processed = pd.get_dummies(input_to_process, columns=['job_level', 'department', 'location', 'salary_band'])

# Align the columns of the input data with the columns the model was trained on
input_aligned = input_processed.reindex(columns=model_columns, fill_value=0)


# --- Make Prediction ---
prediction = model.predict(input_aligned)


# --- Display Prediction Result ---
st.subheader('Prediction Result')
layoff_risk_score = prediction[0] * 100

st.write(f"Predicted Layoff Risk Score:")
st.progress(int(layoff_risk_score))
st.markdown(f"<h2 style='text-align: center; color: orange;'>{layoff_risk_score:.2f}%</h2>", unsafe_allow_html=True)

# Provide a contextual message
if layoff_risk_score > 60:
    st.error('High Risk of Layoff 😟: This employee shows several indicators associated with high layoff risk. Proactive measures may be required.')
elif layoff_risk_score > 30:
    st.warning('Moderate Risk of Layoff 🤔: This employee has some indicators of layoff risk. Monitoring and support are recommended.')
else:
    st.success('Low Risk of Layoff 😊: This employee currently shows a low risk of layoff based on the provided data.')

# Add a disclaimer
st.info("**Disclaimer:** This prediction is based on a machine learning model and historical data. It should be used as a supportive tool and not as the sole basis for personnel decisions.")
