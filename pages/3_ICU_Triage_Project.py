import os
# Memory fixes
os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"
os.environ["MALLOC_TRIM_THRESHOLD_"] = "0"

# Imports
import streamlit as st
import numpy as np
import pandas as pd
import tensorflow as tf
import joblib

# To help with issues running on computer
st.cache_data.clear()
st.cache_resource.clear()

# Safely import Keras to avoid app crashes on local hardware
try:
    import keras
    MODEL_AVAILABLE = True
except ImportError:
    MODEL_AVAILABLE = False

# Load Model
def load_icu_lstm_model():
    model = tf.keras.models.load_model("lstm_model_new.keras", compile=False)
    return model

# Page Headers
st.title("ICU Bed-Day Forecasting Dashboard")
st.caption("Predicting remaining length of stay and discharge logistics using live sequential vital trends.")

### PATIENT PROFILE
with st.expander("Current Patient Intake Profile", expanded=True):
    
    # Create two columns
    profile_col1, profile_col2 = st.columns(2)
    
    with profile_col1:
        # Patient age input
        patient_age = st.slider("Patient Age", 18, 100, 22, key="patient_age_slider")
        
    with profile_col2:
        # Patient gender dropdown
        selected_gender = st.selectbox("Gender", ["Female", "Male"])

### VITAL SIGN INPUTS
st.subheader("4-Day Clinical Timeline Entry")
st.write("Adjust the 5 core vital signs for each sequential observation day below:")

# Create tabs for each day
tabs= st.tabs(["Observation Day 1", "Observation Day 2", "Observation Day 3", "Observation Day 4"])

# -------------------------------------------------------------------------
# AI USAGE CITATION
# Tool: Gemini
# Prompt: "Write streamlit code that creates 5 vital sign sliders within each 
# of 4 observation day tabs."
# Usage: Used default mapping approach; used looping slider creation approach.
# -------------------------------------------------------------------------

# Dictionary to hold inputs
vitals = {day: {} for day in [1, 2, 3, 4]}

# Map out unique daily defaults
defaults = {
    1: [70, 14, 120, 100, 98.6],
    2: [70, 14, 120, 100, 98.6],
    3: [70, 14, 120, 100, 98.6],
    4: [70, 14, 120, 100, 98.6]
}

# Use loop to create sliders for each vital sign
for day, tab in zip([1, 2, 3, 4], tabs):
    with tab:
        col1, col2, col3, col4, col5 = st.columns(5)
        d = defaults[day]
        vitals[day]["daily_heart_rate"] = col1.slider("Heart Rate (bpm)", 30, 220, d[0], key=f"hr{day}")
        vitals[day]["daily_resp_rate"] = col2.slider("Resp Rate (breaths/min)", 4, 60, d[1], key=f"rr{day}")
        vitals[day]["daily_sys_bp"] = col3.slider("Systolic BP (mmHg)", 40, 280, d[2], key=f"sbp{day}")
        vitals[day]["daily_spo2"] = col4.slider("Oxygen Saturation (%)", 50, 100, d[3], key=f"spo2{day}")
        vitals[day]["daily_temp"] = col5.slider("Temperature (°F)", 88.0, 110.0, d[4], step=0.1, key=f"temp{day}")

# DATA PIPELINE

# Safely load preprocessor
@st.cache_resource
def get_preprocessor():
    return joblib.load("preprocessor.pkl")

# Intialize preprocessor
preprocessor = get_preprocessor()

# Encode gender for model compatability
gender_encoded = 0 if selected_gender == "Female" else 1

# Establish setup for necessary inputs
day_rows = []
for day in [1, 2, 3, 4]:
    day_rows.append({
        "age": float(patient_age),
        "gender": int(gender_encoded),
        "race": int(0),
        "admission_type": int(1),
        "admission_location": int(0),
        "daily_heart_rate": float(vitals[day]["daily_heart_rate"]),
        "daily_resp_rate": float(vitals[day]["daily_resp_rate"]),
        "daily_spo2": float(vitals[day]["daily_spo2"]),
        "daily_sys_bp": float(vitals[day]["daily_sys_bp"]),
        "daily_temp": float(vitals[day]["daily_temp"]),
    })

# Create input dataframe
input_df = pd.DataFrame(day_rows)

# Establish correct column order
expected_columns = ["gender", "race", "admission_type", "admission_location", "age", "daily_heart_rate", 
                    "daily_resp_rate", "daily_spo2", "daily_sys_bp", "daily_temp"]
input_df = input_df[expected_columns]

# Ensure categorical columns are strings
cat_cols = ["gender", "race", "admission_type", "admission_location"]
input_df[cat_cols] = input_df[cat_cols].astype(str)

# -------------------------------------------------------------------------
# AI USAGE CITATION
# Tool: Gemini
# Prompt: "Write streamlit code that will tranform my input datafame with
# my preprocessor, but that has safeguards in place to avoid a crash if
# the transformation fails. Please also add code that enforces the numeric
# variable input to be 6, splits the data by numeric vs. static, and reshapes
# for proper input formatting."
# Usage: Used try/except approach to avoid local errors; used scaled splitting
# and reshaping approach.
# -------------------------------------------------------------------------

# Transform data while ensuring consistent distribution
try:
    clean_X = preprocessor.transform(input_df)
    clean_X = np.asarray(clean_X, dtype=np.float32)
except Exception as e:
    st.error(f"Transformation Error: {e}")
    st.write("DEBUG - Data Types:", input_df.dtypes)
    st.stop()

# Execute final reshaping for model input
num_size = 6 
X_vitals_scaled = clean_X[:, :num_size].reshape(1, 4, num_size)
X_static_scaled = clean_X[0, num_size:].reshape(1, -1)

# LIVE MODEL PREDICTION
st.markdown("---")
st.subheader("Operational Capacity & Bed-Day Forecasting")

# -------------------------------------------------------------------------
# AI USAGE CITATION
# Tool: Gemini
# Prompt: "Write streamlit code that will generate a patient's probability of
# an extended ICU stay (>4 days), as well as their probability of discharge
# within 4 days. While I want to generate these two percentages, I want to display 
# only an operational alert that labels each patient as high, medium, or low risk
# based on their extended stay probability, which will be set later."
# Usage: Used prediction creation and probability extraction approach; used 
# error, warning, and success elements to display operational alerts in 
# relevant colors; used recommended approach to handling potential model errors.
# -------------------------------------------------------------------------

model = None

if model is None:
    model = load_icu_lstm_model()

if model is not None:
    try:  
        prediction = model.predict([X_vitals_scaled, X_static_scaled])

        # Extract probability of an extended stay (> 4 days)
        extended_stay_prob = float(prediction.flatten()[-1])
        
        # Extract probability of a standard/short stay (Discharge Readiness within 4 days)
        standard_stay_prob = 1.0 - extended_stay_prob
        
        # Operational alert logic tied to classification threshold
        if extended_stay_prob >= 0.25:
            st.error(f"{extended_stay_prob:.1%} High risk of exceeding a 4-day stay. Flagged for intensive long-term care routing and bed-mitigation logistics.")
        elif extended_stay_prob >= 0.15:
            st.warning(f"{extended_stay_prob:.1%} Borderline risk of an extended stay. Monitor closely; patient is tracking near the critical 4-day operational threshold.")
        else:
            st.success(f"{extended_stay_prob:.1%} Optimized Capacity Stream. High confidence for standard discharge under 4 days/rapid bed turnover.")

    # Throw error if model is None        
    except Exception as eval_error:
        st.error(f"Pipeline Execution Mismatch: {eval_error}")
