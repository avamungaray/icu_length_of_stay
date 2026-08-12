import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE" # Prevents memory conflict
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"   # Mutes heavy logging

import streamlit as st
import numpy as np
import pandas as pd
import tensorflow as tf
import joblib

st.cache_data.clear()
st.cache_resource.clear()

# Safely import Keras to avoid app crashes on local hardware
try:
    import keras
    MODEL_AVAILABLE = True
except ImportError:
    MODEL_AVAILABLE = False

# Load Model
@st.cache_resource
def load_icu_lstm_model():
    model = tf.keras.models.load_model("lstm_model_new.keras", compile=False)
    return model

@st.cache_resource
def load_scalers():
    base_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    # Load the single scaler object
    scaler = joblib.load(os.path.join(base_path, "icu_scaler.pkl"))
    return scaler

scaler = load_scalers()

# Page Headers
st.title("ICU Bed-Day Forecasting Dashboard")
st.caption("Predicting remaining length of stay and discharge logistics using live sequential vital trends.")

### PATIENT PROFILE
with st.expander("Current Patient Intake Profile", expanded=True):
    
    # Create two columns
    profile_col1, profile_col2 = st.columns(2)
    
    with profile_col1:
        # Patient age input
        patient_age = st.slider("Patient Age", 18, 100, 65, key="patient_age_slider")

        # Patient gender dropdown
        selected_gender = st.selectbox("Gender", ["Female", "Male"])
        
        # Admission Type Grouping
        admission_type_mapping = {
            "Emergency / Acute Care": ["Direct Emergency", "Emergency Ward", "Urgent", "Emergency Room"],
            "Observation Status": ["Ambulatory Observation", "Direct Observation", "Emergency Unit Observation", "Observation Admit"],
            "Scheduled / Elective Surgery": ["Elective", "Surgical Same Day Admission"]
        }
        # Admission category dropdown
        selected_type_group = st.selectbox(
            "Admission Category",
            options=list(admission_type_mapping.keys())
        )
        
    with profile_col2:
        # Race Grouping
        race_mapping = {
            "White": "race_WHITE",
            "Black": "race_BLACK/AFRICAN AMERICAN",
            "Hispanic/Latino": "race_HISPANIC OR LATINO",
            "Asian": "race_ASIAN",
            "Other/Declined": "race_OTHER"
            }
        # Patient Rrce dropdown
        selected_race_group = st.selectbox(
            "Race/Ethnicity", 
            options=list(race_mapping.keys())
        )

        # Grab matching text column
        chosen_race = race_mapping[selected_race_group]
        
        # Admission Location Grouping
        admission_location_mapping = {
            "Emergency Department": ["Emergency Room", "Walk-in/Self Referral"],
            "Surgical / Interventional Units": ["Post Anesthesia Care Unit (PACU)", "Ambulatory Surgery Transfer", "Procedure Site"],
            "Outpatient / Provider Referral": ["Clinical Referral", "Physician Referral"],
            "Facility / Internal Transfer": ["Transfer From Hospital", "Transfer From Skilled Nursing Facility", "Internal Transfer To or From Psych", "Information Not Available"]
        }
        # Admission location dropdown
        selected_location_group = st.selectbox(
            "Admission Location",
            options=list(admission_location_mapping.keys())
        )

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
    1: [75, 16, 120, 95, 98.6],
    2: [78, 17, 118, 78, 98.7],
    3: [85, 20, 130, 85, 99.1],
    4: [92, 24, 142, 90, 100.4]
}

# Use loop to create sliders for each vital sign
for day, tab in zip([1, 2, 3, 4], tabs):
    with tab:
        col1, col2, col3, col4, col5 = st.columns(5)
        d = defaults[day]
        vitals[day]["daily_heart_rate"] = col1.slider("Heart Rate (bpm)", 30, 220, d[0], key=f"hr{day}")
        vitals[day]["daily_resp_rate"] = col2.slider("Resp Rate (breaths/min)", 4, 60, d[1], key=f"rr{day}")
        vitals[day]["daily_spo2"] = col4.slider("Oxygen Saturation (%)", 50, 100, d[3], key=f"spo2{day}")
        vitals[day]["daily_sys_bp"] = col3.slider("Systolic BP (mmHg)", 40, 280, d[2], key=f"sbp{day}")
        vitals[day]["daily_temp"] = col5.slider("Temperature (°F)", 88.0, 110.0, d[4], step=0.1, key=f"temp{day}")

# DATA PIPELINE

# Admission type and location values from dataframe
all_admission_types = [
    'admission_type_AMBULATORY OBSERVATION', 'admission_type_DIRECT EMER.',
    'admission_type_DIRECT OBSERVATION', 'admission_type_ELECTIVE',
    'admission_type_EU OBSERVATION', 'admission_type_EW EMER.',
    'admission_type_OBSERVATION ADMIT',
    'admission_type_SURGICAL SAME DAY ADMISSION', 'admission_type_URGENT'
]

all_admission_locations = [
    'admission_location_AMBULATORY SURGERY TRANSFER', 'admission_location_CLINIC REFERRAL',
    'admission_location_EMERGENCY ROOM', 'admission_location_INFORMATION NOT AVAILABLE',
    'admission_location_INTERNAL TRANSFER TO OR FROM PSYCH', 'admission_location_PACU', 
    'admission_location_PHYSICIAN REFERRAL', 'admission_location_PROCEDURE SITE',
    'admission_location_TRANSFER FROM HOSPITAL',
    'admission_location_TRANSFER FROM SKILLED NURSING FACILITY', 'admission_location_WALK-IN/SELF REFERRAL'
]

all_races = [
    'race_AMERICAN INDIAN/ALASKA NATIVE', 'race_ASIAN', 'race_ASIAN - ASIAN INDIAN', 
    'race_ASIAN - CHINESE', 'race_ASIAN - KOREAN', 'race_ASIAN - SOUTH EAST ASIAN',
    'race_BLACK/AFRICAN', 'race_BLACK/AFRICAN AMERICAN', 'race_BLACK/CAPE VERDEAN', 
    'race_BLACK/CARIBBEAN ISLAND', 'race_HISPANIC OR LATINO', 'race_HISPANIC/LATINO - CENTRAL AMERICAN',
    'race_HISPANIC/LATINO - COLUMBIAN', 'race_HISPANIC/LATINO - CUBAN', 'race_HISPANIC/LATINO - DOMINICAN', 
    'race_HISPANIC/LATINO - GUATEMALAN', 'race_HISPANIC/LATINO - HONDURAN', 'race_HISPANIC/LATINO - MEXICAN',
    'race_HISPANIC/LATINO - PUERTO RICAN', 'race_HISPANIC/LATINO - SALVADORAN', 'race_MULTIPLE RACE/ETHNICITY',
    'race_NATIVE HAWAIIAN OR OTHER PACIFIC ISLANDER', 'race_OTHER', 'race_PATIENT DECLINED TO ANSWER', 
    'race_PORTUGUESE', 'race_SOUTH AMERICAN', 'race_UNABLE TO OBTAIN', 'race_UNKNOWN',
    'race_WHITE', 'race_WHITE - BRAZILIAN', 'race_WHITE - EASTERN EUROPEAN',
    'race_WHITE - OTHER EUROPEAN', 'race_WHITE - RUSSIAN'
]

# -------------------------------------------------------------------------
# AI USAGE CITATION
# Tool: Gemini
# Prompt: "Write streamlit code that one-hot encodes variables based on an
# existing master list and loops through each day (1-4) to add the one-hot
# encoded values. My intial attempts are throwing me an error (model expects
# a strict 3D tensore shape of (1, 4, 66)). Please provide guidance on
# fixing this error."
# Usage: Used one-hot encoding approach; used matrix loop approach; used
# dynamic padding approach to resolve initial error messages.
# -------------------------------------------------------------------------

# Create one-hot encoded lists
type_one_hot = [1.0 if any(cat.upper() in t.upper() or (cat == "Emergency Ward" and "EW EMER" in t) for cat in admission_type_mapping[selected_type_group]) else 0.0 for t in all_admission_types]
location_one_hot = [1.0 if any(loc.replace(" (PACU)", "").split(" / ")[0][:15].upper() in s.upper() for loc in admission_location_mapping[selected_location_group]) else 0.0 for s in all_admission_locations]
race_one_hot = [1.0 if r == chosen_race else 0.0 for r in all_races]
gender_one_hot = [1.0 if g == f"gender_{selected_gender[0]}" else 0.0 for g in ["gender_F", "gender_M"]]

# Build multi-day matrix loop
sequence_matrix = []
for day in [1, 2, 3, 4]:
    day_row = [
        float(patient_age),
        float(vitals[day]["daily_heart_rate"]),
        float(vitals[day]["daily_resp_rate"]),
        float(vitals[day]["daily_spo2"]),
        float(vitals[day]["daily_sys_bp"]),
        float(vitals[day]["daily_temp"])
    ] + gender_one_hot + race_one_hot + type_one_hot + location_one_hot
    sequence_matrix.append(day_row)

# Build matrices
matrix_all = np.array(sequence_matrix)

# Scale full feature matrix
X_scaled = scaler.transform(matrix_all)

#X_vitals_scaled = X_scaled[:, 1:6].reshape(1, 4, 5)
#X_static_scaled = X_scaled[0].reshape(1, 61)

# LIVE MODEL PREDICTION
st.markdown("---")
st.subheader("Operational Capacity & Bed-Day Forecasting")

# -------------------------------------------------------------------------
# AI USAGE CITATION
# Tool: Gemini
# Prompt: "Write streamlit code that will display a patient's probability of
# an extended ICU stay (>4 days), as well as their probability of discharge
# within 4 days. In addition to these two percentages, I want to display 
# an operational alert that labels each patient as high, medium, or low risk
# based on their extended stay probability, with 36.84% being the low threshold
# and 65% being the medium theshold."
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
        
        # Display operational metrics
        metric1, metric2 = st.columns(2)
        
        # Metric 1: Probability of the patient exceeding the 4-day ICU threshold
        metric1.metric(label="Probability of Extended Stay (>4 Days)", value=f"{extended_stay_prob:.1%}")
        
        # Metric 2: Probability of a standard/short stay (Discharge Readiness within 4 days)
        standard_stay_prob = 1.0 - extended_stay_prob
        metric2.metric(label="Targeted 4-Day Discharge Probability", value=f"{standard_stay_prob:.1%}")
        
        # Operational alert logic tied to classification threshold
        if extended_stay_prob >= 0.70:
            st.error(f"**High Capacity Risk:** {extended_stay_prob:.1%} probability of exceeding a 4-day stay. Flagged for intensive long-term care routing and bed-mitigation logistics.")
        elif extended_stay_prob >= 0.35:
            st.warning(f"**Borderline Capacity Risk:** {extended_stay_prob:.1%} probability of an extended stay. Monitor closely; patient is tracking near the critical 4-day operational threshold.")
        else:
            st.success(f"**Optimized Capacity Stream:** {standard_stay_prob:.1%} probability of standard discharge under 4 days. High confidence for rapid bed turnover.")

    # Throw error if model is None        
    except Exception as eval_error:
        st.error(f"Pipeline Execution Mismatch: {eval_error}")
