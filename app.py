import streamlit as st
import pandas as pd
import numpy as np
import pickle
from datetime import datetime

# Load the trained model
@st.cache_resource
def load_model():
    with open('model123(1).pkl', 'rb') as file:
        model = pickle.load(file)
    return model

try:
    model = load_model()
except FileNotFoundError:
    st.error("Model file 'model123(1).pkl' not found. Please ensure it is in the same directory.")
    st.stop()

# Initialize in-memory session history if it doesn't exist
if "prediction_history" not in st.session_state:
    st.session_state.prediction_history = []

st.title("Student Placement Predictor")
st.write("Enter the required features to check the binary placement classification output.")

# Numeric Input Fields
cgpa = st.number_input("CGPA", min_value=0.0, max_value=10.0, value=7.5, step=0.1)
internships = st.number_input("Internships", min_value=0, max_value=10, value=1, step=1)
projects = st.number_input("Projects", min_value=0, max_value=10, value=1, step=1)

# Slider Input Fields
aptitude_score = st.slider("Aptitude Test Score", min_value=0, max_value=100, value=75)
soft_skills = st.slider("Soft Skills Rating", min_value=1.0, max_value=5.0, value=4.0, step=0.1)

# Categorical Dropdown Mapped to Binary Variable
extracurricular = st.selectbox("Extracurricular Activities", options=["No", "Yes"], index=0)
extracurricular_mapped = 1 if extracurricular == "Yes" else 0

# Prediction Execution
if st.button("Predict"):
    # Format structural payload matching the model's expected features
    input_data = pd.DataFrame([{
        "CGPA": cgpa,
        "Internships": internships,
        "Projects": projects,
        "AptitudeTestScore": aptitude_score,
        "SoftSkillsRating": soft_skills,
        "ExtracurricularActivities": extracurricular_mapped
    }])
    
    try:
        # Extract explicit binary target value (0 or 1)
        prediction = int(model.predict(input_data)[0])
        
        st.subheader("Model Output Class:")
        st.code(f"{prediction}", language="text")
        
        if prediction == 1:
            st.success("Result: The student will be placed")
        else:
            st.warning("Result: The student will not be placed but if he/she improves one or two parameters, he/she can get placed")
            
        # Log values inside in-memory session storage array
        new_log = {
            "Timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "CGPA": cgpa,
            "Internships": internships,
            "Projects": projects,
            "AptitudeTestScore": aptitude_score,
            "SoftSkillsRating": soft_skills,
            "ExtracurricularActivities": extracurricular,
            "Prediction_Class": prediction
        }
        st.session_state.prediction_history.append(new_log)
        
    except Exception as e:
        st.error(f"Prediction Error: {e}")

# In-Memory Session Storage History and Local File Export Controls
if st.session_state.prediction_history:
    st.write("---")
    st.subheader("Session Log History")
    
    # Render pure DataFrame grid interface
    history_df = pd.DataFrame(st.session_state.prediction_history)
    st.dataframe(history_df, use_container_width=True)
    
    # Download Trigger without file writes
    csv_buffer = history_df.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="Download Logs as CSV",
        data=csv_buffer,
        file_name=f"placement_logs_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
        mime="text/csv"
    )