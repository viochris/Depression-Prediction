# ==============================================================================
# IMPORT NECESSARY LIBRARIES
# ==============================================================================
import pandas as pd
import numpy as np
import streamlit as st
import joblib

# ==============================================================================
# GLOBAL CONSTANTS & CONFIGURATIONS
# ==============================================================================
RANDOM_SEED = 42

# Define feature categories (excluding the features dropped during selection)
NUMERIC_COLUMNS = [
    'Age', 'Academic Pressure', 'CGPA', 'Study Satisfaction', 
    'Work/Study Hours', 'Financial Stress'
]

OBJECT_COLUMNS = [
    'City', 'Profession', 'Sleep Duration', 'Dietary Habits', 
    'Degree', 'Have you ever had suicidal thoughts ?', 
    'Family History of Mental Illness'
]

TARGET_LABELS = ["Not Depressed (0)", "Depressed (1)"]

# ==============================================================================
# HELPER FUNCTIONS
# ==============================================================================
def change_data_to_df(
    age, gender, city, cgpa, sleep_duration, profession, work_pressure,
    academic_pressure, study_satisfaction, job_satisfaction, dietary_habits,
    degree, suicidal_thoughts, work_study_hours, financial_stress, family_history
):
    """
    Converts raw input data from the Streamlit UI into a structured Pandas DataFrame.
    Automatically drops features that were proven insignificant during model training.
    
    Returns:
        pd.DataFrame: A single-row dataframe ready for pipeline inference.
    """
    
    # Wrap scalar values in lists [] to successfully construct a single-row DataFrame
    raw_data = {
        "Age": [age],
        "Gender": [gender],
        "City": [city],
        "CGPA": [cgpa],
        "Sleep Duration": [sleep_duration],
        "Profession": [profession],
        "Work Pressure": [work_pressure],
        "Academic Pressure": [academic_pressure],
        "Study Satisfaction": [study_satisfaction],
        "Job Satisfaction": [job_satisfaction],
        "Dietary Habits": [dietary_habits],
        "Degree": [degree],
        "Have you ever had suicidal thoughts ?": [suicidal_thoughts],
        "Work/Study Hours": [work_study_hours],
        "Financial Stress": [financial_stress],
        "Family History of Mental Illness": [family_history]
    }
    
    df_testing = pd.DataFrame(raw_data)
    
    # Drop columns deemed insignificant during Feature Selection to match the model's expected input
    insignificant_columns = ["Gender", "Job Satisfaction", "Work Pressure"]
    df_testing.drop(columns=insignificant_columns, inplace=True)
    
    return df_testing

# ==============================================================================
# MODEL LOADING
# ==============================================================================
@st.cache_resource
def load_models():
    """
    Loads the trained machine learning model and LIME training data.
    Includes robust error handling to halt execution safely if files are missing/corrupted.
    """
    try:
        best_model = joblib.load("Depression-Prediction-Model/best_model.joblib")
        return best_model
        
    except Exception as e:
        if isinstance(e, FileNotFoundError):
            st.error("Error: Model files not found. Please ensure 'best_model.joblib' and 'lime_training_data.npy' exist in the 'Depression-Prediction-Model' directory.")
        elif isinstance(e, EOFError):
            st.error("Error: Model file is corrupted or incomplete. Please re-export the model from your Jupyter Notebook.")
        else:
            st.error(f"Error: An unexpected system error occurred while loading the models. Details: {str(e)}")
        
        # Halt Streamlit execution immediately to prevent further cascading errors
        st.stop()

# Initialize the model and training data globally
best_model = load_models()


# ==============================================================================
# PREDICTION LOGIC
# ==============================================================================
def predict_status(df_testing, best_model=best_model):
    """
    Runs the cleaned input data through the model pipeline to get predictions 
    and confidence probabilities.
    """
    # 1. Get binary prediction (0 or 1)
    y_pred = best_model.predict(df_testing)
    
    # 2. Get probability scores for both classes
    y_pred_proba = best_model.predict_proba(df_testing)

    # 3. Map the numeric prediction to the actual string label
    result = [TARGET_LABELS[pred] for pred in y_pred]

    # 4. Extract final outputs for the Streamlit UI
    pred = result[0]
    depressed_proba = ((y_pred_proba[:, 1] * 100).round(2).astype(str) + '%')[0]
    conf = ((y_pred_proba.max(axis=1) * 100).round(2).astype(str) + '%')[0]
    
    return pred, depressed_proba, conf