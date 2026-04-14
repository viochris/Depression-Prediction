# ==============================================================================
# IMPORT NECESSARY LIBRARIES
# ==============================================================================
import streamlit as st
import streamlit.components.v1 as components

# Import custom helper functions from the local 'function.py' module
from function import change_data_to_df, predict_status

# ==============================================================================
# STREAMLIT PAGE CONFIGURATION
# ==============================================================================
# Configures the browser tab title, favicon, and sets the layout width
st.set_page_config(
    page_title="Student Depression Predictor", 
    page_icon="🧠", 
    layout="centered"
)

# ==============================================================================
# HERO SECTION & CUSTOM CSS STYLING
# ==============================================================================
# Injects custom HTML and CSS to create a modern, dark-themed dashboard header.
# The styling includes a gradient title and a highlighted description box.
st.markdown(
    """
    <style>
    /* Main Hero Container */
    .hero-container { 
        text-align: center; 
        padding-bottom: 2rem; 
    }

    /* Gradient Title - Health & Tech Theme */
    .gradient-text { 
        font-size: 2.8rem; 
        font-weight: 800; 
        background: -webkit-linear-gradient(45deg, #11998e, #38ef7d); 
        -webkit-background-clip: text; 
        -webkit-text-fill-color: transparent; 
        margin-bottom: 0.5rem; 
    }

    /* Sub-hook styling for secondary emphasis */
    .sub-hook { 
        font-size: 1.2rem; 
        font-weight: 500; 
        color: #A0AEC0; 
        margin-bottom: 2rem; 
    }

    /* Description Box with Accent Border */
    .description-box { 
        background-color: #1E1E2E; 
        padding: 1.5rem 2rem; 
        border-radius: 8px; 
        border-left: 4px solid #38ef7d; 
        text-align: left; 
        font-size: 1rem; 
        line-height: 1.6; 
        color: #E2E8F0; 
        margin-top: 1.5rem; 
    }
    </style>

    <div class="hero-container">
        <div class="gradient-text">🧠 Student Depression Predictor</div>
        <div class="sub-hook">Early detection of mental health risks in academic environments.</div>
        <div class="description-box">
            Input the student's profile and academic details below. Our Machine Learning model 
            (powered by <b>Logistic Regression</b>) will analyze the behavioral patterns and estimate 
            the probability of depression, allowing for proactive mental health support.
        </div>
    </div>
    """,
    unsafe_allow_html=True
)

# ==============================================================================
# DATA INPUT FORM - COLUMN 1 (Demographics & Academic Profile)
# ==============================================================================
with st.form("depression_prediction_form"):
    
    # Form Header and Description
    st.markdown("### 📋 Student Profile Entry")
    st.markdown("Please fill out the details below. All fields are required for an accurate prediction.")

    # Initialize a 2-column layout to organize inputs cleanly
    column_1, column_2 = st.columns(2)

    with column_1:
        # Demographic Input: Gender
        gender = st.selectbox(
            label="Gender",
            options=["Male", "Female"],
            index=0,
            help="Select the biological gender. This feature is collected but excluded from final calculations to prevent model bias."
        )

        # Demographic Input: Age
        age = st.number_input(
            label="Student Age (Years)",
            value=20,
            min_value=18,
            max_value=59,
            help="Enter the current age of the student. Valid range is 18 to 59 years."
        )

        # Demographic Input: Living Environment
        city = st.selectbox(
            label="Residential City Tier",
            options=["Tier 1", "Tier 2", "Tier 3"],
            index=0,
            help="Select the living environment. Tier 1 = Major metropolitan city; Tier 2 = Mid-sized developing city; Tier 3 = Smaller towns or rural areas."
        )

        # Academic Input: Performance Metric
        cgpa = st.number_input(
            label="Current CGPA (Scale 0.0 - 10.0)",
            value=3.0,
            min_value=0.0,
            max_value=10.0,
            step=0.1,
            help="Enter the cumulative grade point average. Use decimals if necessary."
        )

        # Lifestyle Input: Sleep Habits
        sleep_duration = st.selectbox(
            label="Daily Sleep Duration",
            options=['Less than 5 hours', '5-6 hours', '7-8 hours', 'More than 8 hours'],
            index=2,
            help="Select the average number of hours the student sleeps per night."
        )

        # Professional Input: Primary Role
        profession = st.selectbox(
            label="Primary Occupation",
            options=[
                'Student', 'Working'
            ],
            index=0,
            help="Select the student's primary job or role. If not working, select 'Student'."
        )

        # Psychological Input: Work Stress
        work_pressure = st.number_input(
            label="Work Pressure Level (0-5)",
            min_value=0,
            max_value=5,
            value=2,
            help="Rate the stress level originating from work. 0 = No pressure, 5 = Extreme pressure."
        )

        # Psychological Input: Academic Stress
        academic_pressure = st.number_input(
            label="Academic Pressure Level (0-5)",
            min_value=0,
            max_value=5,
            value=2,
            help="Rate the stress level originating from studies. 0 = No pressure, 5 = Extreme pressure."
        )

    with column_2:
        # Academic Input: Satisfaction Metric
        study_satisfaction = st.number_input(
            label="Study Satisfaction Level (0-5)",
            min_value=0,
            max_value=5,
            value=2,
            help="Rate the student's satisfaction with their academic life. 0 = Highly Dissatisfied, 5 = Highly Satisfied."
        )

        # Professional Input: Satisfaction Metric
        job_satisfaction = st.number_input(
            label="Job Satisfaction Level (0-4)",
            min_value=0,
            max_value=4,
            value=1,
            step=1,
            help="Rate the satisfaction with current employment. Note: This feature is collected but excluded during the final model prediction."
        )

        # Lifestyle Input: Diet Quality
        dietary_habits = st.selectbox(
            label="Dietary Habits",
            options=['Healthy', 'Moderate', 'Unhealthy'],
            index=1,
            help="Select the category that best describes the student's daily nutritional intake and eating habits."
        )

        # Academic Input: Current Education Level
        degree = st.selectbox(
            label="Current Degree Program",
            options=['Bachelors', 'Masters', 'Doctorate', 'High School'],
            index=1,
            help="Select the academic degree or certification the student is currently pursuing."
        )

        # Psychological Input: Critical Mental Health Indicator
        suicidal_thoughts = st.selectbox(
            label="History of Suicidal Thoughts",
            options=["No", "Yes"],
            index=1, 
            help="Indicate whether the student has ever experienced suicidal ideation or thoughts of self-harm."
        )

        # Lifestyle Input: Daily Workload
        work_or_study_hours = st.number_input(
            label="Daily Work/Study Hours",
            min_value=0,
            max_value=12,
            value=3,
            help="Enter the average number of hours spent actively working or studying per day."
        )

        # Psychological Input: Financial Burden
        financial_stress = st.number_input(
            label="Financial Stress Level (1-5)",
            min_value=1,
            max_value=5,
            value=3,
            help="Rate the level of stress caused by financial constraints or obligations. 1 = Minimal Stress, 5 = Severe Stress."
        )

        # Psychological Input: Genetic/Family Factor
        mental_illness_history = st.selectbox(
            label="Family History of Mental Illness",
            options=["No", "Yes"],
            index=1,
            help="Indicate if there is any known history of mental health disorders in the student's immediate family."
        )

    # ==============================================================================
    # FORM SUBMISSION TRIGGER
    # ==============================================================================
    # This button binds all the inputs within the form. When clicked, it sets 
    # 'submitted' to True, which subsequently triggers the ML inference pipeline.
    submitted = st.form_submit_button("Predict Depression Status 🚀", use_container_width=True)

# ==============================================================================
# PREDICTION EXECUTION & UI OUTPUT
# ==============================================================================
# This block executes only when the user clicks the submit button
if submitted:
    with st.spinner("🤖 AI is analyzing student profile..."):
        # Step 1: Ingest raw inputs from the UI and map them to a Pandas DataFrame
        # Note: Variable names here must exactly match the variables defined in the form
        df_testing = change_data_to_df(
            age,
            gender,
            city,
            cgpa,
            sleep_duration,
            profession,
            work_pressure,
            academic_pressure,
            study_satisfaction,
            job_satisfaction,
            dietary_habits,
            degree,
            suicidal_thoughts,
            work_or_study_hours, 
            financial_stress,
            mental_illness_history
        )
                
        # Step 2: Run the Logistic Regression model to get prediction and confidence metrics
        prediction, depressed_proba, conf = predict_status(df_testing)

    # Step 3: Display the primary prediction results in a modern, customized HTML card
    st.markdown(
        f"""
        <div style='background-color: #1E1E1E; padding: 25px; border-radius: 12px; border-left: 6px solid #00CC96; box-shadow: 0 4px 8px rgba(0,0,0,0.2); margin-top: 20px;'>
            <h3 style='color: #00CC96; margin-top: 0; font-family: sans-serif;'>🎉 Analysis Complete!</h3>
            <p style='font-size: 16px; color: #E0E0E0; margin-bottom: 5px; font-family: sans-serif;'>Predicted Student Status:</p>
            <p style='font-size: 32px; font-weight: bold; color: #FFD700; margin-top: 0; margin-bottom: 15px; font-family: monospace;'>
                {prediction}
            </p>
            <hr style='border-color: #333333;'>
            <p style='color: #A0A0A0; margin-bottom: 5px; font-size: 14px; font-family: sans-serif;'>
                📊 <strong>Probability of Depression:</strong> {depressed_proba}
            </p>
            <p style='color: #A0A0A0; margin-bottom: 0; font-size: 14px; font-family: sans-serif;'>
                🤖 <strong>Model Confidence Score:</strong> {conf}
            </p>
        </div>
        """,
        unsafe_allow_html=True
    )