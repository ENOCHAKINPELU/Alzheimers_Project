import google.generativeai as genai
import pandas as pd
import streamlit as st
import json
import time
import os

# Load environment variable for API key
GOOGLE_API_KEY = "AIzaSyCYT1rz4iRaadHKWei_OYm6XTXIQDvRO40"
genai.configure(api_key=GOOGLE_API_KEY)

#Model Selection
MODEL_NAME = "gemini-pro"

# --- Utility Functions ---
def load_data_columns(file_path="data_columns.json"):
    """Loads data columns from the JSON file and handles potential errors."""
    try:
        with open(file_path, "r") as f:
            return json.load(f).get("data_columns", [])
    except FileNotFoundError:
        st.error(f"{file_path} not found. Please make sure the file exists.")
    except json.JSONDecodeError:
        st.error(f"Error parsing {file_path}. Please check the file content.")
    except Exception as e:
        st.error(f"Error loading {file_path}: {e}")
    return []


def is_valid_json(text):
    """Checks if a string is valid JSON before parsing."""
    try:
        json.loads(text)
        return True
    except json.JSONDecodeError:
        return False


def parse_json_recommendations(recommendation_text):
    """Parses the JSON output. Returns list of recommendations or None if fails."""
    if not recommendation_text:
        return None
    if not is_valid_json(recommendation_text):
        st.error(f"Invalid JSON response: {recommendation_text}")
        return None
    try:
        return json.loads(recommendation_text)
    except json.JSONDecodeError as e:
        st.error(f"Error parsing JSON response: {e}")
        return None
    except Exception as e:
        st.error(f"Error processing recommendations: {e}")
        return None


def get_gemini_recommendations(prompt, max_retries=3, retry_delay=2):
    """Sends the prompt to Gemini API with retries."""
    model = genai.GenerativeModel(MODEL_NAME)
    for attempt in range(max_retries):
        try:
            response = model.generate_content(prompt)
            return response.text
        except Exception as e:
            st.error(f"Error getting Gemini response (Attempt {attempt + 1}/{max_retries}): {e}")
            if attempt < max_retries - 1:
                time.sleep(retry_delay)  # Wait before retrying
            else:
                return None
    return None


def display_recommendations(recommendations):
    """Displays recommendations on Streamlit."""
    if recommendations:
        for i, rec in enumerate(recommendations):
            st.markdown(f"**{i + 1}. Intervention:** {rec['intervention']}")
            st.markdown(f"   **Rationale:** {rec['rationale']}")
            st.markdown(f"   **Confidence:** {rec['confidence']}")
            st.markdown("---")
    else:
        st.warning(
            "No recommendations could be generated or there was an error processing the response. Please check your input or try again."
        )

# --- End Utility Functions ---

# Load data columns (outside main to prevent reloading)
data_columns = load_data_columns()
if not data_columns:
    st.stop()

ETHNICITY_LIST = ["White", "Black", "Asian", "Hispanic or Latino", "Other"]
EDUCATION_LEVELS = [
    "Less than high school",
    "High school",
    "Some college",
    "Bachelor's degree",
    "Graduate degree",
]


def generate_prompt(patient_data, observations=""):
    """Constructs a prompt for the Gemini API."""
    prompt_prefix = """
        Recommend personalized behavioral interventions to improve the quality of life for the following patient.
        Please return recommendations strictly in a valid JSON format, using double quotes for keys and strings.
        Do not provide text before or after the JSON format.

        The JSON format should be like this:
        [
          {"intervention": "Specific intervention 1", "rationale": "Rationale for intervention 1", "confidence": "high/medium/low"},
          {"intervention": "Specific intervention 2", "rationale": "Rationale for intervention 2", "confidence": "high/medium/low"},
          {"intervention": "Specific intervention 3", "rationale": "Rationale for intervention 3", "confidence": "high/medium/low"}
        ]

    """

    prompt_middle = f"""
        Patient Age: {patient_data.get('age')}
        Patient Ethnicity: {patient_data.get('ethnicity')}
        Patient Education Level: {patient_data.get('educationlevel')}
        BMI: {patient_data.get('bmi')}
        Alcohol Consumption: {patient_data.get('alcoholconsumption')}
        Physical Activity: {patient_data.get('physicalactivity')}
        Diet Quality: {patient_data.get('dietquality')}
        Sleep Quality: {patient_data.get('sleepquality')}
        Systolic BP: {patient_data.get('systolicbp')}
        Diastolic BP: {patient_data.get('diastolicbp')}
        Total Cholesterol: {patient_data.get('cholesteroltotal')}
        LDL Cholesterol: {patient_data.get('cholesterolldl')}
        HDL Cholesterol: {patient_data.get('cholesterolhdl')}
        Triglycerides: {patient_data.get('cholesteroltriglycerides')}
        MMSE: {patient_data.get('mmse')}
        Functional Assessment: {patient_data.get('functionalassessment')}
        ADL: {patient_data.get('adl')}
        Risk Factor Score: {patient_data.get('riskfactorscore')}
        Symptom Count: {patient_data.get('symptomcount')}
        Gender: {patient_data.get('gender')}
        Smoking: {patient_data.get('smoking')}
        Family History Alzheimer's: {patient_data.get('familyhistoryalzheimers')}
        Cardiovascular Disease: {patient_data.get('cardiovasculardisease')}
        Diabetes: {patient_data.get('diabetes')}
        Depression: {patient_data.get('depression')}
        Head Injury: {patient_data.get('headinjury')}
        Hypertension: {patient_data.get('hypertension')}
        Memory Complaints: {patient_data.get('memorycomplaints')}
        Behavioral Problems: {patient_data.get('behavioralproblems')}
        Confusion: {patient_data.get('confusion')}
        Disorientation: {patient_data.get('disorientation')}
        Personality Changes: {patient_data.get('personalitychanges')}
        Difficulty Completing Tasks: {patient_data.get('difficultycompletingtasks')}
        Forgetfulness: {patient_data.get('forgetfulness')}
        Diabetes CVD: {patient_data.get('diabetes_cvd')}
        Has Any Risk Factor: {patient_data.get('hasanyriskfactor')}
        Has Any Symptom: {patient_data.get('hasanysymptom')}
        Age Group 70-80: {patient_data.get('agegroup_70-80')}
        Age Group 80-90: {patient_data.get('agegroup_80-90')}
        Age Group 90+: {patient_data.get('agegroup_90+')}
        Observations: {observations}
     """
    prompt = prompt_prefix + prompt_middle
    return prompt


def main():
    st.title("Behavioral Intervention Recommender")

    with st.sidebar:
        st.header("Patient Information")
        input_data = {}
        for column in data_columns:
            if column == "age":
                input_data[column] = st.number_input("Age", min_value=0, step=1, value=30)
            elif column == "ethnicity":
                input_data[column] = st.selectbox("Ethnicity", ETHNICITY_LIST)
            elif column == "gender":
                input_data[column] = st.selectbox("Gender", ["male", "female", "other"])
            elif column == "educationlevel":
                input_data[column] = st.selectbox("Education Level", EDUCATION_LEVELS)
            elif column in ['bmi', 'alcoholconsumption', 'physicalactivity', 'dietquality', 'sleepquality', 'systolicbp', 'diastolicbp', 'cholesteroltotal', 'cholesterolldl', 'cholesterolhdl', 'cholesteroltriglycerides', 'mmse', 'functionalassessment', 'adl', 'riskfactorscore', 'symptomcount']:
                input_data[column] = st.number_input(column, step=0.1, value=0.0)
            else:
                input_data[column] = st.checkbox(column)
        observations = st.text_area("Observations or notes", "")
        submit_button = st.button("Get Recommendations")


    if submit_button:
        recommendations_state = st.session_state.get("recommendations", None) # gets previous recommendation in session state

        prompt = generate_prompt(input_data, observations)
        with st.spinner("Generating recommendations..."):
            recommendation_text = get_gemini_recommendations(prompt)
        recommendations = parse_json_recommendations(recommendation_text)

        if recommendations:
            st.session_state["recommendations"] = recommendations # stores recommendation in session state
        
        st.header("Recommendations:")
        display_recommendations(recommendations)
       
        # Feedback Section
        if recommendations:
            st.subheader("Feedback:")
            feedback = st.radio("Was this recommendation helpful?", ["Yes", "No", "Partially"], key="feedback_radio") # sets a unique key for radio button
            if st.button("Submit Feedback"):
                if feedback == "No":
                    with st.spinner("Generating new recommendations..."):
                        new_recommendation_text = get_gemini_recommendations(prompt)
                    new_recommendations = parse_json_recommendations(new_recommendation_text)
                    if new_recommendations and new_recommendations != recommendations_state: # checks if new response and not the same as previous
                        st.session_state["recommendations"] = new_recommendations # update the recommendation
                        st.success("New recommendations generated successfully!")
                        st.header("New Recommendations:")
                        display_recommendations(new_recommendations)

                    else:
                        st.warning("Could not generate different recommendations. Please try again later.")
                        if recommendations_state:
                            st.header("Previous Recommendations:")
                            display_recommendations(recommendations_state)

                else:
                  st.success(f"Feedback submitted: {feedback}")


if __name__ == "__main__":
    main()