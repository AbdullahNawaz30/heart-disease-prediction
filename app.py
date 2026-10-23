
import streamlit as st
import pandas as pd
import joblib

@st.cache_resource
def load_model():
    return joblib.load("heart_disease_model.pkl")

model = load_model()

st.set_page_config(
    page_title="Heart Disease Prediction",
    page_icon="❤️",
    layout="centered"
)

st.title("Heart Disease Prediction System")
st.write(
    "Enter the four attributes below to get a machine learning prediction."
)

st.info(
    "Educational project only. This result is not a medical diagnosis."
)

age = st.number_input(
    "Age (years)", min_value=1, max_value=120, value=50
)

sex = st.selectbox(
    "Sex (dataset coding)",
    options=[0, 1],
    format_func=lambda x: "0" if x == 0 else "1"
)

cp = st.selectbox(
    "Chest Pain Type (dataset code)",
    options=[1, 2, 3, 4]
)

thalach = st.number_input(
    "Maximum Heart Rate Achieved",
    min_value=50, max_value=250, value=150
)

if st.button("Predict", type="primary"):
    patient = pd.DataFrame(
        [[age, sex, cp, thalach]],
        columns=["age", "sex", "cp", "thalach"]
    )

    prediction = int(model.predict(patient)[0])

    # LabelEncoder sorted the class names alphabetically:
    # 0 = Disease Present; 1 = No Disease
    if prediction == 0:
        st.warning("Model prediction: Heart disease present.")
    else:
        st.success("Model prediction: No heart disease detected.")

    st.caption(
        "This prediction is based on a limited set of four attributes "
        "and must not be used for medical decisions."
    )
