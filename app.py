import joblib
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Heart Disease Prediction", page_icon="❤️")

@st.cache_resource
def load_artifacts():
    model = joblib.load("model/heart_model.pkl")
    le_sex = joblib.load("model/le_sex.pkl")
    le_cp = joblib.load("model/le_cp.pkl")
    return model, le_sex, le_cp

model, le_sex, le_cp = load_artifacts()

st.title("❤️ Heart Disease Prediction System")
st.write("Enter the patient's details to estimate the likelihood of heart disease.")

age = st.slider("Age", 20, 90, 50)
sex = st.selectbox("Sex", list(le_sex.classes_))
cp = st.selectbox("Chest pain type", list(le_cp.classes_))
thalach = st.slider("Maximum heart rate achieved", 70, 210, 150)

if st.button("Predict"):
    X = pd.DataFrame([{
        "age": age,
        "sex": le_sex.transform([sex])[0],
        "cp": le_cp.transform([cp])[0],
        "thalach": thalach,
    }])[["age", "sex", "cp", "thalach"]]

    pred = model.predict(X)[0]
    prob = model.predict_proba(X)[0][1]

    if pred == 1:
        st.error(f"⚠️ Heart disease likely (probability {prob:.1%})")
    else:
        st.success(f"✅ Heart disease unlikely (probability of disease {prob:.1%})")
    st.caption("Educational project only – not a medical diagnosis.")
