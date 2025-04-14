# -*- coding: utf-8 -*-
"""
Created on Sun Apr 13 23:28:14 2025

@author: laxmi
"""

import pickle
import streamlit as st
from streamlit_option_menu import option_menu
import plotly.graph_objects as go
import numpy as np

# Load models
diabetes_model = pickle.load(open(r"C:\Users\laxmi\disease_prediction\diabetic.sav", 'rb'))
heart_disease = pickle.load(open(r"C:\Users\laxmi\disease_prediction\heart_disease.sav", 'rb'))
parkinson_model = pickle.load(open(r"C:\Users\laxmi\disease_prediction\parkinsons_model.sav", 'rb'))

# Sidebar
with st.sidebar:
    st.image(r"D:\Machine_Learning\medico.webp", width=220)
    st.title("Disease Prediction System 🧬")
    selected = option_menu("Choose Disease", ["Diabetes Prediction", "Heart Disease Prediction", "Parkinson's Prediction"],
                           icons=["activity", "heart", "person"], default_index=0)

# Diabetes Prediction Page
if selected == "Diabetes Prediction":
    st.title("🩸 Diabetes Prediction")

    # Input fields
    Pregnancies = st.number_input("Number of Pregnancies", min_value=0, max_value=20, value=0)
    Glucose = st.slider("Glucose Level", 0, 200, 120)
    BloodPressure = st.slider("Blood Pressure", 0, 140, 70)
    SkinThickness = st.slider("Skin Thickness", 0, 100, 20)
    Insulin = st.slider("Insulin Level", 0, 900, 80)
    BMI = st.slider("BMI", 0.0, 70.0, 25.0)
    DiabetesPedigreeFunction = st.number_input("Diabetes Pedigree Function", 0.0, 2.5, 0.5)
    Age = st.slider("Age", 1, 100, 30)

    input_data = np.array([[Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin,
                            BMI, DiabetesPedigreeFunction, Age]])

    # Feature Overview
    diabetes_features = {
        "Glucose": Glucose,
        "Blood Pressure": BloodPressure,
        "BMI": BMI,
        "Age": Age,
        "Insulin": Insulin
    }

    st.markdown("### 📈 Patient Input Overview")
    fig1 = go.Figure([go.Bar(x=list(diabetes_features.keys()),
                             y=list(diabetes_features.values()),
                             marker_color='skyblue')])
    fig1.update_layout(title_text='Input Features', height=400)
    st.plotly_chart(fig1)

    # Glucose Gauge
    fig2 = go.Figure(go.Indicator(
        mode="gauge+number",
        value=Glucose,
        title={'text': "Glucose Level"},
        gauge={
            'axis': {'range': [0, 200]},
            'bar': {'color': "darkblue"},
            'steps': [
                {'range': [0, 99], 'color': "lightgreen"},
                {'range': [100, 125], 'color': "orange"},
                {'range': [126, 200], 'color': "red"}],
        }))
    st.plotly_chart(fig2)

    # Prediction
    if st.button("Predict Diabetes"):
        result = diabetes_model.predict(input_data)
        if hasattr(diabetes_model, "predict_proba"):
            prob = diabetes_model.predict_proba(input_data)[0]
            st.markdown("### 🎯 Prediction Probability")
            st.success(f"Non-Diabetic: {prob[0]*100:.2f}%")
            st.error(f"Diabetic: {prob[1]*100:.2f}%")

        st.markdown("### ✅ Result")
        st.success("The person is Diabetic." if result[0] == 1 else "The person is NOT Diabetic.")

# Heart Disease Prediction Page
elif selected == "Heart Disease Prediction":
    st.title("❤️ Heart Disease Prediction")

    # Input fields
    age = st.slider("Age", 20, 100, 45)
    sex = st.selectbox("Sex", ["Male", "Female"])
    cp = st.selectbox("Chest Pain Type", [0, 1, 2, 3])
    trestbps = st.slider("Resting Blood Pressure", 80, 200, 120)
    chol = st.slider("Cholesterol", 100, 600, 200)
    fbs = st.selectbox("Fasting Blood Sugar > 120", [0, 1])
    restecg = st.selectbox("Resting ECG Results", [0, 1, 2])
    thalach = st.slider("Max Heart Rate Achieved", 60, 220, 150)
    exang = st.selectbox("Exercise Induced Angina", [0, 1])
    oldpeak = st.slider("ST Depression", 0.0, 6.0, 1.0)
    slope = st.selectbox("Slope of ST", [0, 1, 2])
    ca = st.selectbox("Number of Major Vessels", [0, 1, 2, 3, 4])
    thal = st.selectbox("Thalassemia", [0, 1, 2, 3])

    input_data = np.array([[age, 1 if sex == "Male" else 0, cp, trestbps, chol, fbs,
                            restecg, thalach, exang, oldpeak, slope, ca, thal]])

    # Visualization
    heart_features = {
        "Age": age,
        "BP": trestbps,
        "Cholesterol": chol,
        "Max HR": thalach,
        "ST Depression": oldpeak
    }

    st.markdown("### 📈 Patient Input Overview")
    fig3 = go.Figure([go.Bar(x=list(heart_features.keys()),
                             y=list(heart_features.values()),
                             marker_color='salmon')])
    fig3.update_layout(title_text='Heart Disease Features', height=400)
    st.plotly_chart(fig3)

    if st.button("Predict Heart Disease"):
        result = heart_disease.predict(input_data)
        st.markdown("### ✅ Result")
        st.success("The person has Heart Disease." if result[0] == 1 else "The person does NOT have Heart Disease.")

# Parkinson's Prediction Page
elif selected == "Parkinson's Prediction":
    st.title("🧠 Parkinson's Disease Prediction")

    fo = st.number_input("MDVP:Fo(Hz)")
    fhi = st.number_input("MDVP:Fhi(Hz)")
    flo = st.number_input("MDVP:Flo(Hz)")
    jitter_percent = st.number_input("MDVP:Jitter(%)")
    shimmer = st.number_input("MDVP:Shimmer")
    rap = st.number_input("Shimmer:APQ3")
    dda = st.number_input("Shimmer:DDA")
    nhr = st.number_input("NHR")
    hnr = st.number_input("HNR")
    rpde = st.number_input("RPDE")
    dfa = st.number_input("DFA")
    spread1 = st.number_input("Spread1")
    spread2 = st.number_input("Spread2")
    d2 = st.number_input("D2")
    PPE = st.number_input("PPE")

    input_data = np.array([[fo, fhi, flo, jitter_percent, shimmer, rap, dda, nhr, hnr,
                            rpde, dfa, spread1, spread2, d2, PPE]])

    # Visualization
    parkinson_features = {
        "Fo(Hz)": fo,
        "Fhi(Hz)": fhi,
        "Flo(Hz)": flo,
        "HNR": hnr,
        "PPE": PPE
    }

    st.markdown("### 📈 Patient Input Overview")
    fig4 = go.Figure([go.Bar(x=list(parkinson_features.keys()),
                             y=list(parkinson_features.values()),
                             marker_color='mediumseagreen')])
    fig4.update_layout(title_text="Parkinson's Key Features", height=400)
    st.plotly_chart(fig4)

    if st.button("Predict Parkinson's"):
        result = parkinson_model.predict(input_data)
        st.markdown("### ✅ Result")
        st.success("The person has Parkinson's Disease." if result[0] == 1 else "The person does NOT have Parkinson's Disease.")
