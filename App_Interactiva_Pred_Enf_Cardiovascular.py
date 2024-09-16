#!/usr/bin/env python
# coding: utf-8

# In[ ]:

# Importar librerías
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.svm import SVC
import joblib

# Función para cargar el modelo pre-entrenado y scaler
def load_model():
    # Cargar el modelo y el scaler que utilizaste durante el entrenamiento
    voting_clf_final = joblib.load('voting_clf_final.pkl')
    scaler = joblib.load('scaler.pkl')
    return voting_clf_final, scaler

# Cargar el modelo y el scaler
model, scaler = load_model()

# Título de la aplicación con estilo
st.markdown("<h1 style='text-align: center; color: #4682B4;'>TFM-UCM: Predictor de Enfermedad Coronaria</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; color: #A9A9A9;'>Ingrese los datos del paciente para predecir el riesgo de enfermedad coronaria utilizando un modelo basado en aprendizaje automático.</p>", unsafe_allow_html=True)

# Usar la barra lateral para los datos del paciente
st.sidebar.header("Datos Médicos del Paciente")

# Cargar una imagen
st.sidebar.image('Fondo_Negro.jpeg', width=270)

# Input para el nombre y apellido
nombre = st.sidebar.text_input("Nombre")
apellido = st.sidebar.text_input("Apellido")

# Entrada de datos del usuario en la barra lateral
sex = st.sidebar.selectbox("Sexo", [0, 1], format_func=lambda x: ["0 = Mujer", "1 = Hombre"][x])
age = st.sidebar.number_input("Edad", 0, 120, 0)
cp = st.sidebar.selectbox("Tipo de dolor en el pecho", [0, 1, 2, 3], format_func=lambda x: ["1 = Angina típica", "2 = Angina atípica", "3 = Dolor no anginoso", "4 = Asintomático"][x])
trestbps = st.sidebar.number_input("Presión arterial en reposo", 0, 200, 0)
chol = st.sidebar.number_input("Colesterol", 0, 564, 0)
restecg = st.sidebar.selectbox("Resultados electrocardiográficos", [0, 1, 2], format_func=lambda x: ["0 = Normal", "1 = Anormalidad ST-T", "2 = Hipertrofia ventricular izquierda"][x])
thalach = st.sidebar.number_input("Frecuencia cardíaca máxima alcanzada", 0, 202, 0)
exang = st.sidebar.selectbox("Angina inducida por el ejercicio", [0, 1], format_func=lambda x: ["0 = No", "1 = Sí"][x])
oldpeak = st.sidebar.number_input("Depresión del ST inducida por el ejercicio", 0.0, 6.2, 1.0)
slope_options = {1: "1 = Pendiente ascendente", 2: "2 = Pendiente plana", 3: "3 = Pendiente descendente"}
slope = st.sidebar.selectbox("Pendiente del segmento ST", [1, 2, 3], format_func=lambda x: slope_options[x])
ca = st.sidebar.number_input("Número de vasos coloreados por fluoroscopía", 0, 3, 1)
thal_options = {3: "3 = Normal", 6: "6 = Defecto fijo", 7: "7 = Defecto reversible"}
thal = st.sidebar.selectbox("Resultado de la prueba de talasemia", [3, 6, 7], format_func=lambda x: thal_options[x])


# Crear un array con los datos del paciente
paciente = [[age, sex, cp, trestbps, chol, restecg, thalach, exang, oldpeak, slope, ca, thal]]

# Escalar los datos del paciente
paciente_scaled = scaler.transform(paciente)

# Estilizar el botón de predicción con CSS
st.markdown("""
    <style>
    div.stButton > button:first-child {
        background-color: #E5E4E2;
        color: black;
        border-radius: 6px;
        height: 3em;
        width: 14em;
        font-size: 20px;
        font-weight: 1200;
        text-transform: uppercase;
    }
    div.stButton > button:first-child:hover {
        background-color: #4682B4;
        color: #E5E4E2;
    }
    </style>
""", unsafe_allow_html=True)

# Botón para la predicción con color personalizado
if st.button("Predicción"):
    # Hacer la predicción con el modelo cargado
    prediccion = model.predict(paciente_scaled)
    proba = model.predict_proba(paciente_scaled)

    # Mostrar los resultados con formato y estilo
    st.markdown("<h2 style='text-align: center;'>Resultados de la Predicción</h2>", unsafe_allow_html=True)

    # Formatear el nombre completo del paciente
    nombre_completo = f"{nombre} {apellido}".strip()

    if prediccion[0] == 1:
        st.markdown(f"<h3 style='color: #DC143C; text-align: center;'>El modelo predice que el paciente <b style='color: #4169E1;'>{nombre_completo}</b> tiene <b>Riesgo</b> de padecer enfermedad coronaria.</h3>", unsafe_allow_html=True)
    else:
        st.markdown(f"<h3 style='color: #2E8B57; text-align: center;'>El modelo predice que el paciente <b style='color: #4169E1;'>{nombre_completo}</b> <b>No tiene Riesgo</b> de padecer enfermedad coronaria.</h3>", unsafe_allow_html=True)

    # Mostrar las probabilidades en una tabla
    st.markdown("<h3 style='text-align: center;'>Probabilidades</h3>", unsafe_allow_html=True)
    
    # Convertir las probabilidades a porcentajes cerrados
    proba_percent = (proba * 100).astype(int)
    
    # Crear un DataFrame para mostrar las probabilidades
    prob_df = pd.DataFrame(proba_percent, columns=["No Riesgo (%)", "Riesgo (%)"])
    prob_df["Clase Predicha"] = ["No Riesgo" if pred == 0 else "Riesgo" for pred in prediccion]
    
    # Ordenar las columnas de la tabla
    prob_df = prob_df[["Clase Predicha", "No Riesgo (%)", "Riesgo (%)"]]

    # Estilizar la tabla con CSS
    st.markdown("""
    <style>
    table {
        width: 100%;
        border-collapse: collapse;
    }
    th, td {
        border: 1px solid #dddddd;
        text-align: center;
        padding: 8px;
    }
    tr:nth-child(even) {
        background-color: #D3D3D3;
    }
    th {
        background-color: #2F4F4F;
        color: white;
    }
    </style>
    """, unsafe_allow_html=True)

    # Mostrar el DataFrame estilizado en HTML
    st.markdown(prob_df.to_html(index=False, justify="center"), unsafe_allow_html=True)
                
        

# Agregar pie de página
st.markdown("<hr>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; color: #808080;'>Desarrollado por Carlos A. Jiménez. © 2024</p>", unsafe_allow_html=True)

