import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os

# Configuración de la página
st.set_page_config(
    page_title="Predicción de Abandono - Netflix",
    page_icon="🎬",
    layout="centered"
)

# Cargar el modelo con caché de recursos para mayor eficiencia
@st.cache_resource
def load_model():
    model_path = "model/netflix_churn_model.pkl"
    if os.path.exists(model_path):
        try:
            return joblib.load(model_path)
        except Exception as e:
            st.error(f"Error al cargar el modelo: {e}")
            return None
    return None

model = load_model()

# Título y descripción
st.title("🎬 Predicción de Abandono (Churn) de Suscriptores")
st.markdown("""
Esta aplicación permite anticipar si un suscriptor de Netflix tiene probabilidad de abandonar la plataforma 
basándose en su comportamiento, nivel de participación y características de su cuenta.
""")

st.sidebar.header("📝 Instrucciones")
st.sidebar.info(
    "Complete los datos del suscriptor en el formulario y presione "
    "**Predecir Abandono** para ver el diagnóstico y la probabilidad estimada."
)

# Formulario de entrada de variables
with st.form("churn_form"):
    st.subheader("Parámetros del Suscriptor y Cuenta")
    
    col1, col2 = st.columns(2)
    
    with col1:
        user_id = st.text_input("User_ID (Identificador)", value="U9999")
        edad = st.number_input("Edad", min_value=18, max_value=100, value=35, step=1)
        ingresos = st.selectbox("Ingresos", options=["Low", "Mid", "High"], index=1)
        plan = st.selectbox("Plan", options=["Ads", "Standard", "Premium"], index=1)
        precio = st.number_input("Precio ($)", min_value=0.0, max_value=100.0, value=19.99, step=0.01)
        aumento_precio = st.selectbox("Aumento_Precio", options=[0, 1], format_func=lambda x: "Sí (1)" if x == 1 else "No (0)")
        baja_plan = st.selectbox("Baja_Plan", options=[0, 1], format_func=lambda x: "Sí (1)" if x == 1 else "No (0)")
        tendencia = st.selectbox("Tendencia", options=["Stable", "Increasing", "Declining"], index=0)

    with col2:
        horas = st.number_input("Horas (Horas de visualización)", min_value=0.0, max_value=50.0, value=7.5, step=0.1)
        contenido = st.selectbox("Contenido", options=["Series", "Movies"], index=0)
        originales = st.slider("Originales (Proporción)", min_value=0.0, max_value=1.0, value=0.48, step=0.01)
        maraton = st.selectbox("Maraton (Binge)", options=[0, 1], format_func=lambda x: "Sí (1)" if x == 1 else "No (0)")
        duracion_sesion = st.number_input("Duracion_Sesion (Horas)", min_value=0.1, max_value=10.0, value=1.7, step=0.1)
        diversidad = st.slider("Diversidad (Géneros)", min_value=1.0, max_value=10.0, value=5.5, step=0.1)
        perfiles = st.number_input("Perfiles", min_value=1, max_value=10, value=2, step=1)
        antiguedad = st.number_input("Antiguedad (Meses)", min_value=0.0, max_value=120.0, value=18.0, step=0.5)

    submitted = st.form_submit_button("🔍 Predecir Abandono")

if submitted:
    # Construir DataFrame con las columnas originales que espera el modelo
    input_data = pd.DataFrame({
        'User_ID': [user_id],
        'Age': [edad],
        'Income': [ingresos],
        'Plan': [plan],
        'Price': [precio],
        'Price_Increase': [aumento_precio],
        'Downgrade': [baja_plan],
        'Trend': [tendencia],
        'Watch_Hours': [horas],
        'Content_Type': [contenido],
        'Originals': [originales],
        'Binge': [maraton],
        'Session': [duracion_sesion],
        'Diversity': [diversidad],
        'Profiles': [perfiles],
        'Account_Age': [antiguedad]
    })
    
    if model is None:
        st.warning("⚠️ No se encontró el archivo del modelo en `model/netflix_churn_model.pkl`. Se muestra una predicción simulada para demostración.")
        # Simulación de respaldo si no se ha cargado el archivo físico del modelo
        prob = 0.74 if horas < 3.0 or tendencia == "Declining" else 0.22
        pred = 1 if prob > 0.5 else 0
    else:
        try:
            # Excluimos User_ID ya que es solo un identificador y no participa en la predicción
            X_inference = input_data.drop(columns=['User_ID'], errors='ignore')
            pred = model.predict(X_inference)[0]
            prob = model.predict_proba(X_inference)[0][1] # Probabilidad de la clase Churn = 1
        except Exception as e:
            st.error(f"Error al ejecutar la predicción: {e}")
            st.stop()

    st.markdown("---")
    st.subheader("📊 Resultado del Análisis")
    
    col_res1, col_res2 = st.columns(2)
    with col_res1:
        if pred == 1:
            st.error("🚨 **Alerta: Alto riesgo de ABANDONO (Churn = 1)**")
        else:
            st.success("✅ **El suscriptor SE MANTIENE activo (Churn = 0)**")
            
    with col_res2:
        st.metric(label="Probabilidad estimada de Churn", value=f"{prob * 100:.2f}%")
        
    st.markdown("---")
    st.info("💡 **Recomendación comercial:** Si la probabilidad de abandono es elevada, considere ofrecer incentivos de retención, planes promocionales o recomendaciones orientadas a sus géneros favoritos.")
