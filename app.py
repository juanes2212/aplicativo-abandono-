import streamlit as st
import pandas as pd
import joblib
import os

st.set_page_config(
    page_title="Predicción de Abandono - Netflix",
    page_icon="🎬",
    layout="wide"
)

st.markdown("""
<style>
    .section-header {
        display: flex;
        align-items: center;
        gap: 10px;
        margin-bottom: 2px;
    }
    .section-badge {
        background-color: #1f6feb;
        color: white;
        font-weight: 700;
        border-radius: 50%;
        width: 26px;
        height: 26px;
        display: inline-flex;
        align-items: center;
        justify-content: center;
        font-size: 0.9rem;
    }
    .section-sub {
        color: #6c757d;
        font-size: 0.9rem;
        margin-bottom: 14px;
    }
    div.stButton > button[kind="primary"] {
        background-color: #2b6b84;
        border: none;
        padding: 0.65rem 1rem;
        font-weight: 600;
    }
    div.stButton > button[kind="primary"]:hover {
        background-color: #22566a;
    }
    .footer-note {
        text-align: center;
        color: #8c959f;
        font-size: 0.82rem;
        margin-top: 2.5rem;
    }
</style>
""", unsafe_allow_html=True)

MODEL_PATH = "netflix_churn_model.pkl"

PLAN_PRICES_COP = {
    "Ads": 18900,
    "Standard": 29900,
    "Premium": 44900,
}
COP_TO_MODEL_USD = {
    18900: 7.99,
    29900: 15.49,
    44900: 22.99,
}
PRICE_OPTIONS_COP = [18900, 29900, 44900]

def format_cop(valor: int) -> str:
    return f"${valor:,.0f} COP".replace(",", ".")

if "plan_actual" not in st.session_state:
    st.session_state.plan_actual = "Standard"
if "precio_mensual_cop" not in st.session_state:
    st.session_state.precio_mensual_cop = PLAN_PRICES_COP["Standard"]

def sync_price_with_plan():
    st.session_state.precio_mensual_cop = PLAN_PRICES_COP[st.session_state.plan_actual]

@st.cache_resource
def load_model(path: str):
    if os.path.exists(path):
        try:
            return joblib.load(path)
        except Exception as e:
            st.error(f"Error al cargar el modelo en `{path}`: {e}")
            return None
    return None

model = load_model(MODEL_PATH)

st.title("🎬 Predicción de Abandono de Suscriptores")
st.caption("Evalúa el riesgo de cancelación a partir del perfil de cuenta, plan y hábitos de visualización.")

# Selector de plan y precio en COP fuera del form para que el precio cambie al instante
with st.container(border=True):
    st.markdown('<div class="section-header"><span class="section-badge">1</span><h4 style="margin:0;">Perfil y plan de suscripción</h4></div>', unsafe_allow_html=True)
    st.markdown('<div class="section-sub">Selecciona el plan y su tarifa mensual en pesos colombianos (COP); el precio se ajusta automáticamente al cambiar de plan.</div>', unsafe_allow_html=True)

    p1, p2 = st.columns(2)
    plan = p1.selectbox(
        "Plan actual",
        options=["Ads", "Standard", "Premium"],
        key="plan_actual",
        on_change=sync_price_with_plan,
        format_func=lambda x: {"Ads": "Con anuncios", "Standard": "Estándar", "Premium": "Premium"}[x]
    )
    precio_cop = p2.selectbox(
        "Precio mensual (COP)",
        options=PRICE_OPTIONS_COP,
        key="precio_mensual_cop",
        format_func=format_cop
    )
    precio_modelo = COP_TO_MODEL_USD[precio_cop]

with st.form("churn_form"):
    # Resto de SECCIÓN 1: Perfil y suscripción
    with st.container(border=True):
        c1, c2, c3, c4 = st.columns(4)
        edad = c1.number_input("Edad", min_value=18, max_value=100, value=35, step=1)
        ingresos = c2.selectbox(
            "Nivel de ingresos",
            options=["Low", "Mid", "High"],
            index=1,
            format_func=lambda x: {"Low": "Bajo", "Mid": "Medio", "High": "Alto"}[x]
        )
        antiguedad = c3.number_input("Antigüedad (meses)", min_value=0, max_value=120, value=18, step=1)
        perfiles = c4.number_input("Perfiles activos", min_value=1, max_value=10, value=2, step=1)

        c5, t1, t2, t3 = st.columns(4)
        tendencia = c5.selectbox(
            "Tendencia de uso",
            options=["Stable", "Increasing", "Declining"],
            index=0,
            format_func=lambda x: {"Stable": "Estable", "Increasing": "Creciente", "Declining": "Decreciente"}[x]
        )
        aumento_precio = 1 if t1.toggle("Aumento reciente de precio", value=False) else 0
        baja_plan = 1 if t2.toggle("Cambio a plan inferior (Downgrade)", value=False) else 0
        maraton = 1 if t3.toggle("Hábito de maratón (Binge)", value=False) else 0

    # SECCIÓN 2: Consumo y contenido
    with st.container(border=True):
        st.markdown('<div class="section-header"><span class="section-badge">2</span><h4 style="margin:0;">Hábitos de consumo y catálogo</h4></div>', unsafe_allow_html=True)
        st.markdown('<div class="section-sub">Tiempo de reproducción, preferencia de formato y variedad de géneros explorados.</div>', unsafe_allow_html=True)

        v1, v2, v3, v4 = st.columns(4)
        horas = v1.number_input("Horas de visualización", min_value=0, max_value=200, value=8, step=1)
        duracion_sesion = v2.number_input("Duración promedio sesión (h)", min_value=1, max_value=12, value=2, step=1)
        contenido = v3.selectbox(
            "Tipo de contenido",
            options=["Series", "Movies"],
            index=0,
            format_func=lambda x: {"Series": "Series", "Movies": "Películas"}[x]
        )
        diversidad = v4.slider("Diversidad de géneros", min_value=1.0, max_value=10.0, value=5.5, step=0.1)

        originales = st.slider("Proporción de contenido original", min_value=0.0, max_value=1.0, value=0.48, step=0.01)

    submitted = st.form_submit_button("Calcular riesgo de abandono", type="primary", use_container_width=True)

if submitted:
    X_inference = pd.DataFrame([{
        "Edad": edad,
        "Precio": precio_modelo,
        "Aumento_Precio": aumento_precio,
        "Baja_Plan": baja_plan,
        "Horas": horas,
        "Originales": originales,
        "Maraton": maraton,
        "Duracion_Sesion": duracion_sesion,
        "Diversidad": diversidad,
        "Perfiles": perfiles,
        "Antiguedad": antiguedad,
        "Ingresos_High": 1 if ingresos == "High" else 0,
        "Ingresos_Low": 1 if ingresos == "Low" else 0,
        "Ingresos_Mid": 1 if ingresos == "Mid" else 0,
        "Plan_Ads": 1 if plan == "Ads" else 0,
        "Plan_Premium": 1 if plan == "Premium" else 0,
        "Plan_Standard": 1 if plan == "Standard" else 0,
        "Tendencia_Declining": 1 if tendencia == "Declining" else 0,
        "Tendencia_Increasing": 1 if tendencia == "Increasing" else 0,
        "Tendencia_Stable": 1 if tendencia == "Stable" else 0,
        "Contenido_Movies": 1 if contenido == "Movies" else 0,
        "Contenido_Series": 1 if contenido == "Series" else 0,
    }])

    if model is None:
        st.warning(f"No se encontró `{MODEL_PATH}` en el repositorio. Se muestra una estimación de demostración.")
        prob = 0.74 if (horas < 3.0 or tendencia == "Declining" or baja_plan == 1) else 0.22
        pred = 1 if prob >= 0.5 else 0
    else:
        try:
            if hasattr(model, "feature_names_in_"):
                expected_cols = list(model.feature_names_in_)
                for col in expected_cols:
                    if col not in X_inference.columns:
                        X_inference[col] = 0
                X_inference = X_inference[expected_cols]

            pred = int(model.predict(X_inference)[0])
            if hasattr(model, "predict_proba"):
                prob = float(model.predict_proba(X_inference)[0][1])
            else:
                prob = float(pred)
        except Exception as e:
            st.error(f"Error al generar la predicción con el modelo: {e}")
            st.stop()

    pct = prob * 100

    with st.container(border=True):
        st.subheader("Resultado del análisis")
        r1, r2 = st.columns([2, 1])
        with r1:
            if pct >= 50:
                st.error(f"🚨 **Hay riesgo de abandono ({pct:.1f}%) — Alta probabilidad de cancelación**")
                recomendacion = (
                    "Se detecta un riesgo alto de abandono. Conviene activar campañas de retención, "
                    "ofrecer beneficios o ajustes de plan y recomendar estrenos alineados a sus gustos."
                )
            elif pct >= 30:
                st.warning(f"⚠️ **Riesgo moderado de abandono ({pct:.1f}%) — Suscriptor en observación**")
                recomendacion = (
                    "La probabilidad no supera el umbral crítico, pero merece seguimiento preventivo: "
                    "envía recomendaciones personalizadas y vigila si disminuye su tiempo de visualización."
                )
            else:
                st.success(f"✅ **Sin riesgo significativo de abandono ({pct:.1f}%) — Suscriptor estable**")
                recomendacion = (
                    "El suscriptor muestra un comportamiento saludable y baja probabilidad de cancelar. "
                    "Mantén una experiencia fluida y novedades relevantes de su catálogo preferido."
                )
        with r2:
            st.metric("Probabilidad estimada de Churn", f"{pct:.2f}%")

        st.info(f"💡 **Recomendación:** {recomendacion}")

st.markdown('<div class="footer-note">Proyecto de predicción de abandono de suscriptores · Streamlit</div>', unsafe_allow_html=True)
