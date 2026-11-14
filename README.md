# Predicción de Abandono (Churn) de Suscriptores de Netflix 🎬

Aplicación web interactiva desarrollada con **Python** y **Streamlit** orientada a predecir el abandono (Churn) de usuarios de servicios de streaming.

---

## 📋 Supuestos y Decisiones Técnicas

1. **Variable Objetivo (`Churn`)**:
   - `0`: El suscriptor permanece activo.
   - `1`: El suscriptor cancela/abandona el servicio. No interviene como variable predictora de entrada.
2. **Identificador (`User_ID`)**:
   - Se solicita en la interfaz únicamente con fines de registro o auditoría del usuario evaluado, pero **se excluye** del vector de características (`features`) ingresado al modelo predictivo, ya que un código alfanumérico aleatorio no aporta valor de generalización estadística.
3. **Formato del Modelo**:
   - Se asume un pipeline serializado de Scikit-Learn (`.pkl` o `.joblib`) ubicado en la ruta `model/netflix_churn_model.pkl`. Si el archivo no está presente, la interfaz implementa un comportamiento de respaldo (fallback) demostrativo.
4. **Mapeo de Etiquetas**:
   - Los nombres originales de las columnas del dataset se han mapeado a etiquetas claras en español en los controles del formulario de Streamlit.

---

## 🛠️ Instalación y Ejecución Local

1. Clona el repositorio y sitúate en su carpeta raíz.
2. Crea y activa un entorno virtual:
   ```bash
   python -m venv venv
   source venv/bin/activate  # En Windows: venv\Scripts\activate
