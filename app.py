import streamlit as st
import pandas as pd
import joblib
import numpy as np
import time

# --- 1. Configuración de la página ---
st.set_page_config(
    page_title="Predicción Logística UPAO",
    page_icon="🚚",
    layout="centered"
)

# --- 2. Cargar el modelo ---
@st.cache_resource
def load_model():
    model = joblib.load('modelo_entregas_peru.pkl')
    cols = joblib.load('modelo_entregas_columnas.pkl')
    return model, cols

try:
    rf_model, model_columns = load_model()
except Exception as e:
    st.error(f"Error crítico al cargar el sistema: {e}")
    st.stop()

# --- 3. Lógica de Datos (Diccionarios) ---
# Mapeo para que los menús sean inteligentes
ubicaciones = {
    'LIMA': ['Lima'],
    'LA_LIBERTAD': ['Trujillo'],
    'AREQUIPA': ['Arequipa'],
    'CUSCO': ['Cusco'],
    'PIURA': ['Piura'],
    'LAMBAYEQUE': ['Chiclayo']
}

pagos = ['credit_card', 'boleto', 'voucher', 'debit_card']
categorias = ['utilidades_domesticas', 'perfumaria', 'automotivo', 'bebes', 'relogios_presentes']

# --- 4. Cabecera ---
# st.image("https://upload.wikimedia.org/wikipedia/commons/2/23/Logo_UPAO.png", width=150) # Opcional si no carga
st.title("🚚 Predicción de Envíos")
st.markdown("### Sistema Inteligente de Detección de Retrasos")

# --- 5. Panel Lateral (Inputs Inteligentes) ---
st.sidebar.header("📦 Configuración del Pedido")
st.sidebar.markdown("---")

# --- Lógica de Filtros Dinámicos ---
st.sidebar.subheader("📍 Destino (Cliente)")
# 1. Primero elegimos la región
customer_state = st.sidebar.selectbox("Región", list(ubicaciones.keys()))
# 2. Las ciudades se filtran según la región elegida arriba
customer_city = st.sidebar.selectbox("Ciudad", ubicaciones[customer_state])

st.sidebar.markdown("---")

st.sidebar.subheader("🏭 Origen (Vendedor)")
seller_state = st.sidebar.selectbox("Región Vendedor", list(ubicaciones.keys()), index=1) # Default: La Libertad
seller_city = st.sidebar.selectbox("Ciudad Vendedor", ubicaciones[seller_state])

st.sidebar.markdown("---")

payment_type = st.sidebar.selectbox("Método de Pago", pagos)
product_cat = st.sidebar.selectbox("Categoría", categorias)

# Datos numéricos
price = st.sidebar.number_input("Precio (S/.)", min_value=0.0, value=120.0, step=10.0)
freight = st.sidebar.number_input("Flete (S/.)", min_value=0.0, value=30.0, step=5.0)
weight = st.sidebar.number_input("Peso (gramos)", min_value=0, value=800, step=100)

with st.sidebar.expander("📏 Dimensiones (Opcional)"):
    length = st.slider("Largo (cm)", 0, 100, 30)
    height = st.slider("Alto (cm)", 0, 100, 10)
    width = st.slider("Ancho (cm)", 0, 100, 20)

# --- 6. Creación del DataFrame (EN TIEMPO REAL) ---
# Creamos los datos fuera del botón para mostrarlos siempre
input_data = {
    "payment_value": price + freight,
    "price": price,
    "freight_value": freight,
    "product_weight_g": weight,
    "product_length_cm": length,
    "product_height_cm": height,
    "product_width_cm": width,
    "customer_city": customer_city,
    "customer_state": customer_state,
    "seller_city": seller_city,
    "seller_state": seller_state,
    "payment_type": payment_type,
    "product_category_name": product_cat
}
input_df = pd.DataFrame([input_data])

# --- 7. Visualización Previa de Datos ---
# Esto aparece ANTES de presionar el botón, como pediste
with st.expander("👁️ Ver datos que serán analizados (Vista Previa)", expanded=False):
    st.dataframe(input_df)
    st.caption("El modelo vectorizará estos datos para realizar la inferencia.")

# --- 8. Lógica Auxiliar (Tiempos) ---
def get_estimated_time(region):
    if region == 'LIMA': return "1 - 3 días"
    elif region in ['LA_LIBERTAD', 'LAMBAYEQUE', 'PIURA']: return "3 - 5 días (Costa Norte)"
    else: return "5 - 8 días (Sierra/Sur)"

# --- 9. Botón de Acción ---
if st.button("🚀 Analizar Riesgo Ahora", type="primary", use_container_width=True):
    
    with st.spinner('El modelo Random Forest está calculando probabilidades...'):
        time.sleep(1) # Efecto visual
        
        # Preprocesamiento interno
        cols_num = ["payment_value", "price", "freight_value", "product_weight_g", 
                    "product_length_cm", "product_height_cm", "product_width_cm"]
        cols_cat = ["customer_city", "customer_state", "seller_city", "seller_state", 
                    "payment_type", "product_category_name"]
        
        X_num = input_df[cols_num]
        X_cat = pd.get_dummies(input_df[cols_cat])
        X_final = pd.concat([X_num, X_cat], axis=1)
        X_final = X_final.reindex(columns=model_columns, fill_value=0)
        
        # Predicción
        prediction = rf_model.predict(X_final)[0]
        probability = rf_model.predict_proba(X_final)[0][1]
        
        # --- 10. Mostrar Resultados Principales ---
        st.divider()
        st.markdown("### 📊 Resultado del Diagnóstico")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**Predicción:**")
            if prediction == 1:
                st.error("⚠️ **SE PRONOSTICA RETRASO**")
                st.write("El pedido tiene alto riesgo de incumplimiento.")
            else:
                st.success("✅ **ENTREGA A TIEMPO**")
                st.write("Flujo logístico normal.")
        
        with col2:
            st.markdown("**Tiempo Estimado (Sin Incidencias):**")
            tiempo = get_estimated_time(customer_state)
            st.info(f"⏱️ {tiempo}")
            
        # Barra de probabilidad
        st.markdown(f"**Probabilidad de Retraso calculada:** `{probability:.1%}`")
        st.progress(int(probability * 100))

        # --- 11. Gráfico de "Explicabilidad" (NUEVO) ---
        # Aquí mostramos QUÉ variables usó el modelo para decidir
        st.divider()
        st.subheader("🧠 ¿Por qué el modelo tomó esta decisión?")
        st.caption("Factores de mayor peso en la predicción global del modelo:")
        
        # Extraer importancia de características
        importances = rf_model.feature_importances_
        feature_names = model_columns
        
        # Crear DataFrame para el gráfico
        importance_df = pd.DataFrame({
            'Variable': feature_names,
            'Importancia': importances
        }).sort_values(by='Importancia', ascending=False).head(7) # Top 7
        
        # Mostrar gráfico de barras
        st.bar_chart(importance_df.set_index('Variable'), color="#FF4B4B")
        
        st.toast("Análisis completado", icon="✅")

# --- 12. Footer Desplegable ---
st.markdown("<br><br>", unsafe_allow_html=True)
st.divider()

col_f1, col_f2 = st.columns([3, 1])
with col_f1:
    st.caption("© 2025-II | Cómputo Distribuido y Paralelo")
with col_f2:
    st.caption("v2.0 Pro")

with st.expander("ℹ️ Ver Créditos del Equipo y Docente"):
    st.markdown("""
    **Universidad Privada Antenor Orrego**
    
    **Docente:**
    * Ing. Elías Santa Cruz
    
    **Integrantes (Developers):**
    * Cortez Acon, Jonaiker
    * Flores Rodriguez, Diego
    * Lopez Gonzalez, Jorge
    * Ventura Florian, Steffano
    * Vergaray Colonia, Jose
    """)
