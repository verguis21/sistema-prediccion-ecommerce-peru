import streamlit as st
import pandas as pd
import joblib
import numpy as np
import time

# --- 1. Configuración de la página ---
st.set_page_config(
    page_title="Predicción Logística UPAO",
    page_icon="🚚",
    layout="centered" # "centered" se ve mejor en móviles y laptops que "wide" para este contenido
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

# --- 3. Cabecera ---
st.image("https://upload.wikimedia.org/wikipedia/commons/2/23/Logo_UPAO.png", width=150)
st.title("🚚 Predicción de Envíos E-Commerce")
st.markdown("### Sistema Inteligente de Detección de Retrasos")
st.info("Ingrese los detalles del pedido en el panel lateral para evaluar el riesgo logístico.")

# --- 4. Panel Lateral (Inputs) ---
st.sidebar.header("📦 Configuración del Pedido")
st.sidebar.markdown("---")

# Opciones de Datos
ciudades = ['Lima', 'Trujillo', 'Arequipa', 'Cusco', 'Piura', 'Chiclayo']
estados = ['LIMA', 'LA_LIBERTAD', 'AREQUIPA', 'CUSCO', 'PIURA', 'LAMBAYEQUE']
pagos = ['credit_card', 'boleto', 'voucher', 'debit_card']
categorias = ['utilidades_domesticas', 'perfumaria', 'automotivo', 'bebes', 'relogios_presentes']

# Inputs
customer_state = st.sidebar.selectbox("Región de Destino", estados)
customer_city = st.sidebar.selectbox("Ciudad de Destino", ciudades)
st.sidebar.markdown("---")
seller_state = st.sidebar.selectbox("Región de Origen (Vendedor)", estados)
seller_city = st.sidebar.selectbox("Ciudad de Origen", ciudades)
st.sidebar.markdown("---")
payment_type = st.sidebar.selectbox("Método de Pago", pagos)
product_cat = st.sidebar.selectbox("Categoría", categorias)

# Datos numéricos con mejor formato
price = st.sidebar.number_input("Precio (S/.)", min_value=0.0, value=120.0, step=10.0)
freight = st.sidebar.number_input("Flete (S/.)", min_value=0.0, value=30.0, step=5.0)
weight = st.sidebar.number_input("Peso (gramos)", min_value=0, value=800, step=100)

with st.sidebar.expander("📏 Dimensiones del Paquete"):
    length = st.slider("Largo (cm)", 0, 100, 30)
    height = st.slider("Alto (cm)", 0, 100, 10)
    width = st.slider("Ancho (cm)", 0, 100, 20)

# --- 5. Lógica Auxiliar (Tiempos Estimados por Región) ---
def get_estimated_time(region):
    # Lógica heurística para mostrar dinamismo
    if region == 'LIMA':
        return "1 - 3 días"
    elif region in ['LA_LIBERTAD', 'LAMBAYEQUE', 'PIURA']:
        return "3 - 5 días (Costa Norte)"
    else:
        return "5 - 8 días (Sierra/Sur)"

# --- 6. Botón de Acción ---
if st.button("🚀 Analizar Riesgo Ahora", type="primary", use_container_width=True):
    
    # Efecto de carga
    with st.spinner('Procesando datos con Random Forest...'):
        time.sleep(1) # Pequeña pausa dramática para que se vea el efecto
        
        # Construir DataFrame
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
        
        # Preprocesamiento
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
        
        # --- 7. Mostrar Resultados ---
        st.divider()
        st.subheader("📊 Resultados del Análisis")
        
        col_res1, col_res2, col_res3 = st.columns(3)
        
        # Columna 1: Decisión del Modelo
        with col_res1:
            if prediction == 1:
                st.error("⚠️ **ALTO RIESGO**")
                st.write("Se pronostica **RETRASO**.")
            else:
                st.success("✅ **A TIEMPO**")
                st.write("Entrega puntual probable.")
        
        # Columna 2: Probabilidad
        with col_res2:
            st.metric("Probabilidad de Retraso", f"{probability:.1%}")
        
        # Columna 3: Tiempo Estimado (Dinamismo extra)
        with col_res3:
            tiempo_est = get_estimated_time(customer_state)
            st.metric("Tiempo Estándar", tiempo_est)
            if prediction == 1:
                st.caption("⚠️ Posible adición de +2 días por riesgo detectado.")

        # Barra de progreso visual
        st.write("Nivel de Riesgo:")
        color_bar = "red" if probability > 0.5 else "green"
        st.progress(int(probability * 100))
        
        st.toast("Cálculo finalizado exitosamente", icon="✅")

        # --- 8. Tabla de Datos (Desplegable Post-Análisis) ---
        with st.expander("🔎 Ver detalles técnicos de los datos procesados"):
            st.dataframe(input_df)
            st.caption("Estos datos fueron vectorizados y analizados por el modelo en tiempo real.")

# --- 9. Footer Minimalista y Desplegable ---
st.markdown("<br><br>", unsafe_allow_html=True)
st.divider()

# Footer visible simple
col_f1, col_f2 = st.columns([3, 1])
with col_f1:
    st.caption("© 2025-II | Universidad Privada Antenor Orrego")
with col_f2:
    st.caption("v1.0.0 Stable")

# Ventana desplegable con los créditos (Lo que pediste)
with st.expander("ℹ️ Créditos del Proyecto y Equipo"):
    st.markdown("""
    ### 🏫 Cómputo Distribuido y Paralelo
    **Facultad de Ingeniería - UPAO**
    
    **Docente:**
    * Ing. Elías Santa Cruz
    
    **Equipo de Desarrollo (Autores):**
    * 👨‍💻 Cortez Acon, Jonaiker
    * 👨‍💻 Flores Rodriguez, Diego
    * 👨‍💻 Lopez Gonzalez, Jorge
    * 👨‍💻 Ventura Florian, Steffano
    * 👨‍💻 Vergaray Colonia, Jose
    
    **Stack Tecnológico:**
    * Python, Scikit-Learn, Apache Spark, Streamlit.
    """)
