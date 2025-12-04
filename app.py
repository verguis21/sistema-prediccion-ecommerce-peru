import streamlit as st
import pandas as pd
import joblib
import numpy as np

# --- 1. Configuración de la página ---
st.set_page_config(
    page_title="Predicción Logística E-Commerce",
    page_icon="🚚",
    layout="wide"
)

# --- 2. Cargar el modelo y las columnas ---
@st.cache_resource
def load_model():
    model = joblib.load('modelo_entregas_peru.pkl')
    cols = joblib.load('modelo_entregas_columnas.pkl')
    return model, cols

try:
    rf_model, model_columns = load_model()
    st.success("Sistema de IA cargado correctamente.")
except Exception as e:
    st.error(f"Error al cargar el modelo: {e}")
    st.stop()

# --- 3. Título y Descripción ---
st.title("🚚 Sistema Inteligente de Predicción de Envíos")
st.markdown("""
Esta aplicación utiliza un modelo de **Random Forest** entrenado con datos históricos
para predecir la probabilidad de retraso en pedidos de un E-Commerce Peruano.
""")
st.divider()

# --- 4. Panel de Inputs (Barra Lateral) ---
st.sidebar.header("📝 Parámetros del Pedido")

# Opciones
ciudades = ['Lima', 'Trujillo', 'Arequipa', 'Cusco', 'Piura', 'Chiclayo']
estados = ['LIMA', 'LA_LIBERTAD', 'AREQUIPA', 'CUSCO', 'PIURA', 'LAMBAYEQUE']
pagos = ['credit_card', 'boleto', 'voucher', 'debit_card']
categorias = ['utilidades_domesticas', 'perfumaria', 'automotivo', 'bebes', 'relogios_presentes']

# Inputs del usuario
customer_city = st.sidebar.selectbox("Ciudad del Cliente", ciudades)
customer_state = st.sidebar.selectbox("Región del Cliente", estados)
seller_city = st.sidebar.selectbox("Ciudad del Vendedor", ciudades)
seller_state = st.sidebar.selectbox("Región del Vendedor", estados)
payment_type = st.sidebar.selectbox("Método de Pago", pagos)
product_cat = st.sidebar.selectbox("Categoría del Producto", categorias)

price = st.sidebar.number_input("Precio del Producto (S/.)", min_value=0.0, value=120.0)
freight = st.sidebar.number_input("Costo de Envío (S/.)", min_value=0.0, value=30.0)
weight = st.sidebar.number_input("Peso (g)", min_value=0, value=800)

length = st.sidebar.slider("Largo (cm)", 0, 100, 30)
height = st.sidebar.slider("Alto (cm)", 0, 100, 10)
width = st.sidebar.slider("Ancho (cm)", 0, 100, 20)

# --- 5. Creación del DataFrame (ESTO ES LO QUE ARREGLA EL ERROR) ---
# Creamos los datos AQUÍ, fuera del botón, para que siempre existan
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

# --- 6. Mostrar Tabla Interactiva (Ahora sí funciona) ---
with st.expander("📊 Ver datos procesados (Vista Previa)"):
    st.dataframe(input_df)
    st.info("Estos son los datos exactos que el modelo recibirá para el análisis.")

# --- 7. Botón y Lógica de Predicción ---
if st.button("🔍 Analizar Riesgo de Envío", type="primary"):
    
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
    
    # Resultados
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Resultado del Análisis")
        if prediction == 1:
            st.error(f"⚠️ **ALERTA: Retraso Detectado**")
            st.write("Se estima que este pedido llegará tarde.")
        else:
            st.success(f"✅ **Envío A Tiempo**")
            st.write("El sistema no detecta riesgos significativos.")
            
    with col2:
        st.subheader("Probabilidad de Retraso")
        st.metric(label="Probabilidad", value=f"{probability:.2%}")
        st.progress(int(probability * 100))

# --- 8. Footer Académico ---
st.markdown("---")
with st.container():
    col1, col2 = st.columns([1, 3])
    
    with col1:
        # Logo UPAO (opcional, si no carga se ve el texto alternativo)
        st.image("https://upload.wikimedia.org/wikipedia/commons/2/23/Logo_UPAO.png", width=100)
    
    with col2:
        st.markdown("""
        ### Universidad Privada Antenor Orrego
        **Facultad de Ingeniería** - Escuela de Ingeniería de Computación y Sistemas
        
        * **Curso:** Cómputo Distribuido y Paralelo
        * **Semestre:** 2025-II
        * **Docente:** Ing. Elías Santa Cruz
        * **Proyecto:** Sistema Inteligente Distribuido para la Predicción de Entregas Tardías
        """)
