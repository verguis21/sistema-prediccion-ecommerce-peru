# 🚚 Sistema Inteligente Distribuido para la Predicción de Entregas Tardías

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://sistema-prediccion-ecommerce-peru-5ybgfegjgvbhfyulaxq6b3.streamlit.app/)

> **Un sistema de Machine Learning capaz de anticipar riesgos logísticos en el comercio electrónico peruano.**

---

## 🔗 Demo en Vivo
¡Prueba la aplicación ahora mismo sin instalar nada!
👉 **[Click aquí para acceder al Sistema de Predicción](https://sistema-prediccion-ecommerce-peru-5ybgfegjgvbhfyulaxq6b3.streamlit.app/)**

---

## 📖 Introducción
En el competitivo mundo del e-commerce, la puntualidad es clave. Este proyecto aborda la problemática de los **retrasos en las entregas** utilizando datos históricos transaccionales.

Hemos desarrollado un modelo predictivo (**Random Forest**) validado mediante técnicas de **Cómputo Distribuido (Apache Spark)** para identificar patrones de riesgo basados en la ubicación del cliente, el vendedor, y las características físicas del producto. El resultado es una herramienta interactiva que permite a los gestores logísticos tomar decisiones proactivas.

---

## 🏫 Información Académica

**Universidad Privada Antenor Orrego (UPAO)**
*Facultad de Ingeniería - Escuela Profesional de Ingeniería de Computación y Sistemas*

* **Curso:** Cómputo Distribuido y Paralelo
* **Semestre:** 2025-II
* **Docente:** Ing. Elías Santa Cruz

### 👥 Equipo de Desarrollo
* Cortez Acon, Jonaiker
* Flores Rodriguez, Diego
* Lopez Gonzalez, Jorge
* Ventura Florian, Steffano
* Vergaray Colonia, Jose

---

## 🛠️ Tecnologías y Arquitectura

El sistema se construyó utilizando un flujo de trabajo moderno de Ciencia de Datos:

1.  **Procesamiento de Datos:** `Pandas` para manipulación local y `PySpark` para simulación de carga distribuida.
2.  **Modelado (Machine Learning):** `Scikit-Learn` (Random Forest Classifier) optimizado para balance de clases.
3.  **Serialización:** `Joblib` para la persistencia del modelo entrenado.
4.  **Despliegue (Frontend):** `Streamlit` para la interfaz de usuario web.
5.  **Infraestructura:** Alojado en Streamlit Community Cloud.

---

## 📊 Características del Sistema

* **Predicción en Tiempo Real:** Análisis instantáneo de nuevos pedidos.
* **Filtros Inteligentes:** Selección dinámica de regiones y ciudades del Perú (Lima, Trujillo, Arequipa, Cusco, etc.).
* **Explicabilidad:** Gráfico de importancia de variables para entender *por qué* se pronostica un retraso.
* **Estimación de Tiempos:** Cálculo heurístico de tiempos de entrega estándar según la región.

---

## 🚀 Ejecución Local (Opcional)

Si deseas correr este proyecto en tu propia máquina:

1.  **Clonar el repositorio:**
    ```bash
    git clone [https://github.com/verguis21/sistema-prediccion-ecommerce-peru.git](https://github.com/verguis21/sistema-prediccion-ecommerce-peru.git)
    ```

2.  **Instalar dependencias:**
    ```bash
    pip install -r requirements.txt
    ```

3.  **Lanzar la aplicación:**
    ```bash
    streamlit run app.py
    ```

---
*Trujillo, Perú - Diciembre 2025*
