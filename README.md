# Proyecto Final Máster - E-commerce Analytics Dashboard

Aplicación analítica desarrollada como parte del Trabajo Final de Máster.  
El proyecto integra procesos de limpieza y transformación de datos, segmentación de clientes, análisis de sentimiento de reseñas, sistema de recomendación y una demo interactiva de tienda online.

## Objetivo del proyecto

El objetivo es construir una solución analítica para datos de e-commerce que permita:

- Analizar indicadores generales de negocio.
- Explorar ventas por categoría.
- Segmentar clientes según comportamiento de compra.
- Analizar automáticamente el sentimiento de reseñas.
- Recomendar productos a clientes.
- Simular la llegada de nuevas compras y reseñas en una tienda online.
- Visualizar eventos demo en tiempo real dentro del dashboard.

## Tecnologías utilizadas

- Python
- Pandas
- Scikit-learn
- Joblib
- Streamlit
- Plotly
- Spark / PySpark
- RapidMiner

## Estructura del proyecto

```text
ProyectoFinalMaster/
│
├── app/
│   └── streamlit_app.py
│
├── data/
│   ├── dataset_sales.csv
│   ├── dataset_reviews.csv
│   ├── reviews_nlp.csv
│   ├── reviews_nlp_rapidminer.csv
│   ├── rfm_customers.csv
│   ├── rfm_customers_clustered_k3.csv
│   ├── rfm_customers_original_clustered_k3.csv
│   ├── cluster_summary_k3.csv
│   └── category_sales.csv
│
├── models/
│   └── sentiment_pipeline.pkl
│
├── notebooks/
│   ├── 01_train_sentiment_model.ipynb
│   └── 02_build_recommender.ipynb
│
├── outputs/
│   ├── figures/
│   ├── metrics/
│   ├── recommendations/
│   │   ├── customer_recommendations.csv
│   │   ├── customer_favorite_category.csv
│   │   └── global_product_recommendations.csv
│   └── live_events/
│       ├── demo_orders.csv
│       └── demo_reviews.csv
│
├── requirements.txt
└── README.md
```

## Módulos principales de la aplicación

### 1. Inicio

Pantalla resumen con indicadores generales del negocio:

- Pedidos históricos.
- Clientes históricos.
- Productos distintos.
- Ingresos históricos.
- Ventas por categoría.
- Flujo general del proyecto.

### 2. Demo tienda

Simula una tienda online básica.  
Permite generar una compra y una reseña nueva. La reseña se analiza automáticamente mediante el modelo NLP entrenado.

### 3. Eventos en vivo

Muestra los eventos generados desde la demo de tienda:

- Compras demo.
- Reseñas demo.
- Ingresos demo.
- Porcentaje de reseñas negativas.
- Últimas reseñas analizadas.

### 4. Simulador nuevo cliente

Permite introducir datos de un cliente nuevo y estimar a qué segmento se parece más.  
También genera recomendaciones de productos según la categoría de interés seleccionada.

### 5. Segmentación de clientes

Muestra los segmentos obtenidos mediante clustering a partir de variables de comportamiento del cliente:

- Recencia.
- Frecuencia.
- Valor monetario.
- Valoración media.

### 6. Análisis de sentimiento

Permite introducir una reseña y clasificarla como positiva o negativa.  
El modelo utilizado es un pipeline basado en TF-IDF y Regresión Logística.

### 7. Recomendador

Muestra productos recomendados para cada cliente según su categoría favorita y productos populares/bien valorados.

### 8. Logística y satisfacción

Analiza la relación entre retrasos en entregas y valoración media de los clientes.

## Instalación

Crear un entorno virtual:

```bash
python -m venv .venv
```

Activar el entorno virtual en Windows:

```bash
.venv\Scripts\activate
```

Instalar dependencias:

```bash
pip install -r requirements.txt
```

## Ejecución de la aplicación

Desde la raíz del proyecto:

```bash
python -m streamlit run app/streamlit_app.py
```

## Modelo de sentimiento

El modelo de sentimiento se entrena en el notebook:

```text
notebooks/01_train_sentiment_model.ipynb
```

El modelo entrenado se guarda en:

```text
models/sentiment_pipeline.pkl
```

## Sistema de recomendación

El recomendador se genera en el notebook:

```text
notebooks/02_build_recommender.ipynb
```

Los resultados se guardan en:

```text
outputs/recommendations/
```

## Simulación de eventos

La demo de tienda genera eventos locales en:

```text
outputs/live_events/demo_orders.csv
outputs/live_events/demo_reviews.csv
```

Estos ficheros simulan la llegada de nuevos datos en un entorno de producción.

## Nota sobre producción

En esta versión, la persistencia de eventos demo se realiza mediante ficheros CSV locales.  
En un entorno real, esta capa podría sustituirse por una base de datos, una API o un sistema de mensajería de eventos.

## Autor

Proyecto desarrollado para el Trabajo Final de Máster.