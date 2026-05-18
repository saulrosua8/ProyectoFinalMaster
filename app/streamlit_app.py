import streamlit as st
import pandas as pd
import joblib
import os
import plotly.express as px

# =========================
# CONFIGURACIÓN
# =========================
st.set_page_config(
    page_title="E-commerce Analytics Dashboard",
    page_icon="📦",
    layout="wide"
)

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DATA_PATH = os.path.join(BASE_DIR, "data", "dataset_sales.csv")
RFM_PATH = os.path.join(BASE_DIR, "data", "rfm_customers_clustered_k3.csv")
CLUSTER_SUMMARY_PATH = os.path.join(BASE_DIR, "data", "cluster_summary_k3.csv")
MODEL_PATH = os.path.join(BASE_DIR, "models", "sentiment_pipeline.pkl")
RECS_PATH = os.path.join(BASE_DIR, "outputs", "recommendations", "customer_recommendations.csv")
FAV_CAT_PATH = os.path.join(BASE_DIR, "outputs", "recommendations", "customer_favorite_category.csv")
LIVE_EVENTS_DIR = os.path.join(BASE_DIR, "outputs", "live_events")
DEMO_ORDERS_PATH = os.path.join(LIVE_EVENTS_DIR, "demo_orders.csv")
DEMO_REVIEWS_PATH = os.path.join(LIVE_EVENTS_DIR, "demo_reviews.csv")
GLOBAL_RECS_PATH = os.path.join(BASE_DIR, "outputs", "recommendations", "global_product_recommendations.csv")

os.makedirs(LIVE_EVENTS_DIR, exist_ok=True)

# =========================
# CARGA DE DATOS
# =========================
@st.cache_data
@st.cache_data
def load_data():
    sales = pd.read_csv(DATA_PATH)

    rfm = pd.read_csv(
        RFM_PATH,
        sep=";",
        quotechar='"',
        engine="python",
        encoding="utf-8"
    )

    cluster_summary = pd.read_csv(
        CLUSTER_SUMMARY_PATH,
        sep=";",
        quotechar='"',
        engine="python",
        encoding="utf-8"
    )

    recommendations = pd.read_csv(RECS_PATH)
    favorite_category = pd.read_csv(FAV_CAT_PATH)
    global_recommendations = pd.read_csv(GLOBAL_RECS_PATH)

    # Limpiar nombres de columnas por si vienen con comillas
    rfm.columns = rfm.columns.str.replace('"', '', regex=False).str.strip()
    cluster_summary.columns = cluster_summary.columns.str.replace('"', '', regex=False).str.strip()

    return sales, rfm, cluster_summary, recommendations, favorite_category, global_recommendations
@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)


sales, rfm, cluster_summary, recommendations, favorite_category, global_recommendations = load_data()
sentiment_model = load_model()


def append_row_to_csv(path, row_dict):
    row_df = pd.DataFrame([row_dict])

    # Si el archivo no existe o está vacío, se crea desde cero
    if (not os.path.exists(path)) or os.path.getsize(path) == 0:
        row_df.to_csv(path, index=False, encoding="utf-8")
        return

    try:
        existing = pd.read_csv(path)
        updated = pd.concat([existing, row_df], ignore_index=True)
    except pd.errors.EmptyDataError:
        updated = row_df

    updated.to_csv(path, index=False, encoding="utf-8")


def safe_read_csv(path):
    if (not os.path.exists(path)) or os.path.getsize(path) == 0:
        return pd.DataFrame()

    try:
        return pd.read_csv(path)
    except pd.errors.EmptyDataError:
        return pd.DataFrame()


def load_live_events():
    demo_orders = safe_read_csv(DEMO_ORDERS_PATH)
    demo_reviews = safe_read_csv(DEMO_REVIEWS_PATH)
    return demo_orders, demo_reviews

def prepare_cluster_summary_for_simulation(cluster_summary):
    summary = cluster_summary.copy()

    summary.columns = (
        summary.columns
        .str.replace("average\\(", "", regex=True)
        .str.replace("\\)", "", regex=True)
        .str.replace('"', "", regex=False)
        .str.strip()
    )

    return summary

# =========================
# SIDEBAR
# =========================
st.sidebar.title("TFM E-commerce Analytics")

st.sidebar.markdown("""
Aplicación analítica para explorar ventas, clientes, reseñas y recomendaciones
en un entorno de e-commerce.
""")

st.sidebar.divider()

section = st.sidebar.radio(
    "Selecciona una sección",
    [
    "Inicio",
    "Demo tienda",
    "Eventos en vivo",
    "Simulador nuevo cliente",
    "Segmentación de clientes",
    "Análisis de sentimiento",
    "Recomendador",
    "Logística y satisfacción"
    ]
)
st.sidebar.divider()
st.sidebar.caption(
    "Proyecto final · Spark · RapidMiner · Python · Streamlit"
)
# =========================
# INICIO
# =========================
if section == "Inicio":
    st.title("Dashboard analítico de e-commerce")

    st.markdown("""
    Esta aplicación integra los principales resultados del proyecto analítico desarrollado sobre datos de e-commerce.

    El objetivo es convertir los datos procesados en información útil para la toma de decisiones:
    análisis de ventas, segmentación de clientes, detección de reseñas negativas y recomendación de productos.
    """)

    st.info("""
    La aplicación combina datos históricos procesados con una demo interactiva que simula la llegada de nuevas compras
    y reseñas, mostrando cómo podría funcionar el sistema en un entorno real.
    """)

    st.subheader("Qué permite hacer esta aplicación")

    col_a, col_b = st.columns(2)

    with col_a:
        st.markdown("""
        **Análisis histórico**
        - Consultar indicadores generales del negocio.
        - Analizar ventas por categoría.
        - Revisar logística y satisfacción.
        - Explorar segmentos de clientes.
        """)

    with col_b:
        st.markdown("""
        **Funcionalidades inteligentes**
        - Analizar sentimiento de reseñas.
        - Detectar posibles clientes insatisfechos.
        - Generar recomendaciones de productos.
        - Simular eventos nuevos desde una tienda demo.
        """)
    st.divider()
    st.subheader("Indicadores generales")

    col1, col2, col3, col4 = st.columns(4)

    total_orders = sales["order_id"].nunique()
    total_customers = sales["customer_unique_id"].nunique()
    total_products = sales["product_id"].nunique()
    total_revenue = sales["price"].sum()

    col1.metric("Pedidos históricos", f"{total_orders:,}")
    col2.metric("Clientes históricos", f"{total_customers:,}")
    col3.metric("Productos distintos", f"{total_products:,}")
    col4.metric("Ingresos históricos", f"{total_revenue:,.2f}")

    st.markdown("""
    Estos indicadores se calculan a partir del dataset histórico procesado durante la fase de ETL.
    """)
    st.divider()
    st.subheader("Ventas por categoría")

    category_sales = (
        sales.groupby("product_category_name_english", as_index=False)
        .agg(total_sales=("price", "sum"), orders=("order_id", "nunique"))
        .sort_values("total_sales", ascending=False)
        .head(15)
    )

    category_sales_display = category_sales.rename(columns={
        "product_category_name_english": "Categoría",
        "total_sales": "Ventas",
        "orders": "Pedidos"
    })

    fig = px.bar(
        category_sales_display,
        x="Ventas",
        y="Categoría",
        orientation="h",
        title="Top 15 categorías por ventas"
    )
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("""
    **Interpretación:**  
    Este gráfico permite identificar las categorías que generan mayor volumen de ingresos.
    Puede utilizarse para priorizar campañas comerciales, gestión de stock o análisis de rentabilidad.
    """)
    st.divider()
    st.subheader("Flujo del proyecto")

    st.markdown("""
    El proyecto sigue un flujo completo de análisis de datos:

    1. **Spark / PySpark**: limpieza, transformación e integración de datos.
    2. **RapidMiner**: experimentación con clustering y modelos predictivos.
    3. **Python**: entrenamiento del modelo NLP y construcción del recomendador.
    4. **Streamlit**: integración final en una aplicación interactiva.
    """)

    with st.expander("Ver arquitectura funcional de la aplicación"):
        st.markdown("""
        **Arquitectura simplificada:**

        Dataset histórico  
        → Procesamiento ETL  
        → Modelos analíticos  
        → Dashboard interactivo  

        Demo tienda  
        → Eventos simulados  
        → Análisis de sentimiento  
        → Eventos en vivo
        """)
# =========================
# DEMO TIENDA
# =========================
elif section == "Demo tienda":
    st.title("Demo tienda online")

    st.markdown("""
    Esta sección simula el funcionamiento básico de una tienda online conectada al sistema analítico.

    El objetivo es mostrar cómo una nueva compra y una nueva reseña podrían entrar en el sistema
    y ser procesadas automáticamente por el dashboard.

    En una arquitectura real, estos datos llegarían desde una base de datos transaccional,
    una API o un sistema de eventos. En esta demo, se almacenan en ficheros CSV locales.
    """)

    st.info(
        "Flujo simulado: Cliente realiza compra → escribe reseña → el modelo analiza el sentimiento → "
        "el evento aparece en la sección Eventos en vivo."
    )
    st.divider()
    st.subheader("1. Datos de la compra simulada")

    customer_ids = favorite_category["customer_unique_id"].dropna().unique()
    selected_customer = st.selectbox(
        "Cliente",
        customer_ids,
        help="Cliente que realizará la compra simulada."
    )

    categories = sorted(sales["product_category_name_english"].dropna().unique())
    selected_category = st.selectbox(
        "Categoría del producto",
        categories,
        help="Categoría del producto que se va a comprar."
    )

    products_in_category = (
        sales[sales["product_category_name_english"] == selected_category]
        [["product_id", "price"]]
        .dropna()
        .drop_duplicates("product_id")
    )

    selected_product = st.selectbox(
        "Producto",
        products_in_category["product_id"].values,
        help="Producto seleccionado dentro de la categoría."
    )

    product_price = products_in_category[
        products_in_category["product_id"] == selected_product
    ]["price"].iloc[0]

    quantity = st.number_input(
        "Cantidad",
        min_value=1,
        max_value=10,
        value=1,
        help="Número de unidades compradas."
    )

    total_value = product_price * quantity

    col1, col2 = st.columns(2)
    col1.metric("Precio unitario", f"{product_price:.2f}")
    col2.metric("Importe total", f"{total_value:.2f}")
    st.divider()
    st.subheader("2. Reseña del cliente")

    review_text = st.text_area(
        "Texto de la reseña",
        value="produto chegou atrasado e veio quebrado",
        help="Texto que será analizado automáticamente por el modelo de sentimiento."
    )

    st.markdown("""
    Al enviar el evento, la aplicación realizará tres acciones:

    1. Guardará la compra simulada.
    2. Analizará automáticamente la reseña con el modelo NLP.
    3. Guardará el resultado para visualizarlo en Eventos en vivo.
    """)

    if st.button("Enviar compra y reseña demo"):
        import uuid
        from datetime import datetime

        if review_text.strip() == "":
            st.warning("Introduce una reseña antes de enviar el evento.")
        else:
            event_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            demo_order_id = "demo_" + str(uuid.uuid4())[:8]

            prediction = sentiment_model.predict([review_text])[0]
            probabilities = sentiment_model.predict_proba([review_text])[0]
            classes = sentiment_model.classes_
            prob_dict = dict(zip(classes, probabilities))
            confidence = prob_dict[prediction]

            order_row = {
                "event_time": event_time,
                "order_id": demo_order_id,
                "customer_unique_id": selected_customer,
                "product_id": selected_product,
                "product_category_name_english": selected_category,
                "price": product_price,
                "quantity": quantity,
                "total_value": total_value
            }

            review_row = {
                "event_time": event_time,
                "order_id": demo_order_id,
                "customer_unique_id": selected_customer,
                "review_text": review_text,
                "predicted_sentiment": prediction,
                "confidence": confidence
            }

            append_row_to_csv(DEMO_ORDERS_PATH, order_row)
            append_row_to_csv(DEMO_REVIEWS_PATH, review_row)

            st.success("Compra y reseña demo registradas correctamente.")

            col1, col2, col3 = st.columns(3)
            col1.metric("Pedido demo", demo_order_id)
            col2.metric("Sentimiento detectado", prediction.upper())
            col3.metric("Confianza", f"{confidence:.2%}")

            if prediction == "negative":
                st.warning(
                    "Este evento generaría una alerta de atención al cliente, "
                    "ya que la reseña ha sido clasificada como negativa."
                )
            else:
                st.success(
                    "El evento se ha registrado sin señales relevantes de insatisfacción."
                )

            with st.expander("Ver datos guardados del evento"):
                st.write("Compra simulada")
                st.dataframe(pd.DataFrame([order_row]))

                st.write("Reseña analizada")
                st.dataframe(pd.DataFrame([review_row]))
# =========================
# EVENTOS EN VIVO
# =========================
elif section == "Eventos en vivo":
    st.title("Eventos en vivo")

    st.markdown("""
    Esta sección muestra los eventos generados desde la demo de tienda online.

    Su objetivo es simular cómo se actualizaría el sistema analítico cuando entran nuevas compras
    y nuevas reseñas de clientes. En un entorno real, estos datos llegarían desde una base de datos,
    una API o un sistema de mensajería.
    """)

    st.info(
        "Los eventos mostrados aquí no forman parte del dataset histórico original. "
        "Son eventos simulados generados desde la sección Demo tienda."
    )

    if st.button("Limpiar eventos demo"):
        if os.path.exists(DEMO_ORDERS_PATH):
            os.remove(DEMO_ORDERS_PATH)
        if os.path.exists(DEMO_REVIEWS_PATH):
            os.remove(DEMO_REVIEWS_PATH)
        st.success("Eventos demo eliminados. Recarga la página para ver los cambios.")

    demo_orders, demo_reviews = load_live_events()
    st.divider()
    st.subheader("Resumen de actividad demo")

    col1, col2, col3, col4 = st.columns(4)

    total_demo_orders = len(demo_orders)
    total_demo_reviews = len(demo_reviews)

    if not demo_orders.empty and "total_value" in demo_orders.columns:
        demo_revenue = demo_orders["total_value"].sum()
    else:
        demo_revenue = 0

    if not demo_reviews.empty and "predicted_sentiment" in demo_reviews.columns:
        negative_rate = (demo_reviews["predicted_sentiment"] == "negative").mean()
    else:
        negative_rate = 0

    col1.metric("Compras demo", total_demo_orders)
    col2.metric("Reseñas demo", total_demo_reviews)
    col3.metric("Ingresos demo", f"{demo_revenue:.2f}")
    col4.metric("% reseñas negativas", f"{negative_rate:.2%}")

    if demo_reviews.empty:
        st.info("Todavía no hay reseñas demo. Genera una desde la sección Demo tienda.")
    else:
        negative_reviews = demo_reviews[
            demo_reviews["predicted_sentiment"] == "negative"
        ]

        if len(negative_reviews) > 0:
            st.warning(
                f"Hay {len(negative_reviews)} reseña(s) negativas simuladas que podrían requerir revisión."
            )
        else:
            st.success("No se han detectado reseñas negativas en los eventos demo.")
    st.divider()
    st.subheader("Últimas compras simuladas")

    if demo_orders.empty:
        st.info("Todavía no hay compras demo.")
    else:
        orders_display = demo_orders.sort_values("event_time", ascending=False).head(20)

        orders_display = orders_display.rename(columns={
            "event_time": "Fecha del evento",
            "order_id": "Pedido",
            "customer_unique_id": "Cliente",
            "product_id": "Producto",
            "product_category_name_english": "Categoría",
            "price": "Precio unitario",
            "quantity": "Cantidad",
            "total_value": "Importe total"
        })

        st.dataframe(orders_display, use_container_width=True)
    st.divider()
    st.subheader("Últimas reseñas analizadas")

    if demo_reviews.empty:
        st.info("Todavía no hay reseñas demo.")
    else:
        reviews_display = demo_reviews.sort_values("event_time", ascending=False).head(20)

        reviews_display = reviews_display.rename(columns={
            "event_time": "Fecha del evento",
            "order_id": "Pedido",
            "customer_unique_id": "Cliente",
            "review_text": "Texto de la reseña",
            "predicted_sentiment": "Sentimiento detectado",
            "confidence": "Confianza"
        })

        st.dataframe(reviews_display, use_container_width=True)

        sentiment_counts = (
            demo_reviews["predicted_sentiment"]
            .value_counts()
            .reset_index()
        )
        sentiment_counts.columns = ["Sentimiento", "Cantidad"]

        fig = px.bar(
            sentiment_counts,
            x="Sentimiento",
            y="Cantidad",
            title="Distribución de sentimiento en eventos demo",
            text="Cantidad"
        )
        st.plotly_chart(fig, use_container_width=True)

        negative_reviews = demo_reviews[
            demo_reviews["predicted_sentiment"] == "negative"
        ].sort_values("event_time", ascending=False)

        st.subheader("Reseñas negativas recientes")

        if negative_reviews.empty:
            st.success("No hay reseñas negativas demo.")
        else:
            negative_display = negative_reviews.rename(columns={
                "event_time": "Fecha del evento",
                "order_id": "Pedido",
                "customer_unique_id": "Cliente",
                "review_text": "Texto de la reseña",
                "predicted_sentiment": "Sentimiento detectado",
                "confidence": "Confianza"
            })

            st.dataframe(negative_display.head(10))

            st.markdown("""
            **Interpretación:**  
            Estas reseñas deberían revisarse con prioridad, ya que el modelo ha detectado señales de insatisfacción.
            """)
# =========================
# SIMULADOR NUEVO CLIENTE
# =========================
elif section == "Simulador nuevo cliente":
    st.title("Simulador de nuevo cliente")

    st.markdown("""
    Esta sección permite simular la llegada de un nuevo cliente y estimar a qué segmento se parecería más.

    La asignación se realiza comparando los datos introducidos con el perfil medio de los segmentos existentes.
    Después, se proponen productos recomendados según la categoría de interés seleccionada.
    """)

    st.info("""
    Esta simulación no reentrena el modelo de clustering. 
    Asigna el cliente al segmento cuyo perfil medio es más parecido a los valores introducidos.
    """)
    st.divider()
    st.subheader("1. Datos del nuevo cliente")

    col1, col2 = st.columns(2)

    with col1:
        new_recency = st.number_input(
            "Días desde la última compra",
            min_value=0,
            max_value=1000,
            value=30,
            help="Número de días desde la última compra del cliente."
        )

        new_frequency = st.number_input(
            "Número de compras",
            min_value=1,
            max_value=100,
            value=2,
            help="Número total de compras realizadas por el cliente."
        )

    with col2:
        new_monetary = st.number_input(
            "Gasto total estimado",
            min_value=0.0,
            max_value=100000.0,
            value=250.0,
            step=10.0,
            help="Importe total gastado o estimado para el cliente."
        )

        new_rating = st.number_input(
            "Valoración media estimada",
            min_value=1.0,
            max_value=5.0,
            value=4.0,
            step=0.1,
            help="Valoración media asociada al cliente."
        )

    categories = sorted(sales["product_category_name_english"].dropna().unique())

    selected_interest_category = st.selectbox(
        "Categoría de interés",
        categories,
        help="Categoría sobre la que se desean generar recomendaciones para el nuevo cliente."
    )

    if st.button("Analizar nuevo cliente"):
        import numpy as np

        summary = prepare_cluster_summary_for_simulation(cluster_summary)

        # Intentar localizar columnas necesarias
        possible_cluster_cols = ["cluster", "Segmento"]
        cluster_col = next((c for c in possible_cluster_cols if c in summary.columns), None)

        feature_candidates = {
            "recency": ["recency", "Recencia media", "avg_recency"],
            "frequency": ["frequency", "Frecuencia media", "avg_frequency"],
            "monetary": ["monetary", "Gasto medio", "avg_monetary"],
            "avg_review_score": ["avg_review_score", "Valoración media", "review_score"]
        }

        selected_cols = {}

        for feature, candidates in feature_candidates.items():
            found_col = next((c for c in candidates if c in summary.columns), None)
            selected_cols[feature] = found_col

        missing = [k for k, v in selected_cols.items() if v is None]

        if cluster_col is None or missing:
            st.error(
                "No se han encontrado las columnas necesarias en el resumen de clusters. "
                "Revisa los nombres de columnas de cluster_summary_k3.csv."
            )
            st.write("Columnas disponibles:", summary.columns.tolist())
        else:
            # Crear matriz de centroides/resúmenes
            cluster_profiles = summary[
                [cluster_col,
                 selected_cols["recency"],
                 selected_cols["frequency"],
                 selected_cols["monetary"],
                 selected_cols["avg_review_score"]]
            ].copy()

            cluster_profiles = cluster_profiles.rename(columns={
                cluster_col: "cluster",
                selected_cols["recency"]: "recency",
                selected_cols["frequency"]: "frequency",
                selected_cols["monetary"]: "monetary",
                selected_cols["avg_review_score"]: "avg_review_score"
            })

            # Asegurar numéricos
            for col in ["recency", "frequency", "monetary", "avg_review_score"]:
                cluster_profiles[col] = pd.to_numeric(cluster_profiles[col], errors="coerce")

            cluster_profiles = cluster_profiles.dropna()

            new_customer = pd.DataFrame([{
                "recency": new_recency,
                "frequency": new_frequency,
                "monetary": new_monetary,
                "avg_review_score": new_rating
            }])

            # Normalización sencilla usando rangos de los perfiles de cluster
            features = ["recency", "frequency", "monetary", "avg_review_score"]

            mins = cluster_profiles[features].min()
            maxs = cluster_profiles[features].max()
            ranges = (maxs - mins).replace(0, 1)

            cluster_norm = (cluster_profiles[features] - mins) / ranges
            customer_norm = (new_customer[features] - mins) / ranges

            distances = np.sqrt(((cluster_norm - customer_norm.iloc[0]) ** 2).sum(axis=1))

            cluster_profiles["distance"] = distances
            assigned_cluster = cluster_profiles.sort_values("distance").iloc[0]

            assigned_cluster_id = assigned_cluster["cluster"]
            st.divider()
            st.subheader("2. Segmento estimado")

            col_a, col_b = st.columns(2)
            col_a.metric("Segmento asignado", assigned_cluster_id)
            col_b.metric("Distancia al perfil del segmento", f"{assigned_cluster['distance']:.3f}")

            st.success(
                f"El nuevo cliente se parece más al segmento {assigned_cluster_id}."
            )

            st.markdown("""
            **Interpretación:**  
            El sistema compara el comportamiento introducido con el perfil medio de cada segmento.
            El segmento asignado es aquel cuya combinación de recencia, frecuencia, gasto y valoración media
            resulta más parecida.
            """)

            st.subheader("Perfil medio del segmento asignado")

            assigned_profile_display = assigned_cluster[
                ["recency", "frequency", "monetary", "avg_review_score"]
            ].to_frame().T.rename(columns={
                "recency": "Recencia media",
                "frequency": "Frecuencia media",
                "monetary": "Gasto medio",
                "avg_review_score": "Valoración media"
            })

            st.dataframe(assigned_profile_display)

            st.subheader("3. Acción comercial sugerida")

            # Reglas simples de negocio
            if new_frequency >= 3 and new_monetary >= 300:
                st.success(
                    "Cliente con potencial alto. Se recomienda campaña de fidelización, "
                    "beneficios exclusivos o recomendaciones personalizadas."
                )
            elif new_recency > 180:
                st.warning(
                    "Cliente con riesgo de inactividad. Se recomienda campaña de recuperación "
                    "o incentivo de recompra."
                )
            elif new_rating < 3:
                st.warning(
                    "Cliente con baja satisfacción estimada. Se recomienda revisar experiencia, "
                    "producto o entrega antes de realizar campañas comerciales."
                )
            else:
                st.info(
                    "Cliente de comportamiento medio. Se recomienda una campaña comercial estándar "
                    "basada en sus categorías de interés."
                )
            st.divider()
            st.subheader("4. Productos recomendados para el nuevo cliente")

            category_recs = global_recommendations[
                global_recommendations["product_category_name_english"] == selected_interest_category
            ].sort_values("recommendation_score", ascending=False).head(5)

            if category_recs.empty:
                st.warning(
                    "No hay suficientes productos recomendables en la categoría seleccionada. "
                    "Se muestran recomendaciones globales."
                )
                category_recs = global_recommendations.sort_values(
                    "recommendation_score",
                    ascending=False
                ).head(5)

            recs_display = category_recs[
                [
                    "product_id",
                    "product_category_name_english",
                    "recommendation_score",
                    "avg_rating",
                    "total_orders",
                    "avg_price"
                ]
            ].rename(columns={
                "product_id": "ID técnico del producto",
                "product_category_name_english": "Categoría",
                "recommendation_score": "Puntuación",
                "avg_rating": "Valoración media",
                "total_orders": "Número de pedidos",
                "avg_price": "Precio medio"
            })

            recs_display = recs_display.reset_index(drop=True)

            recs_display["Producto recomendado"] = (
                recs_display["Categoría"].astype(str)
                + " · opción "
                + (recs_display.index + 1).astype(str)
            )

            recs_display["Puntuación"] = recs_display["Puntuación"].round(3)
            recs_display["Valoración media"] = recs_display["Valoración media"].round(2)
            recs_display["Precio medio"] = recs_display["Precio medio"].round(2)

            st.dataframe(
                recs_display[
                    [
                        "Producto recomendado",
                        "Categoría",
                        "Puntuación",
                        "Valoración media",
                        "Número de pedidos",
                        "Precio medio",
                        "ID técnico del producto"
                    ]
                ], use_container_width=True
            )

            fig = px.bar(
                recs_display,
                x="Producto recomendado",
                y="Puntuación",
                title="Productos recomendados para el nuevo cliente",
                text="Puntuación",
                hover_data={
                    "ID técnico del producto": True,
                    "Categoría": True,
                    "Valoración media": True,
                    "Número de pedidos": True,
                    "Precio medio": True,
                    "Puntuación": True
                }
            )

            fig.update_traces(texttemplate="%{text:.3f}", textposition="outside")

            st.plotly_chart(fig, use_container_width=True)

            with st.expander("Ver explicación técnica de la simulación"):
                st.markdown("""
                **Asignación del segmento:**

                1. Se toman los valores medios de cada cluster.
                2. Se normalizan las variables para que tengan una escala comparable.
                3. Se calcula la distancia entre el nuevo cliente y cada perfil medio.
                4. Se asigna el cliente al cluster con menor distancia.

                **Recomendación:**

                Para clientes nuevos se utilizan productos populares y bien valorados dentro de la categoría de interés.
                Si no hay suficientes productos, se utiliza el ranking global.
                """)
# =========================
# SEGMENTACIÓN
# =========================
elif section == "Segmentación de clientes":
    st.title("Segmentación de clientes")

    st.markdown("""
    Esta sección agrupa a los clientes según su comportamiento de compra.

    La segmentación permite identificar distintos perfiles de clientes y adaptar acciones comerciales
    o estrategias de fidelización a cada grupo.
    """)

    st.info("""
    Variables utilizadas en la segmentación:

    - **Recencia**: tiempo transcurrido desde la última compra.
    - **Frecuencia**: número de compras realizadas.
    - **Valor monetario**: importe total gastado por el cliente.
    - **Valoración media**: satisfacción media asociada a sus compras.
    """)

    st.subheader("Resumen de segmentos")

    cluster_summary_display = cluster_summary.copy()

    cluster_summary_display.columns = (
        cluster_summary_display.columns
        .str.replace("average\\(", "", regex=True)
        .str.replace("\\)", "", regex=True)
        .str.replace('"', "", regex=False)
        .str.strip()
    )

    rename_cols = {
        "cluster": "Segmento",
        "recency": "Recencia media",
        "frequency": "Frecuencia media",
        "monetary": "Gasto medio",
        "avg_review_score": "Valoración media",
        "count": "Número de clientes"
    }

    cluster_summary_display = cluster_summary_display.rename(columns=rename_cols)

    st.dataframe(cluster_summary_display, use_container_width=True)

    st.markdown("""
    **Cómo interpretar esta tabla:**

    - Un segmento con mayor frecuencia y mayor gasto suele representar clientes de mayor valor.
    - Un segmento con mucha recencia puede indicar clientes que llevan más tiempo sin comprar.
    - La valoración media ayuda a entender el nivel de satisfacción de cada grupo.
    """)

    if "cluster" in rfm.columns:
        cluster_counts = rfm["cluster"].value_counts().reset_index()
        cluster_counts.columns = ["Segmento", "Clientes"]

        fig = px.bar(
            cluster_counts,
            x="Segmento",
            y="Clientes",
            title="Número de clientes por segmento",
            text="Clientes"
        )
        st.plotly_chart(fig, use_container_width=True)

    st.subheader("Interpretación de negocio")

    st.markdown("""
    Los segmentos pueden utilizarse para definir acciones diferenciadas:

    - **Clientes de alto valor**: campañas de fidelización, ventajas exclusivas o recomendaciones premium.
    - **Clientes recientes o activos**: promociones cruzadas y recomendaciones personalizadas.
    - **Clientes inactivos**: campañas de recuperación o descuentos específicos.
    - **Clientes con baja satisfacción**: revisión de experiencia, entregas o calidad del producto.

    La etiqueta numérica del segmento no indica por sí sola si un grupo es bueno o malo.
    La interpretación depende de los valores medios de recencia, frecuencia, gasto y satisfacción.
    """)

    st.subheader("Clientes segmentados")

    rfm_display = rfm.copy()

    rfm_display = rfm_display.rename(columns={
        "customer_unique_id": "Cliente",
        "recency": "Recencia",
        "frequency": "Frecuencia",
        "monetary": "Gasto",
        "avg_review_score": "Valoración media",
        "cluster": "Segmento"
    })

    st.dataframe(rfm_display.head(100), use_container_width=True)

    with st.expander("Ver explicación técnica del clustering"):
        st.markdown("""
        **Metodología utilizada:**

        1. Se calcularon variables RFM para cada cliente.
        2. Se incorporó la valoración media como variable adicional de comportamiento.
        3. Se aplicó clustering para agrupar clientes con patrones similares.
        4. El número de clusters se evaluó durante la fase de experimentación.

        El objetivo no es predecir una variable concreta, sino descubrir grupos de clientes con comportamientos parecidos.
        """)

# =========================
# SENTIMIENTO
# =========================
elif section == "Análisis de sentimiento":
    st.title("Análisis de sentimiento de reseñas")

    st.markdown("""
    Este módulo permite analizar automáticamente el texto de una reseña escrita por un cliente.

    El modelo clasifica la reseña en dos posibles resultados:

    - **Positiva**: el cliente muestra una opinión favorable.
    - **Negativa**: el cliente muestra señales de insatisfacción.

    Esta funcionalidad puede ayudar a detectar comentarios negativos de forma temprana y priorizar la atención al cliente.
    """)

    st.info(
        "Modelo utilizado: TF-IDF + Regresión Logística. "
        "El modelo fue entrenado con reseñas históricas clasificadas como positivas o negativas."
    )

    review_text = st.text_area(
        "Escribe aquí la reseña del cliente",
        value="produto chegou atrasado e veio quebrado",
        help="Puedes escribir una reseña en portugués similar a las del dataset original."
    )

    if st.button("Analizar reseña"):
        if review_text.strip() == "":
            st.warning("Introduce un texto para poder analizar la reseña.")
        else:
            prediction = sentiment_model.predict([review_text])[0]
            probabilities = sentiment_model.predict_proba([review_text])[0]
            classes = sentiment_model.classes_

            prob_dict = dict(zip(classes, probabilities))
            confidence = prob_dict[prediction]
            
            st.divider()
            st.subheader("Resultado del análisis")

            col1, col2 = st.columns(2)
            col1.metric("Sentimiento detectado", prediction.upper())
            col2.metric("Confianza del modelo", f"{confidence:.2%}")

            if prediction == "negative":
                st.error("La reseña ha sido clasificada como NEGATIVA.")
                st.markdown("""
                **Interpretación:**  
                El cliente muestra señales de insatisfacción.  
                Se recomienda revisar el pedido, la entrega o el producto asociado.
                """)
            else:
                st.success("La reseña ha sido clasificada como POSITIVA.")
                st.markdown("""
                **Interpretación:**  
                El cliente muestra una opinión favorable sobre la compra.  
                No se detectan señales relevantes de insatisfacción.
                """)

            st.info(
                "La confianza representa la seguridad del modelo en la clasificación realizada. "
                "Cuanto mayor sea el porcentaje, más clara es la predicción para el modelo."
            )

            prob_df = pd.DataFrame({
                "Sentimiento": [
                    "Negativo" if cls == "negative" else "Positivo"
                    for cls in prob_dict.keys()
                ],
                "Probabilidad": list(prob_dict.values())
            })

            fig = px.bar(
                prob_df,
                x="Sentimiento",
                y="Probabilidad",
                title="Probabilidad estimada para cada sentimiento",
                text="Probabilidad"
            )

            fig.update_traces(texttemplate="%{text:.2%}", textposition="outside")
            fig.update_layout(yaxis_tickformat=".0%")

            st.plotly_chart(fig, use_container_width=True)

            with st.expander("Ver detalles técnicos del modelo"):
                st.markdown("""
                **Funcionamiento resumido:**

                1. El texto de la reseña se transforma en variables numéricas mediante TF-IDF.
                2. La Regresión Logística estima la probabilidad de que la reseña sea positiva o negativa.
                3. Se devuelve la clase con mayor probabilidad.

                Este modelo está pensado para apoyar la toma de decisiones, no para sustituir completamente la revisión humana.
                """)

                st.write("Probabilidades internas:")
                st.json({k: float(v) for k, v in prob_dict.items()})
# =========================
# RECOMENDADOR
# =========================
elif section == "Recomendador":
    st.title("Sistema de recomendación")

    st.markdown("""
    Este módulo propone productos recomendados para cada cliente.

    La recomendación se basa en una lógica sencilla y explicable:

    1. Se identifica la categoría favorita del cliente según su historial de compras.
    2. Se buscan productos populares y bien valorados dentro de esa categoría.
    3. Se excluyen productos que el cliente ya ha comprado.
    4. Si no hay suficientes productos en la categoría, se utilizan productos populares globales.

    Este enfoque permite generar recomendaciones comprensibles para negocio y fáciles de justificar.
    """)

    st.info(
        "La puntuación de recomendación combina popularidad del producto y valoración media histórica."
    )

    customer_ids = favorite_category["customer_unique_id"].dropna().unique()

    selected_customer = st.selectbox(
        "Selecciona un cliente",
        customer_ids,
        help="Cliente para el que se generarán recomendaciones."
    )

    customer_info = favorite_category[
        favorite_category["customer_unique_id"] == selected_customer
    ].copy()

    st.subheader("Perfil resumido del cliente")

    if not customer_info.empty:
        customer_info_display = customer_info.rename(columns={
            "customer_unique_id": "Cliente",
            "favorite_category": "Categoría favorita",
            "category_orders": "Pedidos en la categoría",
            "category_spend": "Gasto en la categoría",
            "avg_category_rating": "Valoración media en la categoría"
        })

        st.dataframe(customer_info_display)

        fav_cat = customer_info["favorite_category"].iloc[0]
        st.success(
            f"La categoría favorita de este cliente es: {fav_cat}. "
            "Las recomendaciones se priorizan dentro de esta categoría."
        )
    else:
        st.warning("No se ha encontrado información para este cliente.")

    customer_recs = recommendations[
        recommendations["customer_unique_id"] == selected_customer
    ].sort_values("score", ascending=False)

    st.subheader("Productos recomendados")

    if customer_recs.empty:
        st.warning("No hay recomendaciones disponibles para este cliente.")
    else:
        display_cols = [
            "recommended_product_id",
            "product_category_name_english",
            "score",
            "avg_rating",
            "total_orders",
            "avg_price",
            "reason"
        ]

        customer_recs_display = customer_recs[display_cols].rename(columns={
            "recommended_product_id": "ID técnico del producto",
            "product_category_name_english": "Categoría",
            "score": "Puntuación",
            "avg_rating": "Valoración media",
            "total_orders": "Número de pedidos",
            "avg_price": "Precio medio",
            "reason": "Motivo de recomendación"
        })

        customer_recs_display = customer_recs_display.reset_index(drop=True)

        customer_recs_display["Producto recomendado"] = (
            customer_recs_display["Categoría"].astype(str)
            + " · opción "
            + (customer_recs_display.index + 1).astype(str)
        )

        customer_recs_display["Puntuación"] = customer_recs_display["Puntuación"].round(3)
        customer_recs_display["Valoración media"] = customer_recs_display["Valoración media"].round(2)
        customer_recs_display["Precio medio"] = customer_recs_display["Precio medio"].round(2)

        st.dataframe(
            customer_recs_display[
                [
                    "Producto recomendado",
                    "Categoría",
                    "Puntuación",
                    "Valoración media",
                    "Número de pedidos",
                    "Precio medio",
                    "Motivo de recomendación",
                    "ID técnico del producto"
                ]
            ], use_container_width=True
        )

        best_rec = customer_recs_display.iloc[0]

        st.markdown(f"""
        **Interpretación:**  
        El producto con mayor prioridad para este cliente es `{best_rec["Producto recomendado"]}`.  
        Pertenece a la categoría `{best_rec["Categoría"]}` y tiene una puntuación de recomendación de `{best_rec["Puntuación"]}`.
        """)

        st.info(
            "Estas recomendaciones pueden utilizarse para campañas personalizadas, "
            "emails comerciales o sugerencias dentro de una tienda online."
        )

        fig = px.bar(
            customer_recs_display,
            x="Producto recomendado",
            y="Puntuación",
            title="Puntuación de los productos recomendados",
            text="Puntuación",
            hover_data={
                "ID técnico del producto": True,
                "Categoría": True,
                "Valoración media": True,
                "Número de pedidos": True,
                "Precio medio": True,
                "Motivo de recomendación": True,
                "Puntuación": True
            }
        )

        fig.update_traces(texttemplate="%{text:.3f}", textposition="outside")

        st.plotly_chart(fig, use_container_width=True)

    with st.expander("Ver explicación técnica del recomendador"):
        st.markdown("""
        **Cálculo del score de recomendación:**

        - Se calcula la popularidad del producto a partir del número de pedidos.
        - Se calcula la valoración media histórica del producto.
        - Ambas variables se normalizan.
        - El score final combina ambas métricas:

        `score = 0.7 * popularidad_normalizada + 0.3 * valoración_normalizada`

        Este método no requiere datos complejos y funciona bien cuando muchos clientes tienen pocas compras históricas.
        """)
# =========================
# LOGÍSTICA
# =========================
elif section == "Logística y satisfacción":
    st.title("Logística y satisfacción del cliente")

    st.markdown("""
    Esta sección analiza la relación entre el cumplimiento de los plazos de entrega y la satisfacción del cliente.

    El objetivo es identificar si los retrasos en la entrega tienen impacto en la valoración media de los pedidos
    y qué categorías presentan mayores problemas logísticos.
    """)

    st.info("""
    Una entrega se considera retrasada cuando la fecha real de entrega es posterior a la fecha estimada de entrega.
    """)

    logistics = sales.copy()

    logistics["order_delivered_customer_date"] = pd.to_datetime(
        logistics["order_delivered_customer_date"],
        errors="coerce"
    )
    logistics["order_estimated_delivery_date"] = pd.to_datetime(
        logistics["order_estimated_delivery_date"],
        errors="coerce"
    )

    logistics["delivery_delay_days"] = (
        logistics["order_delivered_customer_date"] -
        logistics["order_estimated_delivery_date"]
    ).dt.days

    logistics["is_late_delivery"] = logistics["delivery_delay_days"] > 0

    col1, col2, col3 = st.columns(3)

    late_rate = logistics["is_late_delivery"].mean()
    avg_delay = logistics["delivery_delay_days"].mean()
    avg_review = logistics["avg_review_score"].mean()

    col1.metric("Pedidos con retraso", f"{late_rate:.2%}")
    col2.metric("Diferencia media entrega-estimada", f"{avg_delay:.2f} días")
    col3.metric("Valoración media", f"{avg_review:.2f}")

    st.markdown("""
    **Interpretación de los indicadores:**

    - Un porcentaje alto de retrasos puede indicar problemas logísticos.
    - Una diferencia media negativa significa que, de media, los pedidos se entregan antes de la fecha estimada.
    - La valoración media permite observar la satisfacción general de los clientes.
    """)

    st.subheader("Satisfacción según cumplimiento de entrega")

    delay_review = (
        logistics.groupby("is_late_delivery", as_index=False)
        .agg(
            avg_review_score=("avg_review_score", "mean"),
            orders=("order_id", "nunique")
        )
    )

    delay_review["Tipo de entrega"] = delay_review["is_late_delivery"].map({
        False: "A tiempo o antes",
        True: "Con retraso"
    })

    delay_review_display = delay_review.rename(columns={
        "avg_review_score": "Valoración media",
        "orders": "Pedidos"
    })

    fig = px.bar(
        delay_review_display,
        x="Tipo de entrega",
        y="Valoración media",
        text="Valoración media",
        title="Valoración media según tipo de entrega"
    )

    fig.update_traces(texttemplate="%{text:.2f}", textposition="outside")

    st.plotly_chart(fig, use_container_width=True)

    st.markdown("""
    **Interpretación:**  
    Este gráfico permite comparar si los pedidos entregados con retraso reciben peor valoración que los entregados a tiempo.
    Si la diferencia es clara, puede justificar acciones de mejora logística.
    """)

    st.subheader("Categorías con mayor retraso medio")

    delay_category = (
        logistics.groupby("product_category_name_english", as_index=False)
        .agg(
            avg_delay=("delivery_delay_days", "mean"),
            orders=("order_id", "nunique"),
            avg_review=("avg_review_score", "mean")
        )
        .dropna()
        .sort_values("avg_delay", ascending=False)
        .head(15)
    )

    delay_category_display = delay_category.rename(columns={
        "product_category_name_english": "Categoría",
        "avg_delay": "Retraso medio",
        "orders": "Pedidos",
        "avg_review": "Valoración media"
    })

    fig2 = px.bar(
        delay_category_display,
        x="Retraso medio",
        y="Categoría",
        orientation="h",
        title="Top 15 categorías con mayor retraso medio",
        text="Retraso medio"
    )

    fig2.update_traces(texttemplate="%{text:.2f}", textposition="outside")

    st.plotly_chart(fig2, use_container_width=True)

    st.markdown("""
    **Interpretación:**  
    Las categorías con mayor retraso medio pueden requerir revisión de proveedores, transporte o gestión de stock.
    """)

    st.subheader("Detalle de categorías")

    st.dataframe(delay_category_display, use_container_width=True)

    with st.expander("Ver explicación técnica del cálculo"):
        st.markdown("""
        **Cálculo utilizado:**

        - Se convierte la fecha real de entrega y la fecha estimada a formato fecha.
        - Se calcula la diferencia en días:

        `delivery_delay_days = fecha_entrega_real - fecha_entrega_estimada`

        - Si el resultado es mayor que 0, el pedido se considera retrasado.
        - Después se agregan los resultados por tipo de entrega y por categoría.
        """)