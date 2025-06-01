# Este archivo simplemente ejecuta el contenido de app.py
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans
import requests
import io
import plotly.express as px
import plotly.graph_objects as go

# Configuración de la página
st.set_page_config(
    page_title="Segmentación Bancaria",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Título principal
st.title("🏦 Sistema de Segmentación Bancaria")
st.markdown("**Clasificación automática de clientes usando K-means Clustering**")

# Función para cargar datos desde GitHub
@st.cache_data
def cargar_datos_github():
    """Cargar datos desde el repositorio de GitHub"""
    url = "https://raw.githubusercontent.com/anthonyllan/segmentacionbancaria/refs/heads/main/segmentacionbancaria.csv"
    try:
        datos = pd.read_csv(url)
        return datos
    except Exception as e:
        st.error(f"Error al cargar datos: {e}")
        return None

# Función para entrenar el modelo
@st.cache_data
def entrenar_modelo(datos):
    """Entrenar el modelo de clustering"""
    # Seleccionar características
    columnas_numericas = ['saldoCuentaAhorro', 'frecuenciaUsoMensual']
    datos_numericos = datos[columnas_numericas]
    
    # Normalizar datos
    scaler = MinMaxScaler()
    datos_escalados = scaler.fit_transform(datos_numericos)
    
    # Entrenar modelo
    kmeans = KMeans(n_clusters=3, random_state=42)
    clusters = kmeans.fit_predict(datos_escalados)
    
    return kmeans, scaler, clusters, datos_escalados

# Función para predecir cluster de nuevo cliente
def predecir_cluster(saldo, frecuencia, kmeans, scaler):
    """Predecir cluster para un nuevo cliente"""
    # Crear DataFrame con las características
    nuevo_cliente = pd.DataFrame({
        'saldoCuentaAhorro': [saldo],
        'frecuenciaUsoMensual': [frecuencia]
    })
    
    # Normalizar
    nuevo_cliente_escalado = scaler.transform(nuevo_cliente)
    
    # Predecir cluster
    cluster = kmeans.predict(nuevo_cliente_escalado)[0]
    
    return cluster

# Función para obtener recomendaciones de productos
def obtener_recomendaciones(cluster):
    """Obtener recomendaciones de productos según el cluster"""
    recomendaciones = {
        0: {
            "perfil": "Cliente Premium - Alto Saldo, Baja Frecuencia",
            "descripcion": "Clientes con alto patrimonio que usan poco sus cuentas",
            "productos": [
                "💰 Cuenta de Ahorro Premium (4.5% anual)",
                "📈 Fondos de Inversión",
                "🏠 Crédito Hipotecario Preferencial",
                "🛡️ Seguro de Vida Premium",
                "💎 Tarjeta de Crédito Platinum"
            ],
            "color": "#FF6B6B"
        },
        1: {
            "perfil": "Cliente Estándar - Bajo Saldo, Baja Frecuencia",
            "descripcion": "Clientes con patrimonio moderado y uso esporádico",
            "productos": [
                "💳 Cuenta de Ahorro Básica (2.5% anual)",
                "🎯 Microcrédito Personal",
                "📱 Banca Móvil Básica",
                "🏥 Seguro Médico Básico",
                "💸 Tarjeta de Débito"
            ],
            "color": "#4ECDC4"
        },
        2: {
            "perfil": "Cliente Digital - Saldo Medio, Alta Frecuencia",
            "descripcion": "Clientes muy activos digitalmente",
            "productos": [
                "📱 Banca Digital Premium",
                "⚡ Transferencias Instantáneas Gratis",
                "💳 Tarjeta de Crédito Digital",
                "🎯 Inversiones Automáticas",
                "🔔 Notificaciones Inteligentes"
            ],
            "color": "#45B7D1"
        }
    }
    return recomendaciones.get(cluster, {})

# Cargar datos y entrenar modelo
datos = cargar_datos_github()
if datos is not None:
    kmeans, scaler, clusters, datos_escalados = entrenar_modelo(datos)
    
    # Agregar clusters a los datos originales
    datos_con_clusters = datos.copy()
    datos_con_clusters['cluster'] = clusters

    # Sidebar para navegación
    st.sidebar.title("🔍 Navegación")
    opcion = st.sidebar.selectbox(
        "Seleccione una opción:",
        ["📊 Dashboard Principal", "👤 Análisis Individual", "📁 Carga Masiva", "📈 Visualización Avanzada"]
    )

    # Dashboard Principal
    if opcion == "📊 Dashboard Principal":
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Clientes", len(datos))
        with col2:
            st.metric("Cluster 0", len(datos_con_clusters[datos_con_clusters['cluster'] == 0]))
        with col3:
            st.metric("Cluster 1", len(datos_con_clusters[datos_con_clusters['cluster'] == 1]))
        with col4:
            st.metric("Cluster 2", len(datos_con_clusters[datos_con_clusters['cluster'] == 2]))
        
        # Gráfico principal
        st.subheader("📈 Distribución de Clusters")
        
        fig = px.scatter(
            datos_con_clusters, 
            x='saldoCuentaAhorro', 
            y='frecuenciaUsoMensual',
            color='cluster',
            hover_data=['nombre'],
            title="Segmentación de Clientes",
            labels={'saldoCuentaAhorro': 'Saldo Cuenta Ahorro', 'frecuenciaUsoMensual': 'Frecuencia Uso Mensual'}
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Análisis por clusters
        st.subheader("🎯 Análisis por Clusters")
        
        col1, col2, col3 = st.columns(3)
        
        for i in range(3):
            cluster_data = datos_con_clusters[datos_con_clusters['cluster'] == i]
            recom = obtener_recomendaciones(i)
            
            with [col1, col2, col3][i]:
                st.markdown(f"**{recom['perfil']}**")
                st.write(f"Clientes: {len(cluster_data)}")
                st.write(f"Saldo promedio: ${cluster_data['saldoCuentaAhorro'].mean():,.0f}")
                st.write(f"Frecuencia promedio: {cluster_data['frecuenciaUsoMensual'].mean():.1f}")

    # Análisis Individual
    elif opcion == "👤 Análisis Individual":
        st.subheader("👤 Análisis de Cliente Individual")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("### Datos del Cliente")
            nombre = st.text_input("Nombre del cliente:")
            saldo = st.number_input("Saldo Cuenta Ahorro:", min_value=0, value=5000, step=100)
            frecuencia = st.number_input("Frecuencia Uso Mensual:", min_value=0, value=10, step=1)
            
            if st.button("🔍 Analizar Cliente", type="primary"):
                # Predecir cluster
                cluster_predicho = predecir_cluster(saldo, frecuencia, kmeans, scaler)
                
                # Obtener recomendaciones
                recomendaciones = obtener_recomendaciones(cluster_predicho)
                
                # Mostrar resultados
                st.success(f"✅ Análisis completado para {nombre}")
                
                with col2:
                    st.write("### 🎯 Resultado del Análisis")
                    st.write(f"**Cluster asignado:** {cluster_predicho}")
                    st.write(f"**Perfil:** {recomendaciones['perfil']}")
                    st.write(f"**Descripción:** {recomendaciones['descripcion']}")
                    
                    st.write("### 💼 Productos Recomendados")
                    for producto in recomendaciones['productos']:
                        st.write(f"• {producto}")

    # Carga Masiva
    elif opcion == "📁 Carga Masiva":
        st.subheader("📁 Carga Masiva de Clientes")
        
        uploaded_file = st.file_uploader("Seleccione archivo CSV", type=['csv'])
        
        if uploaded_file is not None:
            try:
                # Leer archivo
                nuevos_datos = pd.read_csv(uploaded_file)
                
                st.write("### 📋 Vista Previa de Datos")
                st.dataframe(nuevos_datos.head())
                
                # Verificar columnas requeridas
                columnas_requeridas = ['nombre', 'saldoCuentaAhorro', 'frecuenciaUsoMensual']
                if all(col in nuevos_datos.columns for col in columnas_requeridas):
                    
                    if st.button("🚀 Procesar Archivo", type="primary"):
                        # Predecir clusters para todos los clientes
                        clusters_predichos = []
                        
                        for _, row in nuevos_datos.iterrows():
                            cluster = predecir_cluster(
                                row['saldoCuentaAhorro'], 
                                row['frecuenciaUsoMensual'], 
                                kmeans, 
                                scaler
                            )
                            clusters_predichos.append(cluster)
                        
                        # Agregar clusters a los datos
                        nuevos_datos['cluster'] = clusters_predichos
                        
                        # Mostrar resultados
                        st.success("✅ Procesamiento completado")
                        
                        # Estadísticas
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("Cluster 0", sum(1 for c in clusters_predichos if c == 0))
                        with col2:
                            st.metric("Cluster 1", sum(1 for c in clusters_predichos if c == 1))
                        with col3:
                            st.metric("Cluster 2", sum(1 for c in clusters_predichos if c == 2))
                        
                        # Mostrar datos procesados
                        st.write("### 📊 Datos Procesados")
                        st.dataframe(nuevos_datos)
                        
                        # Descargar resultados
                        csv = nuevos_datos.to_csv(index=False)
                        st.download_button(
                            label="💾 Descargar Resultados",
                            data=csv,
                            file_name='clientes_segmentados.csv',
                            mime='text/csv'
                        )
                        
                else:
                    st.error(f"❌ El archivo debe contener las columnas: {columnas_requeridas}")
                    
            except Exception as e:
                st.error(f"❌ Error al procesar archivo: {e}")

    # Visualización Avanzada  
    elif opcion == "📈 Visualización Avanzada":
        st.subheader("📈 Análisis Avanzado de Clusters")
        
        # Método del codo
        st.write("### 🔧 Método del Codo")
        
        k_range = range(1, 11)
        inertias = []
        
        for k in k_range:
            kmeans_temp = KMeans(n_clusters=k, random_state=42)
            kmeans_temp.fit(datos_escalados)
            inertias.append(kmeans_temp.inertia_)
        
        fig_codo = px.line(
            x=list(k_range), 
            y=inertias,
            title="Método del Codo - Determinación del Número Óptimo de Clusters",
            labels={'x': 'Número de Clusters (k)', 'y': 'Inercia (WCSS)'}
        )
        fig_codo.add_vline(x=3, line_dash="dash", line_color="red", annotation_text="k=3 (Óptimo)")
        st.plotly_chart(fig_codo, use_container_width=True)
        
        # Distribuciones
        st.write("### 📊 Distribuciones por Cluster")
        
        col1, col2 = st.columns(2)
        
        with col1:
            fig_saldo = px.box(
                datos_con_clusters, 
                x='cluster', 
                y='saldoCuentaAhorro',
                title="Distribución de Saldos por Cluster"
            )
            st.plotly_chart(fig_saldo, use_container_width=True)
        
        with col2:
            fig_freq = px.box(
                datos_con_clusters, 
                x='cluster', 
                y='frecuenciaUsoMensual',
                title="Distribución de Frecuencia por Cluster"
            )
            st.plotly_chart(fig_freq, use_container_width=True)

else:
    st.error("❌ No se pudieron cargar los datos desde GitHub")

# Footer
st.markdown("---")
st.markdown("**🏦 Sistema de Segmentación Bancaria** | Desarrollado con Streamlit y scikit-learn")
