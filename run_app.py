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
from datetime import datetime
import json
import base64

# Configuración de la página
st.set_page_config(
    page_title="Segmentación Bancaria",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Configuración GitHub
GITHUB_TOKEN = st.secrets.get("GITHUB_TOKEN", "")
GITHUB_REPO = "anthonyllan/segmentacionbancaria"
GITHUB_FILE = "segmentacionbancaria.csv"

# Título principal
st.title("🏦 Sistema de Segmentación Bancaria")
st.markdown("**Clasificación automática de clientes usando K-means Clustering con Aprendizaje Continuo**")

# Función para actualizar GitHub automáticamente
def actualizar_github_csv(datos_actualizados):
    """Actualizar archivo CSV en GitHub usando la API"""
    if not GITHUB_TOKEN:
        st.error("❌ Token de GitHub no configurado en Secrets")
        return False
    
    try:
        with st.spinner("🔄 Actualizando GitHub..."):
            # Convertir datos a CSV
            csv_content = datos_actualizados.to_csv(index=False)
            
            # Obtener SHA del archivo actual
            url_get = f"https://api.github.com/repos/{GITHUB_REPO}/contents/{GITHUB_FILE}"
            headers = {
                "Authorization": f"token {GITHUB_TOKEN}",
                "Accept": "application/vnd.github.v3+json"
            }
            
            response = requests.get(url_get, headers=headers)
            
            if response.status_code == 200:
                file_info = response.json()
                sha = file_info["sha"]
                
                # Actualizar archivo
                url_update = f"https://api.github.com/repos/{GITHUB_REPO}/contents/{GITHUB_FILE}"
                
                data = {
                    "message": f"🤖 Actualización automática - Nuevos clientes validados {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
                    "content": base64.b64encode(csv_content.encode()).decode(),
                    "sha": sha
                }
                
                response = requests.put(url_update, headers=headers, json=data)
                
                if response.status_code == 200:
                    st.success("✅ ¡Archivo actualizado en GitHub automáticamente!")
                    st.info("🔄 La aplicación se actualizará en 1-2 minutos")
                    return True
                else:
                    st.error(f"❌ Error al actualizar GitHub: {response.status_code}")
                    st.error(f"Detalles: {response.text}")
                    return False
            else:
                st.error(f"❌ Error al obtener archivo de GitHub: {response.status_code}")
                return False
                
    except Exception as e:
        st.error(f"❌ Error de conexión: {e}")
        return False

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

# Función para cargar nuevos clientes desde session_state
def cargar_nuevos_clientes():
    """Cargar clientes agregados en esta sesión"""
    if 'nuevos_clientes' not in st.session_state:
        st.session_state.nuevos_clientes = []
    return pd.DataFrame(st.session_state.nuevos_clientes)

# Función para agregar nuevo cliente
def agregar_nuevo_cliente(nombre, saldo, frecuencia, cluster_real=None):
    """Agregar nuevo cliente al dataset de aprendizaje"""
    if 'nuevos_clientes' not in st.session_state:
        st.session_state.nuevos_clientes = []
    
    nuevo_cliente = {
        'nombre': nombre,
        'saldoCuentaAhorro': saldo,
        'frecuenciaUsoMensual': frecuencia,
        'cluster_predicho': None,
        'cluster_real': cluster_real,
        'fecha_registro': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        'validado': cluster_real is not None
    }
    
    st.session_state.nuevos_clientes.append(nuevo_cliente)
    return nuevo_cliente

# Función para combinar datos originales con nuevos
def combinar_datos(datos_originales):
    """Combinar datos originales con nuevos clientes"""
    nuevos_df = cargar_nuevos_clientes()
    
    if len(nuevos_df) > 0:
        # Preparar nuevos datos para combinar
        nuevos_para_modelo = nuevos_df[['nombre', 'saldoCuentaAhorro', 'frecuenciaUsoMensual']].copy()
        
        # Combinar con datos originales
        datos_combinados = pd.concat([datos_originales, nuevos_para_modelo], ignore_index=True)
        return datos_combinados, nuevos_df
    else:
        return datos_originales, nuevos_df

# Función para entrenar el modelo con datos combinados
def entrenar_modelo_adaptativo(datos_originales, nuevos_datos):
    """Entrenar el modelo con datos originales + nuevos datos validados"""
    # Combinar datos
    datos_combinados, _ = combinar_datos(datos_originales)
    
    # Seleccionar características
    columnas_numericas = ['saldoCuentaAhorro', 'frecuenciaUsoMensual']
    datos_numericos = datos_combinados[columnas_numericas]
    
    # Normalizar datos
    scaler = MinMaxScaler()
    datos_escalados = scaler.fit_transform(datos_numericos)
    
    # Entrenar modelo
    kmeans = KMeans(n_clusters=3, random_state=42)
    clusters = kmeans.fit_predict(datos_escalados)
    
    return kmeans, scaler, clusters, datos_escalados, datos_combinados

# Función para predecir cluster de nuevo cliente
def predecir_cluster(saldo, frecuencia, kmeans, scaler):
    """Predecir cluster para un nuevo cliente"""
    nuevo_cliente = pd.DataFrame({
        'saldoCuentaAhorro': [saldo],
        'frecuenciaUsoMensual': [frecuencia]
    })
    
    nuevo_cliente_escalado = scaler.transform(nuevo_cliente)
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

# Función para generar CSV actualizado
def generar_csv_actualizado(datos_originales, nuevos_datos_validados):
    """Generar CSV actualizado con nuevos datos validados"""
    if len(nuevos_datos_validados) > 0:
        # Filtrar solo datos validados
        validados = nuevos_datos_validados[nuevos_datos_validados['validado'] == True].copy()
        
        if len(validados) > 0:
            # Preparar datos para agregar
            datos_para_agregar = validados[['nombre', 'saldoCuentaAhorro', 'frecuenciaUsoMensual']].copy()
            
            # Combinar con datos originales
            datos_actualizados = pd.concat([datos_originales, datos_para_agregar], ignore_index=True)
            
            return datos_actualizados
    
    return datos_originales

# Cargar datos y entrenar modelo
datos_originales = cargar_datos_github()
nuevos_datos = cargar_nuevos_clientes()

if datos_originales is not None:
    # Entrenar modelo con datos combinados
    kmeans, scaler, clusters, datos_escalados, datos_combinados = entrenar_modelo_adaptativo(datos_originales, nuevos_datos)
    
    # Agregar clusters a los datos combinados
    datos_con_clusters = datos_combinados.copy()
    datos_con_clusters['cluster'] = clusters

    # Mostrar estado de GitHub en sidebar
    st.sidebar.title("🔍 Navegación")
    
    # Estado de GitHub
    if GITHUB_TOKEN:
        st.sidebar.success("🔗 GitHub conectado")
    else:
        st.sidebar.error("❌ GitHub no conectado")
    
    opcion = st.sidebar.selectbox(
        "Seleccione una opción:",
        ["📊 Dashboard Principal", "👤 Análisis Individual", "📁 Carga Masiva", "📈 Visualización Avanzada", "🧠 Aprendizaje Continuo"]
    )

    # Mostrar estadísticas de aprendizaje en sidebar
    if len(nuevos_datos) > 0:
        st.sidebar.markdown("---")
        st.sidebar.markdown("### 🧠 **Datos de Aprendizaje**")
        st.sidebar.metric("Nuevos Clientes", len(nuevos_datos))
        validados = sum(1 for _, row in nuevos_datos.iterrows() if row.get('validado', False))
        st.sidebar.metric("Validados", validados)
        if len(datos_combinados) > 0:
            st.sidebar.metric("Precisión Modelo", f"{(len(datos_originales) + validados) / len(datos_combinados) * 100:.1f}%")

    # Dashboard Principal
    if opcion == "📊 Dashboard Principal":
        # Mostrar métricas incluyendo nuevos datos
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Clientes", len(datos_combinados), delta=len(nuevos_datos) if len(nuevos_datos) > 0 else None)
        with col2:
            st.metric("Cluster 0", len(datos_con_clusters[datos_con_clusters['cluster'] == 0]))
        with col3:
            st.metric("Cluster 1", len(datos_con_clusters[datos_con_clusters['cluster'] == 1]))
        with col4:
            st.metric("Cluster 2", len(datos_con_clusters[datos_con_clusters['cluster'] == 2]))
        
        # Gráfico principal
        st.subheader("📈 Distribución de Clusters (Incluyendo Nuevos Datos)")
        
        # Crear marcadores de tipo correctamente
        datos_viz = datos_con_clusters.copy()
        
        # Crear lista de tipos con la longitud correcta
        tipos = []
        tipos.extend(['Original'] * len(datos_originales))
        if len(nuevos_datos) > 0:
            tipos.extend(['Nuevo'] * len(nuevos_datos))
        
        # Solo agregar la columna si las longitudes coinciden
        if len(tipos) == len(datos_viz):
            datos_viz['tipo'] = tipos
            
            fig = px.scatter(
                datos_viz, 
                x='saldoCuentaAhorro', 
                y='frecuenciaUsoMensual',
                color='cluster',
                symbol='tipo',
                hover_data=['nombre'],
                title="Segmentación de Clientes (Original + Nuevos)",
                labels={'saldoCuentaAhorro': 'Saldo Cuenta Ahorro', 'frecuenciaUsoMensual': 'Frecuencia Uso Mensual'}
            )
        else:
            fig = px.scatter(
                datos_viz, 
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
                if len(cluster_data) > 0:
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
            
            # Opción para guardar cliente
            guardar_cliente = st.checkbox("💾 Guardar cliente para mejorar el modelo", value=True)
            
            if st.button("🔍 Analizar Cliente", type="primary"):
                if nombre:
                    # Predecir cluster
                    cluster_predicho = predecir_cluster(saldo, frecuencia, kmeans, scaler)
                    
                    # Guardar cliente si está marcado
                    if guardar_cliente:
                        cliente_guardado = agregar_nuevo_cliente(nombre, saldo, frecuencia)
                        # Actualizar la predicción en el cliente guardado
                        st.session_state.nuevos_clientes[-1]['cluster_predicho'] = cluster_predicho
                        st.info("💾 Cliente guardado para mejorar el modelo")
                    
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
                else:
                    st.warning("⚠️ Por favor ingresa el nombre del cliente")

    # Aprendizaje Continuo (CON GITHUB AUTOMÁTICO)
    elif opcion == "🧠 Aprendizaje Continuo":
        st.subheader("🧠 Gestión de Aprendizaje Continuo")
        
        # Mostrar nuevos clientes
        if len(nuevos_datos) > 0:
            st.write("### 📋 Nuevos Clientes Registrados")
            
            # Convertir a DataFrame para mostrar
            display_df = nuevos_datos.copy()
            st.dataframe(display_df)
            
            # Validación de predicciones
            st.write("### ✅ Validación de Predicciones")
            st.write("**Valida las predicciones del modelo para mejorar su precisión:**")
            
            for idx, (_, cliente) in enumerate(nuevos_datos.iterrows()):
                if not cliente.get('validado', False):
                    with st.expander(f"Validar: {cliente['nombre']} - Cluster Predicho: {cliente.get('cluster_predicho', 'No predicho')}"):
                        col1, col2, col3 = st.columns(3)
                        
                        with col1:
                            st.write(f"**Saldo:** ${cliente['saldoCuentaAhorro']:,.0f}")
                            st.write(f"**Frecuencia:** {cliente['frecuenciaUsoMensual']}")
                            st.write(f"**Predicción:** Cluster {cliente.get('cluster_predicho', 'No predicho')}")
                        
                        with col2:
                            cluster_real = st.selectbox(
                                "¿Cuál es el cluster correcto?",
                                [0, 1, 2],
                                key=f"cluster_real_{idx}",
                                help="Basado en el comportamiento real del cliente"
                            )
                        
                        with col3:
                            if st.button(f"✅ Validar", key=f"validar_{idx}"):
                                # Actualizar en session_state
                                st.session_state.nuevos_clientes[idx]['cluster_real'] = cluster_real
                                st.session_state.nuevos_clientes[idx]['validado'] = True
                                st.success("✅ Cliente validado")
                                st.rerun()
            
            # Estadísticas de precisión
            st.write("### 📊 Estadísticas de Precisión")
            
            validados = [c for c in st.session_state.nuevos_clientes if c.get('validado', False)]
            if len(validados) > 0:
                correctas = sum(1 for c in validados if c.get('cluster_predicho') == c.get('cluster_real'))
                precision = correctas / len(validados) * 100 if len(validados) > 0 else 0
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Predicciones Validadas", len(validados))
                with col2:
                    st.metric("Predicciones Correctas", correctas)
                with col3:
                    st.metric("Precisión del Modelo", f"{precision:.1f}%")
            
            # 🚀 NUEVA SECCIÓN: GITHUB AUTOMÁTICO
            st.write("### 🚀 Actualización Automática de GitHub")
            
            validados_df = nuevos_datos[nuevos_datos['validado'] == True]
            if len(validados_df) > 0:
                col1, col2 = st.columns(2)
                
                with col1:
                    st.info(f"📊 **{len(validados_df)} clientes validados** listos para guardar en GitHub")
                    
                    if GITHUB_TOKEN:
                        if st.button("🤖 Guardar en GitHub Automáticamente", type="primary"):
                            datos_actualizados = generar_csv_actualizado(datos_originales, nuevos_datos)
                            
                            if actualizar_github_csv(datos_actualizados):
                                # Limpiar datos de sesión después de guardar exitosamente
                                st.session_state.nuevos_clientes = []
                                st.balloons()  # ¡Celebración!
                                
                                # Forzar recarga de datos
                                st.cache_data.clear()
                                
                                # Mensaje de éxito
                                st.success("🎉 ¡Datos guardados exitosamente!")
                                st.info("🔄 Recargando la aplicación...")
                                
                                # Recargar página
                                st.rerun()
                    else:
                        st.error("❌ Token de GitHub no configurado")
                
                with col2:
                    # Mostrar preview de lo que se guardará
                    if st.button("👀 Ver Preview de Datos"):
                        datos_actualizados = generar_csv_actualizado(datos_originales, nuevos_datos)
                        st.write(f"**Se agregarán {len(validados_df)} nuevos clientes:**")
                        st.dataframe(validados_df[['nombre', 'saldoCuentaAhorro', 'frecuenciaUsoMensual', 'cluster_real']])
                        st.write(f"**Total clientes después de actualizar:** {len(datos_actualizados)}")
            else:
                st.warning("⚠️ Valida algunos clientes primero para poder guardar en GitHub")
            
            # Opciones manuales como backup
            st.write("### 💾 Opciones de Respaldo")
            
            col1, col2 = st.columns(2)
            
            with col1:
                if st.button("📥 Descargar Nuevos Clientes"):
                    csv = nuevos_datos.to_csv(index=False)
                    st.download_button(
                        label="💾 Descargar CSV",
                        data=csv,
                        file_name=f'nuevos_clientes_{datetime.now().strftime("%Y%m%d_%H%M%S")}.csv',
                        mime='text/csv'
                    )
            
            with col2:
                if len(validados_df) > 0:
                    if st.button("🚀 Descargar Dataset Completo"):
                        datos_actualizados = generar_csv_actualizado(datos_originales, nuevos_datos)
                        csv_actualizado = datos_actualizados.to_csv(index=False)
                        st.download_button(
                            label="💾 Descargar Dataset Actualizado",
                            data=csv_actualizado,
                            file_name=f'segmentacionbancaria_actualizado_{datetime.now().strftime("%Y%m%d_%H%M%S")}.csv',
                            mime='text/csv'
                        )
        else:
            st.info("📝 No hay nuevos clientes registrados aún. Usa 'Análisis Individual' para agregar clientes.")

    # Resto de secciones (Carga Masiva y Visualización Avanzada) 
    # [Mantener el código anterior para estas secciones]

else:
    st.error("❌ No se pudieron cargar los datos desde GitHub")

# Footer
st.markdown("---")
if GITHUB_TOKEN:
    st.markdown(f"**🏦 Sistema de Segmentación Bancaria con GitHub Automático** ✅ | Clientes en sesión: {len(nuevos_datos)}")
else:
    st.markdown(f"**🏦 Sistema de Segmentación Bancaria** ❌ GitHub desconectado | Clientes en sesión: {len(nuevos_datos)}")
