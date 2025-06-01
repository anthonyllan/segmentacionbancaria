# 🏦 Sistema de Segmentación Bancaria

Sistema web para segmentación automática de clientes bancarios usando K-means clustering.

## 🚀 Instalación y Ejecución

1. **Instalar dependencias:**
```bash
pip install -r requirements.txt
```

2. **Ejecutar aplicación:**
```bash
streamlit run app.py
```

3. **O usar el script automático:**
```bash
python run_app.py
```

## 📱 Funcionalidades

### 📊 Dashboard Principal
- Vista general de todos los clusters
- Métricas de clientes por segmento
- Visualización interactiva

### 👤 Análisis Individual
- Formulario para evaluar cliente individual
- Predicción de cluster en tiempo real
- Recomendaciones personalizadas de productos

### 📁 Carga Masiva
- Carga de archivos CSV con múltiples clientes
- Procesamiento automático por lotes
- Descarga de resultados

### 📈 Visualización Avanzada
- Método del codo para validar clusters
- Distribuciones estadísticas
- Análisis de centroides

## 🎯 Clusters y Productos

### Cluster 0: Cliente Premium
- **Perfil:** Alto saldo, baja frecuencia
- **Productos:** Inversiones, seguros premium, hipotecarios

### Cluster 1: Cliente Estándar
- **Perfil:** Bajo saldo, baja frecuencia
- **Productos:** Cuentas básicas, microcréditos

### Cluster 2: Cliente Digital
- **Perfil:** Saldo medio, alta frecuencia
- **Productos:** Banca digital, servicios premium

## 🔧 Tecnologías

- **Streamlit:** Interfaz web
- **scikit-learn:** Modelo de clustering
- **Plotly:** Visualizaciones interactivas
- **Pandas:** Procesamiento de datos
