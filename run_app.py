import subprocess
import sys
import os

def run_app():
    """Ejecutar la aplicación Streamlit"""
    # Verificar que app.py existe
    if os.path.exists("app.py"):
        print("✅ Encontrado app.py, ejecutando...")
        # En Streamlit Cloud, solo necesitamos importar y ejecutar
        import app
    else:
        print("❌ Error: app.py no encontrado en el directorio")
        print("📁 Archivos disponibles:")
        for file in os.listdir("."):
            print(f"   - {file}")
        sys.exit(1)

if __name__ == "__main__":
    print("🚀 Iniciando aplicación de segmentación bancaria...")
    run_app()
