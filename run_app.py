import subprocess
import sys

def install_requirements():
    """Instalar dependencias"""
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])

def run_app():
    """Ejecutar la aplicación Streamlit"""
    subprocess.run(["streamlit", "run", "app.py"])

if __name__ == "__main__":
    print("🏦 Instalando dependencias...")
    install_requirements()
    print("🚀 Iniciando aplicación...")
    run_app()