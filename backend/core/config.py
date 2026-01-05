import os
from dotenv import load_dotenv

# Cargamos el .env solo si existe (en local), en Render se usan las variables del panel
load_dotenv()

# --- CONFIGURACIÓN DE LA APLICACIÓN ---
# Render asigna automáticamente un puerto dinámico en la variable $PORT
# Si no existe (local), usamos el 8000 por defecto
APP_HOST = os.getenv("APP_HOST", "0.0.0.0")
APP_PORT = int(os.getenv("PORT", 8000)) 

# Debug se desactiva en producción por defecto
DEBUG = os.getenv("DEBUG", "False").lower() == "true"

# --- CONFIGURACIÓN DE DATOS ---
# Definimos nombres de archivos fijos para los CSV
VENTAS_CSV = "ventas_full.csv"
STATS_CSV = "stats_db.csv"

# Mantenemos esta variable vacía o con un nombre genérico para no romper imports en otros archivos
SCHEMA = "data_folder"
DATABASE_URL = "sqlite:///./dummy.db" # URL ficticia por si algún script aún la importa
