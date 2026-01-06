import traceback
import os
from fastapi import APIRouter, HTTPException
import pandas as pd
import logging
from core.analyzer import NouxTrendsAnalyzer
from core.models import AnalysisRequest, AnalysisResponse, FilterOptions
from core.models import TrainModelsRequest, TrainModelsResponse, ForecastRequest, ForecastResponse
from core.forecaster import Forecaster

logger = logging.getLogger(__name__)
router = APIRouter()

# --- CONFIGURACIÓN DE RUTAS ---
# 1. Buscamos la ruta de este archivo actual
current_dir = os.path.dirname(os.path.abspath(__file__))

# 2. Subimos UN nivel para llegar a la carpeta 'backend'
BACKEND_DIR = os.path.dirname(CURRENT_DIR)

# 3. Ahora bajamos a la carpeta 'data'
DATA_PATH = os.path.join(BACKEND_DIR, "data")

# Verificación en logs
logger.info(f"📂 Buscando archivos en: {DATA_PATH}")

# Inicializar forecaster y analyzer globales
forecaster = None
analyzer = None

def load_ventas_data():
    """Carga datos de ventas desde el archivo CSV consolidado"""
    try:
        csv_file = os.path.join(DATA_PATH, "ventas_full.csv")
        logger.info(f"📂 Intentando cargar CSV desde: {csv_file}")
        
        # Leemos el CSV (que ya trae todos los JOINs hechos desde SQL)
        df = pd.read_csv(csv_file)
        
        # Convertir fecha a datetime
        if 'fecha' in df.columns:
            df['fecha'] = pd.to_datetime(df['fecha'])
        
        logger.info(f"✅ Datos cargados: {len(df)} registros")
        return df
    except Exception as e:
        logger.error(f"❌ Error cargando CSV de ventas: {str(e)}")
        raise

def get_database_stats():
    """Obtiene estadísticas desde el archivo CSV de stats"""
    try:
        csv_stats = os.path.join(DATA_PATH, "stats_db.csv")
        df_stats = pd.read_csv(csv_stats)
        
        # Asumiendo que el CSV tiene las columnas: 
        # total_registros, fecha_minima, fecha_maxima, locales_unicos, articulos_unicos
        result = df_stats.iloc[0]
        
        return {
            "total_registros": int(result['total_registros']),
            "fecha_minima": str(result['fecha_minima']),
            "fecha_maxima": str(result['fecha_maxima']),
            "locales_unicos": int(result['locales_unicos']),
            "articulos_unicos": int(result['articulos_unicos'])
        }
    except Exception as e:
        logger.error(f"❌ Error en stats CSV: {str(e)}")
        return {
            "total_registros": 0,
            "fecha_minima": "N/A",
            "fecha_maxima": "N/A",
            "locales_unicos": 0,
            "articulos_unicos": 0
        }

@router.on_event("startup")
async def startup_event():
    """Inicialización al arrancar la API"""
    global forecaster, analyzer
    try:
        df_ventas = load_ventas_data()
        
        # Inicializar Analizador
        analyzer = NouxTrendsAnalyzer(df_ventas)
        logger.info("✅ NouxTrendsAnalyzer inicializado con CSV")
        
        # Inicializar Forecaster
        forecaster = Forecaster(df_ventas)
        logger.info("✅ Forecaster inicializado con CSV")
        
    except Exception as e:
        logger.error(f"❌ Error crítico en startup: {str(e)}")
        print(traceback.format_exc())

############## Filter Options Endpoint #################

@router.get("/filters", response_model=FilterOptions)
async def get_filter_options():
    if analyzer is None:
        raise HTTPException(status_code=500, detail="Analyzer no inicializado")
    return analyzer.get_filter_options()

############## Data Analysis Endpoint #################

@router.post("/analyze", response_model=AnalysisResponse)
async def analyze_data(request: AnalysisRequest):
    if analyzer is None:
        raise HTTPException(status_code=500, detail="Analyzer no inicializado")
    
    try:
        results = analyzer.analyze_time_series(request)
        return AnalysisResponse(
            success=True,
            message="Análisis completado exitosamente",
            data=results
        )
    except Exception as e:
        logger.error(f"Error en análisis: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

################## Database Info Endpoint #################

@router.get("/database/info")
async def database_info():
    """Info simulada de base de datos (ahora CSV)"""
    try:
        stats = get_database_stats()
        return {
            "database": "CSV Engine (Production)",
            "schema": "data_folder",
            "table": "ventas_full.csv",
            "stats": stats
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

################# Forecasting Endpoints #################

@router.post("/forecast/train", response_model=TrainModelsResponse)
async def train_forecast_models(request: TrainModelsRequest):
    if forecaster is None:
        raise HTTPException(status_code=500, detail="Forecaster no inicializado")
    
    try:
        resultados = forecaster.entrenar_modelos(
            variables=request.variables_categoricas,
            modelos=request.modelos_a_entrenar,
            train_size=request.porcentaje_entrenamiento
        )
        
        return TrainModelsResponse(
            success=True,
            message="Modelo LightGBM entrenado exitosamente",
            modelos_entrenados=resultados.get('modelos_entrenados', []),
            metricas_entrenamiento=resultados.get('metricas', {}),
            fecha_entrenamiento=resultados.get('fecha_entrenamiento', ''),
            datos_grafico=resultados.get('datos_grafico', {})
        )
    except Exception as e:
        logger.error(f"Error entrenando modelos: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/forecast/models")
async def get_trained_models():
    if forecaster is None:
        raise HTTPException(status_code=500, detail="Forecaster no inicializado")
    try:
        modelos_info = forecaster.get_modelos_disponibles()
        return {
            "modelos_disponibles": modelos_info.get("modelos_disponibles", []),
            "modelos_info": modelos_info.get("modelos_info", {}),
            "total_modelos": modelos_info.get("total_modelos", 0),
            "modelos_entrenados": modelos_info.get("modelos_entrenados", False)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/forecast/multi-level", response_model=ForecastResponse)
async def generate_multi_level_forecast(request: ForecastRequest):
    if forecaster is None:
        raise HTTPException(status_code=500, detail="Forecaster no inicializado")
    
    try:
        resultados = forecaster.generar_pronosticos_multi_nivel(
            variables=request.variables_categoricas,
            modelo_principal=request.modelo,
            horizonte=request.horizonte
        )
        return ForecastResponse(
            success=True,
            message="Pronóstico multi-nivel generado exitosamente",
            data=resultados,
            dimension_ancla=resultados.get('dimension_ancla', 'total'),
            metricas_calidad=resultados.get('metricas_calidad', {})
        )
    except Exception as e:
        logger.error(f"Error pronóstico: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

############# Health Check Endpoint #################

@router.get("/health")
async def health_check():
    if analyzer is None or forecaster is None:
        return {"status": "unhealthy", "error": "Motores no inicializados"}
    try:
        stats = get_database_stats()
        return {
            "status": "healthy", 
            "source": "CSV Files",
            "database_stats": stats
        }
    except Exception as e:
        return {"status": "degraded", "error": str(e)}



