import traceback  # AGREGAR ESTE IMPORT
from fastapi import APIRouter, HTTPException
from sqlalchemy import create_engine, text
import pandas as pd
import logging
from core.analyzer import NouxTrendsAnalyzer
from core.models import AnalysisRequest, AnalysisResponse, FilterOptions
from core.config import DATABASE_URL, SCHEMA
from core.models import TrainModelsRequest, TrainModelsResponse, ForecastRequest, ForecastResponse
from core.forecaster import Forecaster


logger = logging.getLogger(__name__)
router = APIRouter()

# Inicializar forecaster global
forecaster = None

@router.on_event("startup")
async def startup_event():
    global forecaster
    try:
        df_ventas = load_ventas_data()  # Tu función existente
        forecaster = Forecaster(df_ventas)
        print("✅ Forecaster inicializado")
    except Exception as e:
        print(f"❌ Error inicializando forecaster: {str(e)}")
        forecaster = None

# Crear engine y cargar datos una sola vez
engine = create_engine(DATABASE_URL)

def load_ventas_data():
    """Carga datos de ventas desde PostgreSQL"""
    try:
        query = f"""
            WITH productos_all AS (
                SELECT P.*, C.nombre_categoria
                FROM mock_data_ventas.dim_productos P
                LEFT JOIN mock_data_ventas.dim_categoria C 
                    ON P.id_categoria = C.id_categoria 
            )
            SELECT 
                A.*, 
                L.nombre_local AS local, 
                P.nombre_articulo AS articulo, 
                P.nombre_categoria AS categoria, 
                L.tamaño_local, 
                L.ubicacion_local
            FROM mock_data_ventas.ventas A 
            LEFT JOIN mock_data_ventas.dim_local L 
                ON A.id_local = L.id_local
            LEFT JOIN productos_all P 
                ON A.id_articulo = P.id_articulo;
        """
        df = pd.read_sql(query, engine)
        
        # Convertir fecha a datetime si es necesario
        if 'fecha' in df.columns:
            df['fecha'] = pd.to_datetime(df['fecha'])
        
        logger.info(f"✅ Datos cargados: {len(df)} registros")
        logger.info(f"📅 Rango: {df['fecha'].min()} a {df['fecha'].max()}")
        
        return df
    except Exception as e:
        logger.error(f"❌ Error cargando datos: {str(e)}")
        raise

def get_database_stats():
    """Obtiene estadísticas simples de la BD"""
    try:
        query = text(f"""
            SELECT 
            COUNT(*) as total_registros,
            MIN(fecha) as fecha_minima,
            MAX(fecha) as fecha_maxima,
            COUNT(DISTINCT L.nombre_local) as locales_unicos,
            COUNT(DISTINCT P.nombre_articulo) as articulos_unicos
            FROM {SCHEMA}.ventas A 
            left join {SCHEMA}.dim_local L on A.id_local = L.id_local
            left join {SCHEMA}.dim_productos P on A.id_articulo = P.id_articulo
        """)
        
        with engine.connect() as conn:
            result = conn.execute(query).fetchone()
            
        return {
            "total_registros": result[0],
            "fecha_minima": result[1],
            "fecha_maxima": result[2],
            "locales_unicos": result[3],
            "articulos_unicos": result[4]
        }
    except Exception as e:
        logger.error(f"Error en stats: {str(e)}")
        return {}

# Inicializar el analyzer con datos de PostgreSQL
try:
    df_ventas = load_ventas_data()
    analyzer = NouxTrendsAnalyzer(df_ventas)
    logger.info("✅ NouxTrendsAnalyzer inicializado")
except Exception as e:
    logger.error(f"❌ Error inicializando analyzer: {str(e)}")
    analyzer = None

############## Filter Options Endpoint #################

@router.get("/filters", response_model=FilterOptions)
async def get_filter_options():
    """Obtiene opciones disponibles para filtros"""
    if analyzer is None:
        raise HTTPException(status_code=500, detail="Analyzer no inicializado")
    return analyzer.get_filter_options()

############## Data Analysis Endpoint #################

@router.post("/analyze", response_model=AnalysisResponse)
async def analyze_data(request: AnalysisRequest):
    """Endpoint principal de análisis"""
    if analyzer is None:
        raise HTTPException(status_code=500, detail="Analyzer no inicializado")
    
    try:
        print("=== INICIANDO ANÁLISIS ===")  # DEBUG
        print(f"Request: {request}")  # DEBUG
        
        results = analyzer.analyze_time_series(request)
        
        print("=== ANÁLISIS COMPLETADO ===")  # DEBUG
        return AnalysisResponse(
            success=True,
            message="Análisis completado exitosamente",
            data=results
        )
    except Exception as e:
        print(f"❌ ERROR EN ANÁLISIS: {str(e)}")  # DEBUG
        print(traceback.format_exc())  # DEBUG - esto muestra el stack trace completo
        logger.error(f"Error en análisis: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))



################## Database Info Endpoint #################

@router.get("/database/info")
async def database_info():
    """Info de la base de datos"""
    try:
        stats = get_database_stats()
        return {
            "database": "PostgreSQL",
            "schema": SCHEMA,
            "table": "ventas",
            "stats": stats
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    

################# Forecasting Endpoints #################

@router.post("/forecast/train", response_model=TrainModelsResponse)
async def train_forecast_models(request: TrainModelsRequest):
    """Endpoint para entrenar modelos de forecasting"""
    if forecaster is None:  # ✅ Cambiar de 'analyzer' a 'forecaster'
        raise HTTPException(status_code=500, detail="Forecaster no inicializado")
    
    try:
        print(f"=== ENTRENANDO MODELOS LIGHTGBM ===")
        print(f"Variables: {request.variables_categoricas}")
        
        # ✅ Llamar directamente al forecaster
        resultados = forecaster.entrenar_modelos(
            variables=request.variables_categoricas,
            modelos=request.modelos_a_entrenar,  # Aunque solo usamos LightGBM ahora
            train_size=request.porcentaje_entrenamiento
        )
        
        return TrainModelsResponse(
            success=True,
            message="Modelo LightGBM entrenado exitosamente",
            modelos_entrenados=resultados.get('modelos_entrenados', []),
            metricas_entrenamiento=resultados.get('metricas', {}),
            fecha_entrenamiento=resultados.get('fecha_entrenamiento', ''),
            datos_grafico=resultados.get('datos_grafico', {})  # ✅ NUEVO: incluir datos del gráfico
        )
    except Exception as e:
        print(f"❌ ERROR ENTRENANDO MODELOS: {str(e)}")
        logger.error(f"Error entrenando modelos: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/forecast/models")
async def get_trained_models():
    """Obtiene lista de modelos entrenados disponibles"""
    if forecaster is None:
        raise HTTPException(status_code=500, detail="Forecaster no inicializado")
    
    try:
        # Llamar directamente al forecaster
        modelos_info = forecaster.get_modelos_disponibles()
        
        return {
            "modelos_disponibles": modelos_info.get("modelos_disponibles", []),
            "modelos_info": modelos_info.get("modelos_info", {}),
            "total_modelos": modelos_info.get("total_modelos", 0),
            "modelos_entrenados": modelos_info.get("modelos_entrenados", False)  # ✅ Agregar este campo
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    
    

@router.post("/forecast/multi-level", response_model=ForecastResponse)
async def generate_multi_level_forecast(request: ForecastRequest):
    """Endpoint para pronóstico multi-nivel con LightGBM"""
    if forecaster is None:  # ✅ Cambiar de 'analyzer' a 'forecaster'
        raise HTTPException(status_code=500, detail="Forecaster no inicializado")
    
    try:
        print(f"=== PRONÓSTICO MULTI-NIVEL LIGHTGBM ===")
        print(f"Variables: {request.variables_categoricas}")
        print(f"Modelo: {request.modelo}")
        print(f"Horizonte: {request.horizonte}")
        
        # ✅ Llamar directamente al forecaster
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
        print(f"❌ ERROR PRONÓSTICO MULTI-NIVEL: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
    

############# Health Check Endpoint #################

@router.get("/health")  
async def health_check():
    """Health check simple"""
    if analyzer is None:
        return {"status": "unhealthy", "error": "Analyzer no inicializado"}
    
    try:
        stats = get_database_stats()
        return {
            "status": "healthy", 
            "database_stats": stats
        }
    except Exception as e:
        return {"status": "degraded", "error": str(e)}
