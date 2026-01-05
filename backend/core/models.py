from pydantic import BaseModel
from typing import List, Optional, Dict, Any
from datetime import datetime

################ Data Analysis models #################
class AnalysisRequest(BaseModel):
    columna_valor: str = "venta"
    filtros: Optional[Dict[str, Any]] = None  # Ya está como Optional
    grupo_principal: str = "local"
    fecha_inicio: Optional[datetime] = None
    fecha_fin: Optional[datetime] = None


class FilterOptions(BaseModel):
    local: List[str]
    articulo: List[str]
    categoria: List[str]
    tamaño_local: List[str]
    ubicacion_local: List[str]


class AnalysisResponse(BaseModel):
    success: bool
    message: str
    data: Optional[Dict[str, Any]] = None
    nsights: Optional[List[str]] = None  # Nuevo campo para insights

############### Forecasting models #################
class TrainModelsRequest(BaseModel):
    variables_categoricas: List[str] = ["local", "categoria"]
    modelos_a_entrenar: List[str] = ["LightGBM"]  # ✅ Solo LightGBM ahora
    porcentaje_entrenamiento: float = 0.8

class TrainModelsResponse(BaseModel):
    success: bool
    message: str
    modelos_entrenados: List[str]
    metricas_entrenamiento: Dict[str, Any]
    fecha_entrenamiento: str
    datos_grafico: Optional[Dict[str, Any]] = None  # ✅ NUEVO: campo para gráfico

class ForecastRequest(BaseModel):
    horizonte: int = 30
    modelo: str = "lightgbm_global"  # ✅ Modelo por defecto
    variables_categoricas: List[str] = ["local", "categoria"]
    intervalo_confianza: float = 0.9
    filtros: Optional[Dict[str, Any]] = None

class ForecastResponse(BaseModel):
    success: bool
    message: str
    data: Optional[Dict[str, Any]] = None
    dimension_ancla: Optional[str] = None
    metricas_calidad: Optional[Dict[str, Any]] = None