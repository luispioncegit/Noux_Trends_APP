# Especificaciones Técnicas: Ecosistema NouxTrends
Este documento detalla la infraestructura técnica y las decisiones de ingeniería tomadas para el despliegue de la solución.

## 1. Arquitectura de Comunicación (Frontend -> Backend)
    El sistema no corre en una sola máquina; son dos servicios independientes en la nube de Render que se comunican a través de la red:
        Interacción: Cuando el usuario interactúa con el Frontend (Streamlit), este no ejecuta el código de análisis localmente. En su lugar, emite peticiones HTTP (POST/GET) hacia la URL pública del Backend en Render.
        
        Consumo de Funciones: El Backend actúa como un Motor de Cálculo. Recibe los parámetros del Frontend, procesa los CSVs, entrena los modelos y devuelve los resultados en formato JSON que el Frontend traduce en gráficos.
        
        Dependencia: El Frontend requiere que el Backend esté "despierto" y conectado para que cualquier función de la página (filtros, análisis, forecasting) sea operativa.
        
## 2. Configuración de Despliegue (Dashboard Render)
    La ruta de ejecución debe ser exacta para que Render localice los archivos:
    Parámetro       Valor Backend                       Valor Frontend
    Start Command   uvicorn backend.main:app --         streamlit run frontend/main.py --
                    host 0.0.0.0 --port $PORT           server.port $PORT
    Env Var         N/A                                 BACKEND_URL=https://tu-api.onrender.com/api/v1 
    
## 3. Gestión de Rutas y Datos (Pathlib)
    Para evitar errores de "File Not Found", el Backend utiliza rutas absolutas dinámicas calculadas desde la posición del script endpoints.py:
    
    Python (Ubicado en backend/api/endpoints.py)
    
    BASE_BACKEND_DIR = Path(__file__).resolve().parent.parent # Carpeta 'backend'
    DATA_PATH = BASE_BACKEND_DIR / "data"

## 4. Protocolos de Resiliencia (Keep-Alive)
    Debido a que Render Free apaga los servicios tras 15 minutos de inactividad:
    
###     4.1. Función de Reintento (Frontend): 
        Se creó get_health_with_retry en el Frontend. Si el Backend está dormido, el sistema reintenta 3 veces (esperando 5s entre cada uno) para darle tiempo al           Backend de arrancar.
    
###     4.2. Cron-Job Externo: 
        Un servicio de terceros visita el endpoint /health cada 10 minutos para prevenir la suspensión del servicio.
    

###     4.3. Timeouts Extendidos: 
        Las peticiones de Forecasting tienen un timeout=300 para permitir procesos pesados de Machine Learning sin cortes de conexión.
