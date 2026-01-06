import streamlit as st
import requests
import base64
from PIL import Image
import io
import os

# Configuración de página
st.set_page_config(
    page_title="NouxTrends - Análisis",
    page_icon="📊",
    layout="wide",
)

# Cambiar esto:
# BACKEND_URL = "http://localhost:8002/api/v1"

# Por esto:
BACKEND_URL = os.environ.get("BACKEND_URL", "http://localhost:8002/api/v1")
# Creamos una variable para la raíz (útil para el health check)
BASE_URL = BACKEND_URL.replace("/api/v1", "")

def render_analysis_sidebar():
    """Sidebar personalizado para Analysis CON filtros"""
    with st.sidebar:
        st.header("🚀 NouxTrends")
        st.markdown("---")
        
        # Info del sistema
        backend_online = False
        try:
            response = requests.get(BASE_URL, timeout=3) # <--- USAR BASE_URL
            if response.status_code == 200:
                st.success("✅ Sistema Conectado")
                backend_online = True
            else:
                st.error("❌ Sistema Offline")
        except:
            st.error("❌ Sistema Offline")
        
        st.markdown("---")
        st.header("🔍 Filtros de Datos")
        
        filters = {}
        grupo_principal = 'local'
        ejecutar_analisis = False
        
        # Solo mostrar filtros si el backend está online
        if backend_online:
            try:
                response = requests.get(f"{BACKEND_URL}/filters")
                if response.status_code == 200:
                    filter_options = response.json()
                    
                    # Filtro por local
                    locales = st.multiselect(
                        "Locales",
                        options=filter_options.get('local', []),
                        help="Selecciona uno o más locales"
                    )
                    if locales:
                        filters['local'] = locales
                        
                    # Filtro por categoría
                    categorias = st.multiselect(
                        "Categorías",
                        options=filter_options.get('categoria', []),
                        help="Selecciona una o más categorías"
                    )
                    if categorias:
                        filters['categoria'] = categorias
                        
                    # Filtro por artículo
                    articulos = st.multiselect(
                        "Artículos",
                        options=filter_options.get('articulo', []),
                        help="Selecciona uno o más artículos"
                    )
                    if articulos:
                        filters['articulo'] = articulos
                        
                    # Grupo principal para análisis
                    grupo_principal = st.selectbox(
                        "Agrupar por",
                        options=['local', 'articulo', 'categoria']
                    )
                    
                    # Botón para ejecutar análisis
                    ejecutar_analisis = st.button("🚀 Ejecutar Análisis", type="primary", use_container_width=True)
                    
                else:
                    st.error(f"Error del backend: {response.status_code}")
            except Exception as e:
                st.error(f"No se pudo cargar los filtros: {str(e)}")
        else:
            st.info("⏳ Conectando con el backend...")
        
        # SIEMPRE retornar la estructura completa, incluso si hay errores
        return {
            'filters': filters,
            'grupo_principal': grupo_principal,
            'ejecutar_analisis': ejecutar_analisis
        }

def main():
    # Sidebar personalizado CON filtros
    filters_config = render_analysis_sidebar()
    
    st.title("📊 Análisis Descriptivo")
    st.markdown("Análisis completo de datos históricos de ventas con visualizaciones interactivas")
    
    # Mostrar filtros seleccionados
    if filters_config['filters']:
        st.write("**Filtros aplicados:**", filters_config['filters'])
    else:
        st.info("🌐 No hay filtros aplicados - mostrando todos los datos")
    
    # Ejecutar análisis si se presionó el botón
    if filters_config['ejecutar_analisis']:
        ejecutar_analisis_y_mostrar_resultados(filters_config)
    else:
        st.warning("⚙️ Configura los filtros en el sidebar y haz clic en '🚀 Ejecutar Análisis' para ver los resultados")

def ejecutar_analisis_y_mostrar_resultados(filters_config):
    """Ejecuta el análisis y muestra los resultados"""
    
    # Preparar payload para la API
    payload = {
        "columna_valor": "venta",
        "filtros": filters_config['filters'],
        "grupo_principal": filters_config['grupo_principal']
    }
    
    with st.spinner("🔄 Ejecutando análisis..."):
        try:
            # Llamar al endpoint de análisis
            response = requests.post(f"{BACKEND_URL}/analyze", json=payload)
            
            if response.status_code == 200:
                resultados = response.json()
                mostrar_resultados(resultados)
            else:
                st.error(f"❌ Error en el análisis: {response.status_code}")
                st.write("Detalles:", response.json())
                
        except Exception as e:
            st.error(f"❌ Error conectando con el backend: {str(e)}")



def mostrar_resultados(resultados):
    """Muestra los resultados del análisis"""
    
    st.success("✅ Análisis completado exitosamente!")
    st.markdown("---")
    
    data = resultados.get('data', {})

    # Mostrar insights automáticos (NUEVO)
    if 'insights' in resultados and resultados['insights']:
        st.subheader("💡 Insights Automáticos")
        for insight in resultados['insights']:
            st.info(insight)
        st.markdown("---")
    
    # Mostrar métricas
    if 'metrics' in data:
        mostrar_metricas_avanzadas(data['metrics'])
    
    # Mostrar gráficos
    if 'charts' in data:
        mostrar_graficos_avanzados(data['charts'])
    
    # Mostrar info del dataset
    if 'filtered_data_info' in data:
        mostrar_info_dataset(data['filtered_data_info'])

def mostrar_metricas_avanzadas(metrics):
    """Muestra las métricas avanzadas organizadas en secciones"""
    
    # 1. MÉTRICAS BÁSICAS
    st.subheader("📊 Métricas Básicas")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            label="Total Ventas", 
            value=f"${metrics.get('total_ventas', 0):,.2f}",
            help="Suma total de todas las ventas en el período analizado"
        )
    with col2:
        st.metric(
            label="Promedio Diario", 
            value=f"${metrics.get('promedio_diario', 0):,.2f}",
            help="Promedio de ventas por día"
        )
    with col3:
        st.metric(
            label="Días Analizados", 
            value=f"{metrics.get('dias_analizados', 0):,}",
            help="Número total de días en el análisis"
        )
    with col4:
        st.metric(
            label="Locales Únicos", 
            value=f"{metrics.get('locales_unicos', 0):,}",
            help="Cantidad de locales distintos en los datos"
        )
    
    st.markdown("---")
    
    # 2. MÉTRICAS POR TIPO DE DÍA
    st.subheader("🎯 Comportamiento por Tipo de Día")
    
    # Crear 3 columnas para los tipos de día
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("#### 🏢 Días Laborales")
        st.metric(
            label="Promedio", 
            value=f"${metrics.get('promedio_laboral', 0):,.2f}",
            help="Ventas promedio en días laborales"
        )
        st.metric(
            label="% del Total", 
            value=f"{metrics.get('porcentaje_laboral', 0):.1f}%",
            help="Porcentaje de ventas totales en días laborales"
        )
        st.metric(
            label="Días", 
            value=f"{metrics.get('dias_laborales', 0):,}",
            help="Cantidad de días laborales analizados"
        )
    
    with col2:
        st.markdown("#### 🎉 Fin de Semana")
        st.metric(
            label="Promedio", 
            value=f"${metrics.get('promedio_fin_semana', 0):,.2f}",
            help="Ventas promedio en fines de semana"
        )
        st.metric(
            label="% del Total", 
            value=f"{metrics.get('porcentaje_fin_semana', 0):.1f}%",
            help="Porcentaje de ventas totales en fines de semana"
        )
        st.metric(
            label="Días", 
            value=f"{metrics.get('dias_fin_semana', 0):,}",
            help="Cantidad de fines de semana analizados"
        )
    
    with col3:
        st.markdown("#### 🎊 Días Festivos")
        st.metric(
            label="Promedio", 
            value=f"${metrics.get('promedio_festivo', 0):,.2f}",
            help="Ventas promedio en días festivos"
        )
        st.metric(
            label="% del Total", 
            value=f"{metrics.get('porcentaje_festivo', 0):.1f}%",
            help="Porcentaje de ventas totales en días festivos"
        )
        st.metric(
            label="Días", 
            value=f"{metrics.get('dias_festivos', 0):,}",
            help="Cantidad de días festivos analizados"
        )
    
    st.markdown("---")
    
    # 3. MÉTRICAS AVANZADAS (solo si existen)
    if any(key in metrics for key in ['volatilidad_diaria', 'coeficiente_variacion', 'maximo_diario', 'minimo_diario']):
        st.subheader("📈 Métricas Avanzadas")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric(
                label="Volatilidad Diaria", 
                value=f"${metrics.get('volatilidad_diaria', 0):,.2f}",
                help="Desviación estándar de las ventas diarias"
            )
        with col2:
            st.metric(
                label="Coef. Variación", 
                value=f"{metrics.get('coeficiente_variacion', 0):.1f}%",
                help="Variabilidad relativa de las ventas (std/mean)"
            )
        with col3:
            st.metric(
                label="Máximo Diario", 
                value=f"${metrics.get('maximo_diario', 0):,.2f}",
                help="Ventas del día con mayor volumen"
            )
        with col4:
            st.metric(
                label="Mínimo Diario", 
                value=f"${metrics.get('minimo_diario', 0):,.2f}",
                help="Ventas del día con menor volumen"
            )
        
        # Métricas adicionales si existen
    if 'tendencia_crecimiento' in metrics:
        st.markdown("---")
        st.subheader("📊 Análisis de Tendencia")
        
        col1, col2 = st.columns(2)
        
        with col1:
            tendencia = metrics.get('tendencia_crecimiento', 0)
            if tendencia > 0:
                st.metric(
                    label="Tendencia", 
                    value="📈 Creciente",
                    delta=f"+${tendencia:,.2f} por día",
                    help="Dirección general de las ventas"
                )
            elif tendencia < 0:
                st.metric(
                    label="Tendencia", 
                    value="📉 Decreciente", 
                    delta=f"-${abs(tendencia):,.2f} por día",
                    help="Dirección general de las ventas"
                )
            else:
                st.metric(
                    label="Tendencia", 
                    value="➡️ Estable",
                    help="Dirección general de las ventas"
                )
        
        with col2:
            estacionalidad = metrics.get('estacionalidad_fuerte', 'Moderada')
            st.metric(
                label="Estacionalidad", 
                value=estacionalidad,
                help="Nivel de patrones estacionales en los datos"
            )


def mostrar_graficos_avanzados(charts):
    st.markdown("---")
    st.subheader("📊 Gráficos del Análisis")
    
    # Organizar gráficos en pestañas
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📈 Serie Temporal", 
        "🔄 Estacionalidad", 
        "📊 Estacionariedad", 
        "🔗 Correlaciones", 
        "⚡ Volatilidad"
    ])
    
    with tab1:
        if 'time_series' in charts:
            st.write("**Serie Temporal Completa**")
            mostrar_imagen_base64(charts['time_series'])
    
    with tab2:
        if 'seasonality' in charts:
            st.write("**Análisis de Estacionalidad y Días Especiales**")
            mostrar_imagen_base64(charts['seasonality'])
    
    with tab3:
        if 'stationarity' in charts:
            st.write("**Análisis de Estacionariedad**")
            mostrar_imagen_base64(charts['stationarity'])
    
    with tab4:
        if 'correlation' in charts:
            st.write("**Análisis de Correlaciones**")
            mostrar_imagen_base64(charts['correlation'])
    
    with tab5:
        if 'volatility' in charts:
            st.write("**Análisis de Volatilidad**")
            mostrar_imagen_base64(charts['volatility'])

def mostrar_imagen_base64(base64_string):
    """Convierte base64 a imagen y la muestra"""
    try:
        # Remover el prefix si existe
        if base64_string.startswith('data:image/png;base64,'):
            base64_string = base64_string.replace('data:image/png;base64,', '')
        
        # Decodificar base64
        image_data = base64.b64decode(base64_string)
        image = Image.open(io.BytesIO(image_data))
        
        # Mostrar imagen
        st.image(image, use_container_width =True)
        
    except Exception as e:
        st.error(f"❌ Error mostrando gráfico: {str(e)}")

def mostrar_info_dataset(info):
    """Muestra información del dataset filtrado"""
    st.markdown("---")
    st.subheader("📋 Información del Dataset")
    
    with st.expander("Ver detalles del dataset filtrado"):
        st.json(info)

if __name__ == "__main__":

    main()
