import streamlit as st
import requests
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
from datetime import datetime, timedelta

# Al inicio del archivo, después de los imports
if 'pronostico_results' not in st.session_state:
    st.session_state.pronostico_results = None
if 'pronostico_config' not in st.session_state:
    st.session_state.pronostico_config = None

# Configuración de página
st.set_page_config(
    page_title="NouxTrends - Pronósticos",
    page_icon="🔮",
    layout="wide",
)

# URL del backend
BACKEND_URL = "http://localhost:8002/api/v1"

def render_forecasting_sidebar():
    """Sidebar para configuración de pronósticos"""
    with st.sidebar:
        st.header("🔮 Pronósticos")
        st.markdown("---")
        
        # Info del sistema
        try:
            response = requests.get(f"{BACKEND_URL}/health", timeout=5)
            if response.status_code == 200:
                st.success("✅ Sistema Conectado")
            else:
                st.error("❌ Sistema Offline")
        except:
            st.error("❌ Sistema Offline")
        
        st.markdown("---")
        
        # Pestañas para diferentes modos - ESTA PARTE FALTABA
        modo = st.radio(
            "Modo de Pronóstico:",
            options=["🚀 Entrenar Modelos", "📊 Generar Pronósticos"],
            help="Primero entrena los modelos, luego genera pronósticos"
        )
        
        # VERIFICAR que estamos llamando a la función correcta
        if modo == "🚀 Entrenar Modelos":
            config = render_training_sidebar()
        else:
            config = render_prediction_sidebar()
        
        return config

def render_training_sidebar():
    """Sidebar para entrenamiento de modelos"""
    st.header("🤖 Entrenar Modelos")
    
    # Obtener variables categóricas reales del dataset
    variables_opciones = obtener_variables_categoricas()
    
    if not variables_opciones:
        st.error("❌ No se pudieron cargar las variables categóricas")
        return {
            'modo': 'entrenamiento',
            'variables': [],
            'modelos': [],
            'train_size': 0.8,
            'entrenar_modelos': False
        }
    
    variables_seleccionadas = st.multiselect(
        "Variables categóricas para entrenar:",
        options=variables_opciones,
        default=variables_opciones[:2],
        help="Cada variable categórica será usada en el modelo global"
    )
    
    # Información sobre el modelo
    st.info("""
    **🤖 Modelo LightGBM Global**
    - Un solo modelo que aprende de todas las categorías
    - Más rápido y eficiente que modelos individuales
    - Maneja automáticamente variables categóricas
    """)
    
    # Porcentaje de entrenamiento
    train_size = st.slider(
        "% Datos para entrenamiento:",
        min_value=60,
        max_value=90,
        value=80,
        help="Porcentaje de datos históricos para entrenar vs validar"
    )
    
    # Botón de entrenamiento
    entrenar_modelos = st.button(
        "🎯 Entrenar Modelo LightGBM", 
        type="primary", 
        use_container_width=True,
        help="Entrena un modelo global con LightGBM"
    )
    
    return {
        'modo': 'entrenamiento',
        'variables': variables_seleccionadas,
        'modelos': ["LightGBM"],  # Solo LightGBM ahora
        'train_size': train_size / 100,
        'entrenar_modelos': entrenar_modelos
    }


def obtener_variables_categoricas():
    """Obtiene las variables categóricas reales del dataset"""
    try:
        # Usar el endpoint de filtros para obtener las variables disponibles
        response = requests.get(f"{BACKEND_URL}/filters")
        if response.status_code == 200:
            filter_options = response.json()
            # Las keys del JSON son las variables categóricas disponibles
            variables = list(filter_options.keys())
            return variables
        else:
            return ['local', 'categoria', 'articulo']  # Fallback
    except:
        return ['local', 'categoria', 'articulo']  # Fallback
    

def render_prediction_sidebar():
    """Sidebar para generación de pronósticos"""
    st.header("📊 Generar Pronósticos")
    
    # Cargar modelos disponibles
    modelos_disponibles = cargar_modelos_disponibles()

    # ✅ MEJORAR LA VALIDACIÓN
    modelos_lista = modelos_disponibles.get('modelos_disponibles', [])
    modelos_entrenados = modelos_disponibles.get('modelos_entrenados', False)
    
    if not modelos_disponibles.get('modelos_disponibles'):
        st.warning("⚠️ Primero entrena los modelos en la pestaña de entrenamiento")
        return {
            'modo': 'prediccion',
            'modelos_disponibles': False
        }
    
    # ✅ OBTENER VARIABLES USADAS EN EL MODELO ENTRENADO
    variables_entrenadas = obtener_variables_entrenadas(modelos_disponibles)
    
    # ✅ FALLBACK: Si no encuentra variables, usar las básicas
    if not variables_entrenadas:
        st.warning("⚠️ No se pudieron cargar las variables del modelo. Usando variables por defecto.")
        variables_entrenadas = ['local', 'categoria']  # Fallback
        # También intentar obtener variables del endpoint de filtros
        try:
            variables_filtros = obtener_variables_categoricas()
            if variables_filtros:
                variables_entrenadas = variables_filtros[:2]  # Primeras 2 variables
        except:
            pass
    
    # ✅ MOSTRAR INFORMACIÓN SOBRE LAS VARIABLES ENTRENADAS
    st.info(f"**Variables usadas en el modelo:** {', '.join(variables_entrenadas)}")
    
    # ✅ NO PERMITIR SELECCION - USAR LAS MISMAS VARIABLES DEL ENTRENAMIENTO
    variables_seleccionadas = variables_entrenadas
    
    st.write(f"**Variables para pronóstico:** {', '.join(variables_seleccionadas)}")
    st.caption("ℹ️ Usando las mismas variables con las que fue entrenado el modelo")
    
    # Modelo a utilizar
    if 'lightgbm_global' in modelos_disponibles.get('modelos_disponibles', []):
        modelo_seleccionado = "lightgbm_global"
        st.info("🔍 Usando modelo LightGBM global")
    else:
        modelos_lista = modelos_disponibles.get('modelos_disponibles', [])
        if modelos_lista:
            modelo_seleccionado = modelos_lista[0]
            st.info(f"🔍 Usando modelo: {modelo_seleccionado}")
        else:
            modelo_seleccionado = None
            st.error("❌ No hay modelos disponibles")
    
    # Resto de la configuración...
    horizonte = st.slider(
        "Días a predecir:",
        min_value=7,
        max_value=90,
        value=30,
        help="Horizonte de predicción en días"
    )
    
    intervalo_confianza = st.slider(
        "Intervalo de confianza (%):",
        min_value=80,
        max_value=95,
        value=90,
        help="Nivel de confianza para los intervalos de predicción"
    )
    
    generar_pronostico = st.button(
        "🔮 Generar Pronóstico", 
        type="primary", 
        use_container_width=True
    )
    
    return {
        'modo': 'prediccion',
        'variables': variables_seleccionadas,  # ✅ Usar variables del entrenamiento
        'modelo': modelo_seleccionado,
        'horizonte': horizonte,
        'intervalo_confianza': intervalo_confianza / 100,
        'generar_pronostico': generar_pronostico,
        'modelos_disponibles': bool(modelos_disponibles.get('modelos_disponibles')),
        'info_modelos': modelos_disponibles
    }

def obtener_variables_entrenadas(modelos_disponibles):
    """Obtiene las variables usadas en el modelo entrenado"""
    try:
        print("🔍 Buscando variables entrenadas...")
        print(f"   modelos_info keys: {list(modelos_disponibles.get('modelos_info', {}).keys())}")
        
        # Buscar en todos los modelos disponibles
        for modelo_key, modelo_info in modelos_disponibles.get('modelos_info', {}).items():
            print(f"   Revisando modelo: {modelo_key}")
            print(f"   Info del modelo: {modelo_info}")
            
            variables = modelo_info.get('variables_usadas', [])
            if variables:
                print(f"   ✅ Variables encontradas: {variables}")
                return variables
            else:
                print(f"   ❌ No hay variables_usadas en {modelo_key}")
        
        print("   ❌ No se encontraron variables en ningún modelo")
        return []
        
    except Exception as e:
        print(f"❌ Error en obtener_variables_entrenadas: {str(e)}")
        return []


def cargar_modelos_disponibles():
    """Carga la lista de modelos entrenados disponibles"""
    try:
        response = requests.get(f"{BACKEND_URL}/forecast/models")
        if response.status_code == 200:
            data = response.json()
            
            # ✅ DEBUG: Ver qué está retornando el backend
            print("🔍 Respuesta del backend /forecast/models:")
            print(f"   modelos_disponibles: {data.get('modelos_disponibles', [])}")
            print(f"   modelos_info: {data.get('modelos_info', {})}")
            print(f"   total_modelos: {data.get('total_modelos', 0)}")
            print(f"   modelos_entrenados: {data.get('modelos_entrenados', False)}")
            
            return {
                'modelos_disponibles': data.get('modelos_disponibles', []),
                'modelos_info': data.get('modelos_info', {}),
                'total_modelos': data.get('total_modelos', 0),
                'modelos_entrenados': data.get('modelos_entrenados', False)
            }
        else:
            print(f"❌ Error en respuesta: {response.status_code}")
            return {}
    except Exception as e:
        print(f"❌ Error cargando modelos: {str(e)}")
        return {}

def main():
    # Sidebar de configuración
    config = render_forecasting_sidebar()
    
    st.title("🔮 Pronósticos de Ventas")
    st.markdown("Sistema de forecasting multi-nivel con modelos avanzados")
    
    if config['modo'] == 'entrenamiento':
        render_training_section(config)
    else:
        render_prediction_section(config)

def render_training_section(config):
    """Sección de entrenamiento de modelos"""
    st.header("🚀 Entrenamiento de Modelos")
    
    # Mostrar configuración seleccionada
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Variables", len(config['variables']))
    with col2:
        st.metric("Modelos", len(config['modelos']))
    with col3:
        st.metric("Train Size", f"{config['train_size']*100:.0f}%")
    
    # Ejecutar entrenamiento si se presionó el botón
    if config.get('entrenar_modelos', False):
        entrenar_modelos(config)
    else:
        st.info("💡 Configura los parámetros y haz clic en 'Entrenar Modelos' para comenzar")
        
        # Mostrar modelos existentes si hay
        modelos_existentes = cargar_modelos_disponibles()
        if modelos_existentes.get('total_modelos', 0) > 0:
            st.subheader("📋 Modelos Actualmente Entrenados")
            mostrar_modelos_existentes(modelos_existentes)

def entrenar_modelos(config):
    """Ejecuta el entrenamiento de modelos"""
    with st.spinner("🤖 Entrenando modelos... Esto puede tomar varios minutos"):
        try:
            payload = {
                "variables_categoricas": config['variables'],
                "modelos_a_entrenar": config['modelos'],
                "porcentaje_entrenamiento": config['train_size']
            }
            
            response = requests.post(f"{BACKEND_URL}/forecast/train", json=payload)
            
            if response.status_code == 200:
                resultados = response.json()
                mostrar_resultados_entrenamiento(resultados)
            else:
                st.error(f"❌ Error en el entrenamiento: {response.status_code}")
                st.json(response.json())
                
        except Exception as e:
            st.error(f"❌ Error conectando con el backend: {str(e)}")

def mostrar_resultados_entrenamiento(resultados):
    """Muestra los resultados del entrenamiento"""

    
    st.success("✅ Modelos entrenados exitosamente!")
    st.markdown("---")

     # Buscar datos de gráfico con diferentes nombres posibles
    datos_grafico = None
    posibles_nombres = ['datos_grafico', 'grafico_datos', 'training_graph', 'validation_data']
    
    
    
    if datos_grafico:
        st.subheader("📊 Gráfico de Validación del Modelo")
        mostrar_grafico_validacion(datos_grafico)
        st.markdown("---")
    else:
        st.warning("⚠️ No se encontraron datos para el gráfico de validación")

    

    # ✅ NUEVO: Mostrar gráfico de validación si está disponible
    if 'datos_grafico' in resultados and resultados['datos_grafico']:
        st.subheader("📊 Gráfico de Validación del Modelo")
        mostrar_grafico_validacion(resultados['datos_grafico'])
        st.markdown("---")
    
    # Modelos entrenados
    st.subheader("📊 Modelos Entrenados")
    modelos_entrenados = resultados.get('modelos_entrenados', [])
    
    if modelos_entrenados:
        df_modelos = pd.DataFrame([
            {'Modelo': modelo, 'Estado': '✅ Entrenado'} 
            for modelo in modelos_entrenados
        ])
        st.dataframe(df_modelos, use_container_width=True)
    else:
        st.warning("⚠️ No se entrenaron modelos nuevos")
    
    # Métricas de calidad
    st.subheader("📈 Métricas de Calidad")
    metricas = resultados.get('metricas_entrenamiento', {})
    
    if metricas:
        datos_metricas = []
        for modelo, metrics in metricas.items():
            datos_metricas.append({
                'Modelo': modelo,
                'MAPE': f"{metrics.get('mape', 0):.2f}%",
                'RMSE': f"${metrics.get('rmse', 0):,.0f}",
                'R²': f"{metrics.get('r2', 0):.3f}"
            })
        
        df_metricas = pd.DataFrame(datos_metricas)
        st.dataframe(df_metricas, use_container_width=True)
        
        # Interpretación de métricas
        st.info("""
        **Interpretación de métricas:**
        - **MAPE < 10%**: Excelente | **10-20%**: Bueno | **>20%**: Necesita mejora
        - **RMSE**: Error absoluto en unidades monetarias (menor es mejor)
        - **R²**: Proporción de varianza explicada (más cercano a 1 es mejor)
        """)

def render_prediction_section(config):
    """Sección de generación de pronósticos"""
    st.header("📊 Generación de Pronósticos")
    
    if not config.get('modelos_disponibles', False):
        st.error("🚫 No hay modelos entrenados disponibles. Primero entrena modelos en la pestaña de entrenamiento.")
        return
    
    # Mostrar configuración seleccionada
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Modelo", config['modelo'])
    with col2:
        st.metric("Horizonte", f"{config['horizonte']} días")
    with col3:
        st.metric("Variables", len(config['variables']))
    with col4:
        st.metric("Confianza", f"{config['intervalo_confianza']*100:.0f}%")
    
    # Mostrar métricas del modelo seleccionado
    mostrar_metricas_modelo(config['modelo'], config.get('info_modelos', {}))
    
    # Botón para generar pronóstico
    if config.get('generar_pronostico', False):
        generar_pronostico(config)
    elif st.session_state.pronostico_results:
        # ✅ SI HAY RESULTADOS EN SESSION_STATE, MOSTRARLOS
        st.info("📊 Mostrando pronóstico previamente generado. Usa los filtros para explorar los datos.")
        mostrar_resultados_pronostico()  # Sin parámetros = usar session_state
    else:
        st.info("💡 Configura los parámetros y haz clic en 'Generar Pronóstico'")


def mostrar_metricas_modelo(modelo_seleccionado, info_modelos):
    """Muestra las métricas del modelo seleccionado"""
    modelo_key = f"total_{modelo_seleccionado}" if not modelo_seleccionado.startswith('lightgbm') else modelo_seleccionado
    metricas = info_modelos.get('metricas', {}).get(modelo_key, {})
    
    if metricas:
        st.subheader("🎯 Métricas del Modelo Seleccionado")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            mape = metricas.get('mape', 0)
            if mape < 15:
                delta_color = "normal"
                delta_text = "Bueno"
            else:
                delta_color = "inverse"
                delta_text = "Regular"
            st.metric("MAPE", f"{mape:.1f}%", delta=delta_text, delta_color=delta_color)
        
        with col2:
            rmse = metricas.get('rmse', 0)
            st.metric("RMSE", f"${rmse:,.0f}")
        
        with col3:
            r2 = metricas.get('r2', 0)
            if r2 > 0.7:
                delta_color = "normal"
                delta_text = "Excelente"
            else:
                delta_color = "normal"
                delta_text = "Bueno"
            st.metric("R²", f"{r2:.3f}", delta=delta_text, delta_color=delta_color)


def mostrar_grafico_validacion(datos_grafico):
    """Muestra gráfico comparativo de entrenamiento vs validación vs predicciones"""
    try:
        if not datos_grafico:
            st.warning("No hay datos de gráfico disponibles")
            return
            
        # Convertir fechas correctamente
        def convertir_fechas(fechas):
            if not fechas:
                return []
            if isinstance(fechas[0], str):
                return pd.to_datetime(fechas)
            else:
                return pd.to_datetime(fechas, unit='ms')
        
        # Crear DataFrames individuales
        fechas_entrenamiento = convertir_fechas(datos_grafico['fechas_entrenamiento'])
        fechas_validacion = convertir_fechas(datos_grafico['fechas_validacion'])
        
        # ✅ CORREGIR: Agregar datos por fecha (sumar todas las combinaciones por día)
        df_entrenamiento = pd.DataFrame({
            'fecha': fechas_entrenamiento,
            'ventas': datos_grafico['ventas_entrenamiento']
        })
        df_validacion_real = pd.DataFrame({
            'fecha': fechas_validacion, 
            'ventas': datos_grafico['ventas_validacion']
        })
        df_validacion_pred = pd.DataFrame({
            'fecha': fechas_validacion,
            'ventas': datos_grafico['predicciones_validacion']
        })
        # ✅ Filtrar para mostrar solo los últimos 60 días de entrenamiento
        dias_mostrar_entrenamiento = 60
        # Agrupar por fecha (sumar todas las combinaciones)
        df_entrenamiento_agregado = df_entrenamiento.groupby('fecha')['ventas'].sum().reset_index()
        df_validacion_real_agregado = df_validacion_real.groupby('fecha')['ventas'].sum().reset_index()
        df_validacion_pred_agregado = df_validacion_pred.groupby('fecha')['ventas'].sum().reset_index()

        if len(df_entrenamiento_agregado) > dias_mostrar_entrenamiento:
            df_entrenamiento_filtrado = df_entrenamiento_agregado.tail(dias_mostrar_entrenamiento)
        else:
            df_entrenamiento_filtrado = df_entrenamiento_agregado

        
        # ✅ NUEVO: Filtrar para mostrar solo los últimos 60 días de entrenamiento + toda la validación
        dias_mostrar_entrenamiento = 60  # Mostrar solo los últimos 60 días de entrenamiento
        
        
        # ✅ CREAR GRÁFICO CON SEABORN/MATPLOTLIB
        import matplotlib.pyplot as plt
        import seaborn as sns

        # Crear gráfico interactivo
        sns.set_style("whitegrid")
        plt.figure(figsize=(12, 6))
        
        # Datos de entrenamiento (azul) - SOLO últimos días
        plt.plot(df_entrenamiento_filtrado['fecha'], 
                df_entrenamiento_filtrado['ventas'], 
                label=f'Entrenamiento (últimos {len(df_entrenamiento_filtrado)} días)',
                color='#1f77b4', linewidth=2, alpha=0.8)
        
        # Datos reales de validación (naranja)
        plt.plot(df_validacion_real_agregado['fecha'], 
                df_validacion_real_agregado['ventas'], 
                label=f'Validación - Real ({len(df_validacion_real_agregado)} días)',
                color='#ff7f0e', linewidth=2, alpha=0.8)
        
        # Predicciones sobre validación (verde)
        plt.plot(df_validacion_pred_agregado['fecha'], 
                df_validacion_pred_agregado['ventas'], 
                label='Validación - Predicción',
                color='#2ca02c', linewidth=2, linestyle='--', alpha=0.9)
        
        # Línea vertical para separar entrenamiento/validación
        if 'fecha_corte' in datos_grafico:
            try:
                fecha_corte = convertir_fechas([datos_grafico['fecha_corte']])[0]
                plt.axvline(x=fecha_corte, color='red', linestyle=':', linewidth=2, 
                           label='Inicio Validación')
            except:
                pass
        
        # Configurar el gráfico
        plt.title('📊 Validación del Modelo: Entrenamiento vs Predicciones', fontsize=14, fontweight='bold')
        plt.xlabel('Fecha', fontsize=12)
        plt.ylabel('Ventas Totales Diarias', fontsize=12)
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.xticks(rotation=45)
        plt.tight_layout()
        
        # Mostrar el gráfico en Streamlit
        st.pyplot(plt)
        # Mostrar estadísticas
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Días Entrenamiento (mostrados)", len(df_entrenamiento_filtrado))
        with col2:
            st.metric("Días Validación", len(df_validacion_real_agregado))
        with col3:
            if 'fecha_corte' in datos_grafico:
                try:
                    fecha_corte = convertir_fechas([datos_grafico['fecha_corte']])[0]
                    st.metric("Fecha Corte", fecha_corte.strftime('%Y-%m-%d'))
                except:
                    st.metric("Fecha Corte", "N/A")
        # Limpiar la figura para evitar superposición en siguientes renders
        plt.clf()   

    except Exception as e:
        st.error(f"❌ Error mostrando gráfico de validación: {str(e)}")
        import traceback
        st.write("🔍 DEBUG - Traceback completo:")
        st.code(traceback.format_exc())


def generar_pronostico(config):
    """Genera y muestra el pronóstico"""
    with st.spinner("🔮 Generando pronóstico..."):
        try:
            payload = {
                "horizonte": config['horizonte'],
                "modelo": config['modelo'],
                "variables_categoricas": config['variables'],
                "intervalo_confianza": config['intervalo_confianza']
            }
            
            response = requests.post(f"{BACKEND_URL}/forecast/multi-level", json=payload)
            
            if response.status_code == 200:
                resultados = response.json()
                # ✅ GUARDAR EN SESSION_STATE
                st.session_state.pronostico_results = resultados
                st.session_state.pronostico_config = config
                mostrar_resultados_pronostico(resultados, config)
            else:
                st.error(f"❌ Error en el pronóstico: {response.status_code}")
                
        except Exception as e:
            st.error(f"❌ Error conectando con el backend: {str(e)}")

def mostrar_resultados_pronostico(resultados=None, config=None):
    """Muestra los resultados del pronóstico desde session_state"""
    # ✅ USAR SESSION_STATE SI NO SE PASA PARÁMETROS
    if resultados is None:
        resultados = st.session_state.pronostico_results
    if config is None:
        config = st.session_state.pronostico_config
    
    if not resultados or not config:
        st.error("No hay resultados de pronóstico disponibles")
        return
        
    st.success("✅ Pronóstico generado exitosamente!")
    st.markdown("---")
    
    data = resultados.get('data', {})
    
    # Métricas de calidad
    if 'metricas_calidad' in data:
        mostrar_metricas_pronostico(data['metricas_calidad'])
    
    # Pronósticos
    if 'forecast_data' in data:
        forecast_data = data['forecast_data']
        
        # ✅ CORREGIDO: Pasar horizonte como parámetro
        df_filtrado, filtros_aplicados = mostrar_filtros_pronostico(
            forecast_data, 
            config['variables'], 
            config['horizonte']
        )
        
        # Determinar si hay filtros activos
        filtros_activos = any(valores for valores in filtros_aplicados.values())
        
        # Mostrar gráfico según filtros - ✅ SIEMPRE usar datos filtrados
        if filtros_activos:
            mostrar_grafico_desglosado(df_filtrado, filtros_aplicados, config)
        else:
            # ✅ CORREGIDO: Mostrar gráfico con datos filtrados (aunque no haya filtros activos)
            mostrar_grafico_desglosado(df_filtrado, filtros_aplicados, config)
        
        # Tabla de valores filtrados
        st.subheader("📋 Valores Pronosticados")
        mostrar_tabla_pronostico(df_filtrado, config)  # ✅ Pasar df_filtrado, no forecast_data
    
    # Información del modelo
    st.subheader("🤖 Información del Modelo")
    st.write(f"**Modelo utilizado:** {data.get('modelo_utilizado', 'N/A')}")
    st.write(f"**Variables desagregadas:** {', '.join(data.get('variables_desagregadas', []))}")


def mostrar_grafico_desglosado(df_filtrado, filtros, config):
    """Muestra gráfico para las combinaciones filtradas"""
    try:
        # Agrupar por fecha para las combinaciones seleccionadas
        df_filtrado['fecha'] = pd.to_datetime(df_filtrado['fecha'])
        df_agrupado = df_filtrado.groupby('fecha').agg({
            'pronostico': 'sum',
            'intervalo_inferior': 'sum',
            'intervalo_superior': 'sum'
        }).reset_index().sort_values('fecha')
        
        # Determinar título según filtros
        titulo = "Pronóstico para: "
        condiciones = []
        for variable, valores in filtros.items():
            if valores:
                condiciones.append(f"{variable}={', '.join(valores)}")
        
        if condiciones:
            titulo += " | ".join(condiciones)
        else:
            titulo = "Pronóstico Total"
        
        # Crear gráfico
        fig = go.Figure()
        
        # Línea del pronóstico
        fig.add_trace(go.Scatter(
            x=df_agrupado['fecha'],
            y=df_agrupado['pronostico'],
            mode='lines',
            name='Pronóstico',
            line=dict(color='#1f77b4', width=3)
        ))
        
        # Intervalo de confianza
        fig.add_trace(go.Scatter(
            x=df_agrupado['fecha'].tolist() + df_agrupado['fecha'].tolist()[::-1],
            y=df_agrupado['intervalo_superior'].tolist() + df_agrupado['intervalo_inferior'].tolist()[::-1],
            fill='toself',
            fillcolor='rgba(31, 119, 180, 0.2)',
            line=dict(color='rgba(255,255,255,0)'),
            name=f'Intervalo {config["intervalo_confianza"]*100:.0f}% Confianza'
        ))
        
        fig.update_layout(
            title=f"{titulo} - {config['horizonte']} días",
            xaxis_title="Fecha",
            yaxis_title="Ventas ($)",
            hovermode='x unified',
            height=500
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Mostrar estadísticas del filtro
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Días pronosticados", len(df_agrupado))
        with col2:
            promedio = df_agrupado['pronostico'].mean()
            st.metric("Promedio diario", f"${promedio:,.0f}")
            
    except Exception as e:
        st.error(f"❌ Error mostrando gráfico desglosado: {str(e)}")


def mostrar_metricas_pronostico(metricas):
    """Muestra métricas de calidad del pronóstico"""
    st.subheader("📊 Métricas de Calidad")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        mape = metricas.get('mape', 0)
        # ✅ CORREGIR: Streamlit solo acepta 'normal', 'inverse', 'off'
        if mape < 15:
            delta_color = "normal"  # Verde
            delta_text = "Bueno"
        elif mape < 25:
            delta_color = "normal"  # Naranja (pero Streamlit no tiene naranja)
            delta_text = "Regular" 
        else:
            delta_color = "inverse"  # Rojo
            delta_text = "Necesita mejora"
            
        st.metric("MAPE", f"{mape:.1f}%", delta=delta_text, delta_color=delta_color)
    
    with col2:
        rmse = metricas.get('rmse', 0)
        st.metric("RMSE", f"${rmse:,.0f}")
    
    with col3:
        r2 = metricas.get('r2', 0)
        # ✅ Color para R²
        if r2 > 0.7:
            delta_color = "normal"
            delta_text = "Excelente"
        elif r2 > 0.5:
            delta_color = "normal" 
            delta_text = "Bueno"
        else:
            delta_color = "inverse"
            delta_text = "Bajo"
            
        st.metric("R²", f"{r2:.3f}", delta=delta_text, delta_color=delta_color)

def mostrar_grafico_pronostico(forecast_data, config):
    """Muestra gráfico interactivo del pronóstico TOTAL"""
    try:
        # ✅ DEBUG DETALLADO: Analizar estructura de datos
        st.write("🔍 DEBUG COMPLETO - Estructura de forecast_data:")
        # ✅ USAR forecast_data directamente y SUMAR
        df_pronostico = pd.DataFrame(forecast_data)
        df_pronostico['fecha'] = pd.to_datetime(df_pronostico['fecha'])
        st.write(f"Total registros: {len(df_pronostico)}")
        st.write(f"Horizonte configurado: {config['horizonte']} días")
        st.write(f"Combinaciones por día: {len(df_pronostico) // config['horizonte']}")

        # Ver distribución por variables
        for variable in config['variables']:
            if variable in df_pronostico.columns:
                st.write(f"Valores únicos en {variable}: {df_pronostico[variable].unique()}")
        
        # Ver estadísticas de pronósticos individuales
        st.write("📊 Estadísticas de pronósticos individuales:")
        st.write(df_pronostico['pronostico'].describe())
        
        # Ver primeros registros
        st.write("📋 Primeros 5 registros:")
        st.write(df_pronostico.head())


        # Agrupar por fecha para obtener totales diarios
        df_agrupado = df_pronostico.groupby('fecha').agg({
            'pronostico': 'sum',
            'intervalo_inferior': 'sum', 
            'intervalo_superior': 'sum'
        }).reset_index()
        
        df_agrupado = df_agrupado.sort_values('fecha')
        
        # ✅ DEBUG: Verificar totales
        # ✅ DEBUG: Verificar totales después de agrupar
        st.write("🔍 DEBUG - Después de agrupar por fecha:")
        st.write(f"Días únicos: {len(df_agrupado)}")
        st.write(f"Rango pronósticos totales: ${df_agrupado['pronostico'].min():.0f} - ${df_agrupado['pronostico'].max():.0f}")
        st.write(f"Promedio diario total: ${df_agrupado['pronostico'].mean():.0f}")
        st.write(f"🔍 DEBUG - Combinaciones sumadas: {len(df_pronostico) // len(df_agrupado)}")
        
        # Ver distribución de totales diarios
        st.write("📊 Distribución de totales diarios:")
        st.write(df_agrupado['pronostico'].describe())

        # Crear gráfico con los totales CORREGIDOS
        fig = go.Figure()
        
        # Línea del pronóstico TOTAL
        fig.add_trace(go.Scatter(
            x=df_agrupado['fecha'],
            y=df_agrupado['pronostico'],
            mode='lines',
            name='Pronóstico Total',
            line=dict(color='#1f77b4', width=3)
        ))
        
        # Intervalo de confianza TOTAL
        fig.add_trace(go.Scatter(
            x=df_agrupado['fecha'].tolist() + df_agrupado['fecha'].tolist()[::-1],
            y=df_agrupado['intervalo_superior'].tolist() + df_agrupado['intervalo_inferior'].tolist()[::-1],
            fill='toself',
            fillcolor='rgba(31, 119, 180, 0.2)',
            line=dict(color='rgba(255,255,255,0)'),
            name=f'Intervalo {config["intervalo_confianza"]*100:.0f}% Confianza'
        ))
        
        fig.update_layout(
            title=f"Pronóstico de Ventas Totales - {config['horizonte']} días",
            xaxis_title="Fecha",
            yaxis_title="Ventas Totales ($)",
            hovermode='x unified',
            height=500
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
    except Exception as e:
        st.error(f"❌ Error generando gráfico: {str(e)}")

def mostrar_filtros_pronostico(forecast_data, variables, horizonte):
    """Muestra filtros múltiples para todas las variables categóricas"""
    st.subheader("🔍 Filtros de Desglose")
    
    df = pd.DataFrame(forecast_data)
    
    filtros = {}
    
    # MOSTRAR FILTROS MÚLTIPLES PARA TODAS LAS VARIABLES
    for variable in variables:
        if variable in df.columns:
            opciones = sorted(df[variable].unique().tolist())
            seleccionados = st.multiselect(
                f"Filtrar por {variable}:",
                options=opciones,
                default=opciones,  # Por defecto seleccionar todos
                help=f"Selecciona uno o más {variable}"
            )
            filtros[variable] = seleccionados
    
    # Aplicar filtros
    df_filtrado = df.copy()
    
    for variable, valores in filtros.items():
        if valores:  # Si se seleccionaron valores
            df_filtrado = df_filtrado[df_filtrado[variable].isin(valores)]
    
    # ✅ CORREGIDO: Usar horizonte como parámetro
    st.write(f"**Combinaciones mostradas:** {len(df_filtrado) // horizonte} de {len(df) // horizonte}")
    
    return df_filtrado, filtros


def mostrar_tabla_pronostico(forecast_data, config):
    """Muestra tabla con valores pronosticados FILTRADOS"""
    try:
        df_pronostico = pd.DataFrame(forecast_data)
        df_pronostico['fecha'] = pd.to_datetime(df_pronostico['fecha'])
        
        # ✅ AGREGAR POR FECHA PARA MOSTRAR TOTALES CORRECTOS
        df_agrupado = df_pronostico.groupby('fecha').agg({
            'pronostico': 'sum',
            'intervalo_inferior': 'sum',
            'intervalo_superior': 'sum'
        }).reset_index()
        
        df_agrupado = df_agrupado.round(2)
        df_agrupado['fecha'] = df_agrupado['fecha'].dt.strftime('%Y-%m-%d')
        
        st.dataframe(df_agrupado, use_container_width=True)
        
        # Opción de descarga
        csv = df_agrupado.to_csv(index=False)
        st.download_button(
            label="📥 Descargar Pronóstico (CSV)",
            data=csv,
            file_name=f"pronostico_ventas_{datetime.now().strftime('%Y%m%d')}.csv",
            mime="text/csv"
        )
        
    except Exception as e:
        st.error(f"❌ Error mostrando tabla: {str(e)}")

def mostrar_modelos_existentes(modelos_existentes):
    """Muestra información de modelos ya entrenados"""
    if modelos_existentes.get('total_modelos', 0) > 0:
        st.write(f"**Total de modelos entrenados:** {modelos_existentes['total_modelos']}")
        
        # Mostrar algunos modelos principales
        modelos_principales = [m for m in modelos_existentes['modelos_disponibles'] if m.startswith('total_')]
        if modelos_principales:
            st.write("**Modelos principales disponibles:**")
            for modelo in modelos_principales[:5]:  # Mostrar solo 5
                st.write(f"• {modelo}")

if __name__ == "__main__":
    main()