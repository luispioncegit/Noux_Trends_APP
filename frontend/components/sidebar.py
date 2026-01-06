import streamlit as st
import requests
import os

# Configuración dinámica de la URL del backend
# En Render, configuraremos una Variable de Entorno llamada BACKEND_URL
BACKEND_URL = os.environ.get("BACKEND_URL", "http://localhost:8002/api/v1")

def render_sidebar():
    """Renderiza la sección de filtros en la barra lateral"""
    
    st.sidebar.header("🔍 Filtros de Datos")
    
    # Obtener opciones de la API
    try:
        # Nota: Usamos st.cache_data para no saturar al backend a cada segundo
        @st.cache_data(ttl=600) # Caché por 10 minutos
        def get_filter_options():
            response = requests.get(f"{BACKEND_URL}/filters")
            if response.status_code == 200:
                return response.json()
            return None

        filter_options = get_filter_options()
        
        if not filter_options:
            st.sidebar.error("Error: El backend no devolvió opciones.")
            return {}

    except Exception as e:
        st.sidebar.error(f"Error de conexión: {str(e)}")
        return {}
    
    filters = {}
    
    # Renderizamos los filtros dentro del sidebar usando 'with st.sidebar:'
    with st.sidebar:
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
            
        st.divider()
        
        # Grupo principal para análisis
        grupo_principal = st.selectbox(
            "Agrupar por",
            options=['local', 'articulo', 'categoria']
        )
        
        # Botón para ejecutar análisis
        ejecutar_analisis = st.button(
            "🚀 Ejecutar Análisis", 
            type="primary", 
            use_container_width=True
        )
    
    return {
        'filters': filters,
        'grupo_principal': grupo_principal,
        'ejecutar_analisis': ejecutar_analisis
    }
