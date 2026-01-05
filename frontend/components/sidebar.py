import streamlit as st
import requests

# URL del backend - CORREGIR PUERTO
BACKEND_URL = "http://localhost:8002/api/v1"

def render_sidebar():
    """Renderiza solo la parte de filtros del sidebar"""
    
    st.header("🔍 Filtros de Datos")
    
    # Obtener opciones de la API
    try:
        response = requests.get(f"{BACKEND_URL}/filters")
        if response.status_code == 200:
            filter_options = response.json()
        else:
            st.error(f"Error del backend: {response.status_code}")
            return {}
    except Exception as e:
        st.error(f"No se pudo conectar con el backend: {str(e)}")
        st.info("💡 Asegúrate de que el backend esté ejecutándose en puerto 8002")
        return {}
    
    filters = {}
    
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
    
    return {
        'filters': filters,
        'grupo_principal': grupo_principal,
        'ejecutar_analisis': ejecutar_analisis
    }