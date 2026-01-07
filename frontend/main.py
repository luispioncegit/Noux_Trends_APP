import streamlit as st
import requests
import os
import time


# --- CONFIGURACIÓN DE URL ---
# Usamos BACKEND_URL que ya tiene el /api/v1
BACKEND_URL = os.environ.get("BACKEND_URL", "http://localhost:8002/api/v1").rstrip('/')
# Creamos una ruta específica para el health check
HEALTH_URL = f"{BACKEND_URL}/health"

# Configuración de página
st.set_page_config(
    page_title="NouxTrends - Inicio",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="expanded"
)

def load_css():
    try:
        with open('frontend/assets/styles.css') as f:
            st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)
    except:
        st.markdown("""
        <style>
        .main-header {
            font-size: 2.5rem;
            color: #1f77b4;
            margin-bottom: 1rem;
        }
        .feature-card {
            padding: 1.5rem;
            border-radius: 10px;
            background-color: #f0f2f6;
            margin: 0.5rem 0;
            border-left: 4px solid #1f77b4;
        }
        </style>
        """, unsafe_allow_html=True)

def get_health_with_retry(url, retries=3, delay=5):
    """
    Intenta conectar al backend varias veces antes de rendirse.
    Útil para el 'despertar' de Render.
    """
    for i in range(retries):
        try:
            response = requests.get(url, timeout=300)
            if response.status_code == 200:
                return response # Conexión exitosa
        except Exception:
            if i < retries - 1:
                time.sleep(delay) # Espera antes de intentar de nuevo
    return None # Falló tras los reintentos

def render_main_sidebar():
    """Sidebar personalizado para Main SIN filtros"""
    with st.sidebar:
        st.header("🚀 NouxTrends")
        st.markdown("---")
        
        # Info del sistema
        try:
            response = get_health_with_retry(HEALTH_URL)
            if response.status_code == 200:
                st.success("✅ Sistema Conectado")
            else:
                st.error("❌ Sistema Offline")
        except:
            st.error("❌ Sistema Offline")
        
        st.markdown("---")
        st.markdown("### Navegación")
        st.info("Selecciona una página del menú superior")

def main():
    load_css()
    
    # Sidebar personalizado SIN filtros
    render_main_sidebar()
    
    # Página principal de presentación
    st.markdown('<div class="main-header">🚀 NouxTrends - Motor de Predicciones</div>', unsafe_allow_html=True)
    st.subheader("Sistema Inteligente de Análisis y Pronóstico de Ventas")

    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("""
        ### 📊 ¿Qué puedes hacer con NouxTrends?
        
        <div class="feature-card">
        <strong>Análisis Descriptivo</strong><br>
        Visualiza tendencias históricas y métricas clave de tus ventas
        </div>
        
        <div class="feature-card">
        <strong>Pronósticos Avanzados</strong><br>
        Predice ventas futuras con modelos de machine learning
        </div>
        
        <div class="feature-card">
        <strong>Análisis de Locales</strong><br>
        Compara desempeño entre diferentes establecimientos
        </div>
        
        <div class="feature-card">
        <strong>Detección de Patrones</strong><br>
        Identifica tendencias estacionales y comportamientos
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown("""
        ### 🛠️ Cómo Empezar
        
        1. **Ve a la página de Análisis** usando el menú lateral
        2. **Configura los filtros** según tus necesidades  
        3. **Ejecuta el análisis** y explora los resultados
        4. **Genera pronósticos** basados en los datos
        
        ### 📈 Estado del Sistema
        """)
        
        # Información del sistema
        try:
            response = get_health_with_retry(HEALTH_URL)
            if response.status_code == 200:
                health_data = response.json()
                st.success("✅ Backend conectado")
                
                stats = health_data.get('database_stats', {})
                st.metric("Total Registros", stats.get('total_registros', 'N/A'))
                st.metric("Locales Únicos", stats.get('locales_unicos', 'N/A'))
                st.metric("Artículos", stats.get('articulos_unicos', 'N/A'))
            else:
                st.error("❌ Problema con el backend")
        except:
            st.error("❌ No se puede conectar con el backend")

    st.success("¡Usa el menú lateral para navegar a las diferentes secciones!")



if __name__ == "__main__":

    main()


