import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from scipy import stats
import base64
from io import BytesIO
from typing import Dict, Any, Optional
from statsmodels.tsa.stattools import adfuller, acf, pacf
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
import warnings
from datetime import datetime
warnings.filterwarnings('ignore')

# Importar el módulo de festivos
from .festivos import EcuadorHolidays
#importar el forecaster
from .forecaster import Forecaster

def plot_to_base64(fig):
    """Convierte matplotlib figure a base64 para API"""
    buf = BytesIO()
    fig.savefig(buf, format='png', dpi=100, bbox_inches='tight')
    buf.seek(0)
    img_base64 = base64.b64encode(buf.getvalue()).decode('utf-8')
    plt.close(fig)
    return f"data:image/png;base64,{img_base64}"

class NouxTrendsAnalyzer:
    def __init__(self, df: pd.DataFrame):
        self.df = df.copy()
        self.df['fecha'] = pd.to_datetime(self.df['fecha'])
        # Agregar clasificación de días
        self.df = self._add_day_classification(self.df)
        self.forecaster = Forecaster(df)
        self.modelos_entrenados = False



    ############ Analisis de la serie de tiempo ############
        
    def _add_day_classification(self, df):
        """Agrega clasificación de tipos de día al DataFrame"""
        df_copy = df.copy()
        df_copy['tipo_dia'] = df_copy['fecha'].apply(EcuadorHolidays.get_day_type)
        df_copy['es_festivo'] = df_copy['fecha'].apply(EcuadorHolidays.is_holiday)
        df_copy['es_fin_semana'] = df_copy['fecha'].apply(EcuadorHolidays.is_weekend)
        return df_copy
    
    def get_filter_options(self) -> Dict[str, Any]:
        """Obtiene opciones disponibles para filtros"""
        return {
            "local": sorted(self.df['local'].unique()),
            "articulo": sorted(self.df['articulo'].unique()),
            "categoria": sorted(self.df['categoria'].unique()),
            "tamaño_local": sorted(self.df['tamaño_local'].unique()),
            "ubicacion_local": sorted(self.df['ubicacion_local'].unique())
        }
    
    # ✅ MÉTODO PRINCIPAL QUE FALTA
    def analyze_time_series(self, request) -> Dict[str, Any]:
        """Análisis completo de series temporales para pronósticos"""
        # Aplicar filtros
        df_filtrado = self._apply_filters(request)
        
        # Generar todos los gráficos de análisis
        charts = {
            "time_series": self._plot_serie_tiempo_completo(df_filtrado, request.columna_valor, request.grupo_principal),
            "seasonality": self._analisis_estacionalidad_completo(df_filtrado, request.columna_valor, request.grupo_principal),
            "stationarity": self._analisis_estacionariedad(df_filtrado, request.columna_valor),
            "correlation": self._analisis_correlaciones(df_filtrado, request.columna_valor, request.grupo_principal),
            "volatility": self._analisis_volatilidad(df_filtrado, request.columna_valor)
        }
        
        # Calcular métricas avanzadas
        metrics = self._calculate_advanced_metrics(df_filtrado, request.columna_valor)
        
        # Insights automáticos para pronósticos
        insights = self._generate_forecasting_insights(df_filtrado, request.columna_valor)
        
        return {
            "charts": charts,
            "metrics": metrics,
            "insights": insights,
            "filtered_data_info": {
                "rows": len(df_filtrado),
                "date_range": {
                    "start": df_filtrado['fecha'].min().isoformat(),
                    "end": df_filtrado['fecha'].max().isoformat()
                }
            }
        }
    
    # ✅ MÉTODOS AUXILIARES QUE FALTAN
    def _apply_filters(self, request) -> pd.DataFrame:
        """Aplica filtros al DataFrame"""
        df_filtrado = self.df.copy()
        
        if request.filtros:
            for key, value in request.filtros.items():
                if isinstance(value, list):
                    df_filtrado = df_filtrado[df_filtrado[key].isin(value)]
                else:
                    df_filtrado = df_filtrado[df_filtrado[key] == value]
        
        if request.fecha_inicio:
            df_filtrado = df_filtrado[df_filtrado['fecha'] >= request.fecha_inicio]
        if request.fecha_fin:
            df_filtrado = df_filtrado[df_filtrado['fecha'] <= request.fecha_fin]
            
        return df_filtrado
    
    def _plot_serie_tiempo_completo(self, df, columna_valor, grupo_principal):
        """Serie temporal con tendencia y componentes"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Serie temporal principal
        df_agrupado = df.groupby(['fecha', grupo_principal])[columna_valor].sum().reset_index()
        
        for grupo_val in df_agrupado[grupo_principal].unique():
            df_grupo = df_agrupado[df_agrupado[grupo_principal] == grupo_val]
            axes[0,0].plot(df_grupo['fecha'], df_grupo[columna_valor], label=grupo_val, linewidth=1.5, alpha=0.8)
        
        axes[0,0].set_title('Serie Temporal Completa')
        axes[0,0].legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        axes[0,0].grid(True, alpha=0.3)
        
        # Tendencia (media móvil 30 días)
        df_diario = df.groupby('fecha')[columna_valor].sum().reset_index()
        df_diario['media_movil_30'] = df_diario[columna_valor].rolling(window=30).mean()
        axes[0,1].plot(df_diario['fecha'], df_diario[columna_valor], alpha=0.3, label='Valor Diario')
        axes[0,1].plot(df_diario['fecha'], df_diario['media_movil_30'], linewidth=2, label='Tendencia (30 días)')
        axes[0,1].set_title('Tendencia con Media Móvil')
        axes[0,1].legend()
        axes[0,1].grid(True, alpha=0.3)
        
        # Distribución de los datos
        axes[1,0].hist(df[columna_valor], bins=50, alpha=0.7, edgecolor='black')
        axes[1,0].set_title('Distribución de Valores')
        axes[1,0].set_xlabel(columna_valor)
        axes[1,0].set_ylabel('Frecuencia')
        
        # Boxplot por grupo principal
        sns.boxplot(data=df, x=grupo_principal, y=columna_valor, ax=axes[1,1])
        axes[1,1].tick_params(axis='x', rotation=45)
        axes[1,1].set_title('Distribución por Grupo')
        
        plt.tight_layout()
        return plot_to_base64(fig)
    
    def _analisis_estacionariedad(self, df, columna_valor):
        """Análisis de estacionariedad para modelos de pronóstico"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Preparar datos diarios
        df_diario = df.groupby('fecha')[columna_valor].sum().reset_index()
        df_diario = df_diario.set_index('fecha').asfreq('D').fillna(0)
        serie_temporal = df_diario[columna_valor]
        
        serie_temporal_diff = serie_temporal.diff(periods=1).dropna()
        # Test de Dickey-Fuller aumentado
        try:
            result = adfuller(serie_temporal_diff.dropna())
            adf_statistic = result[0]
            adf_pvalue = result[1]
        except:
            adf_statistic = None
            adf_pvalue = None
        
        # Gráfico ACF (Autocorrelación)
        
        plot_acf(serie_temporal_diff, ax=axes[0,0], lags=40, alpha=0.05)
        axes[0,0].set_title(f'Autocorrelación (ACF)\nADF p-value: {adf_pvalue:.4f}')
        
        # Gráfico PACF (Autocorrelación Parcial)
        plot_pacf(serie_temporal_diff, ax=axes[0,1], lags=40, alpha=0.05)
        axes[0,1].set_title('Autocorrelación Parcial (PACF)')
        
        # Media y varianza móviles
        rolling_mean = serie_temporal.rolling(window=30).mean()
        rolling_std = serie_temporal.rolling(window=30).std()
        
        axes[1,0].plot(serie_temporal.index, serie_temporal, label='Serie Original', alpha=0.5)
        axes[1,0].plot(rolling_mean.index, rolling_mean, label='Media Móvil (30 días)', linewidth=2)
        axes[1,0].set_title('Análisis de Media Móvil')
        axes[1,0].legend()
        
        axes[1,1].plot(rolling_std.index, rolling_std, color='red', linewidth=2)
        axes[1,1].set_title('Desviación Estándar Móvil (30 días)')
        axes[1,1].set_ylabel('Desviación Estándar')
        
        plt.tight_layout()
        return plot_to_base64(fig)
    
    def _analisis_correlaciones(self, df, columna_valor, grupo_principal):
        """Análisis de correlaciones y relaciones"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Heatmap de correlación entre grupos
        try:
            ventas_pivot = df.pivot_table(
                index='fecha', 
                columns=grupo_principal, 
                values=columna_valor, 
                aggfunc='sum'
            ).fillna(0)
            
            if len(ventas_pivot.columns) > 1:
                corr_matrix = ventas_pivot.corr()
                sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0, ax=axes[0,0])
                axes[0,0].set_title('Correlación entre Grupos')
            else:
                axes[0,0].text(0.5, 0.5, 'Se necesitan múltiples\ngrupos para correlación', 
                              ha='center', va='center', fontsize=12)
        except Exception as e:
            axes[0,0].text(0.5, 0.5, f'Error en correlación:\n{str(e)}', 
                          ha='center', va='center', fontsize=10)
        
        # Lag correlation (autocorrelación con diferentes lags)
        try:
            df_diario = df.groupby('fecha')[columna_valor].sum().reset_index()
            serie = df_diario.set_index('fecha')[columna_valor]
            
            lags = range(1, 8)  # Lags de 1 a 7 días
            correlaciones = [serie.autocorr(lag=lag) for lag in lags]
            
            axes[0,1].bar(lags, correlaciones, color='skyblue', alpha=0.7)
            axes[0,1].set_xlabel('Lag (días)')
            axes[0,1].set_ylabel('Correlación')
            axes[0,1].set_title('Autocorrelación con Diferentes Lags')
            axes[0,1].grid(True, alpha=0.3)
        except:
            axes[0,1].text(0.5, 0.5, 'Error en autocorrelación', 
                          ha='center', va='center', fontsize=12)
        
        # Scatter plot de crecimiento (día anterior vs día actual)
        try:
            df_diario = df.groupby('fecha')[columna_valor].sum().reset_index()
            df_diario['ventas_previas'] = df_diario[columna_valor].shift(1)
            df_diario = df_diario.dropna()
            
            axes[1,0].scatter(df_diario['ventas_previas'], df_diario[columna_valor], alpha=0.5)
            axes[1,0].set_xlabel('Ventas Día Anterior')
            axes[1,0].set_ylabel('Ventas Día Actual')
            axes[1,0].set_title('Relación Día Anterior vs Día Actual')
        except:
            axes[1,0].text(0.5, 0.5, 'Error en scatter plot', 
                          ha='center', va='center', fontsize=12)
        
        # Distribución de diferencias diarias
        try:
            df_diario = df.groupby('fecha')[columna_valor].sum().reset_index()
            df_diario['diferencias'] = df_diario[columna_valor].diff()
            df_diario = df_diario.dropna()
            
            axes[1,1].hist(df_diario['diferencias'], bins=50, alpha=0.7, edgecolor='black')
            axes[1,1].axvline(df_diario['diferencias'].mean(), color='red', linestyle='--', label='Media')
            axes[1,1].set_xlabel('Diferencias Diarias')
            axes[1,1].set_ylabel('Frecuencia')
            axes[1,1].set_title('Distribución de Cambios Diarios')
            axes[1,1].legend()
        except:
            axes[1,1].text(0.5, 0.5, 'Error en análisis de diferencias', 
                          ha='center', va='center', fontsize=12)
        
        plt.tight_layout()
        return plot_to_base64(fig)
    
    def _analisis_volatilidad(self, df, columna_valor):
        """Análisis de volatilidad y riesgo"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Preparar datos diarios
        df_diario = df.groupby('fecha')[columna_valor].sum().reset_index()
        df_diario = df_diario.set_index('fecha')
        serie = df_diario[columna_valor]
        
        # Bandas de volatilidad (Bollinger Bands)
        window = 20
        rolling_mean = serie.rolling(window=window).mean()
        rolling_std = serie.rolling(window=window).std()
        
        axes[0,0].plot(serie.index, serie, label='Serie Original', alpha=0.5)
        axes[0,0].plot(rolling_mean.index, rolling_mean, label=f'Media Móvil {window}d', linewidth=2)
        axes[0,0].fill_between(rolling_mean.index, 
                              rolling_mean - 2*rolling_std, 
                              rolling_mean + 2*rolling_std, 
                              alpha=0.2, label='Banda ±2σ')
        axes[0,0].set_title('Bandas de Volatilidad (Bollinger Bands)')
        axes[0,0].legend()
        
        # Volatilidad histórica (desviación estándar móvil)
        volatilidad_30d = serie.rolling(window=30).std()
        axes[0,1].plot(volatilidad_30d.index, volatilidad_30d, color='red', linewidth=2)
        axes[0,1].set_title('Volatilidad Histórica (30 días)')
        axes[0,1].set_ylabel('Desviación Estándar')
        axes[0,1].grid(True, alpha=0.3)
        
        # Drawdown (reducción desde máximos)
        rolling_max = serie.rolling(window=len(serie), min_periods=1).max()
        drawdown = (serie - rolling_max) / rolling_max * 100
        
        axes[1,0].fill_between(drawdown.index, drawdown, 0, alpha=0.3, color='red')
        axes[1,0].plot(drawdown.index, drawdown, color='red', linewidth=1)
        axes[1,0].set_title('Drawdown (% desde máximos históricos)')
        axes[1,0].set_ylabel('Drawdown (%)')
        axes[1,0].grid(True, alpha=0.3)
        
        # Distribución de rendimientos diarios
        try:
            rendimientos = serie.pct_change().dropna() * 100
            axes[1,1].hist(rendimientos, bins=50, alpha=0.7, edgecolor='black')
            axes[1,1].axvline(rendimientos.mean(), color='red', linestyle='--', label=f'Media: {rendimientos.mean():.2f}%')
            axes[1,1].set_xlabel('Rendimiento Diario (%)')
            axes[1,1].set_ylabel('Frecuencia')
            axes[1,1].set_title('Distribución de Rendimientos Diarios')
            axes[1,1].legend()
        except:
            axes[1,1].text(0.5, 0.5, 'Error en análisis de rendimientos', 
                          ha='center', va='center', fontsize=12)
        
        plt.tight_layout()
        return plot_to_base64(fig)

    # ✅ MÉTODOS DE CÁLCULO QUE FALTAN
    def _calcular_tendencia(self, serie):
        """Calcula la pendiente de la tendencia"""
        if len(serie) < 2:
            return 0
        x = np.arange(len(serie))
        y = serie.values
        slope = np.polyfit(x, y, 1)[0]
        return slope
    
    def _detectar_estacionalidad_fuerte(self, serie):
        """Detecta si hay estacionalidad fuerte en los datos"""
        if len(serie) < 30:
            return "Datos insuficientes"
        
        # Simple detección basada en variación estacional
        try:
            serie_mensual = serie.resample('M').mean()
            if len(serie_mensual) > 6:
                cv_mensual = serie_mensual.std() / serie_mensual.mean()
                return "Alta" if cv_mensual > 0.3 else "Moderada" if cv_mensual > 0.1 else "Baja"
        except:
            pass
        
        return "Moderada"
    
    
    def _analisis_estacionalidad_completo(self, df, columna_valor, grupo_principal):
        """Análisis detallado de estacionalidad CON días festivos"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # 1. Estacionalidad mensual (mantener)
        df['mes'] = df['fecha'].dt.month
        ventas_mes = df.groupby([grupo_principal, 'mes'])[columna_valor].mean().reset_index()
        sns.lineplot(data=ventas_mes, x='mes', y=columna_valor, hue=grupo_principal, ax=axes[0,0], marker='o')
        axes[0,0].set_title('Estacionalidad Mensual')
        axes[0,0].set_xlabel('Mes')
        
        # 2. Estacionalidad semanal (mantener)
        df['dia_semana'] = df['fecha'].dt.day_name()
        orden_dias = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
        df['dia_semana'] = pd.Categorical(df['dia_semana'], categories=orden_dias, ordered=True)
        ventas_semana = df.groupby([grupo_principal, 'dia_semana'])[columna_valor].mean().reset_index()
        sns.lineplot(data=ventas_semana, x='dia_semana', y=columna_valor, hue=grupo_principal, ax=axes[0,1], marker='o')
        axes[0,1].set_title('Estacionalidad Semanal')
        axes[0,1].tick_params(axis='x', rotation=45)
        
        # 3. ANÁLISIS DE DÍAS FESTIVOS Y FINES DE SEMANA (NUEVO)
        self._plot_analisis_dias_especiales(df, columna_valor, axes[1,0])
        
        # 4. COMPARACIÓN DE PROMEDIOS POR TIPO DE DÍA (NUEVO)
        self._plot_comparacion_tipos_dia(df, columna_valor, grupo_principal, axes[1,1])
        
        plt.tight_layout()
        return plot_to_base64(fig)
    
    def _plot_analisis_dias_especiales(self, df, columna_valor, ax):
        """Gráfico de serie temporal marcando días especiales"""
        # Agrupar datos diarios
        df_diario = df.groupby('fecha').agg({
            columna_valor: 'sum',
            'es_festivo': 'first',
            'es_fin_semana': 'first',
            'tipo_dia': 'first'
        }).reset_index()
        
        # Crear el gráfico de línea principal
        ax.plot(df_diario['fecha'], df_diario[columna_valor], 
                color='blue', alpha=0.7, linewidth=1, label='Ventas Diarias')
        
        # Marcar fines de semana
        fines_semana = df_diario[df_diario['es_fin_semana'] == True]
        ax.scatter(fines_semana['fecha'], fines_semana[columna_valor], 
                  color='orange', s=30, alpha=0.8, label='Fin de Semana', marker='s')
        
        # Marcar días festivos
        festivos = df_diario[df_diario['es_festivo'] == True]
        ax.scatter(festivos['fecha'], festivos[columna_valor], 
                  color='red', s=50, alpha=1.0, label='Día Festivo', marker='D')
        
        ax.set_title('Serie Temporal con Días Especiales')
        ax.set_xlabel('Fecha')
        ax.set_ylabel(columna_valor)
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax.grid(True, alpha=0.3)
        
        # Rotar etiquetas de fecha para mejor legibilidad
        ax.tick_params(axis='x', rotation=45)
    
    def _plot_comparacion_tipos_dia(self, df, columna_valor, grupo_principal, ax):
        """Comparación de métricas por tipo de día"""
        # Calcular promedios por tipo de día
        stats_tipo_dia = df.groupby('tipo_dia').agg({
            columna_valor: ['mean', 'std', 'count']
        }).round(2)
        
        stats_tipo_dia.columns = ['promedio', 'desviacion', 'conteo']
        stats_tipo_dia = stats_tipo_dia.reset_index()
        
        # Crear gráfico de barras
        bars = ax.bar(stats_tipo_dia['tipo_dia'], stats_tipo_dia['promedio'], 
                     color=['lightblue', 'lightcoral', 'lightgreen'], alpha=0.8)
        
        # Agregar valores en las barras
        for bar, valor in zip(bars, stats_tipo_dia['promedio']):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + height*0.01,
                   f'${valor:,.0f}', ha='center', va='bottom', fontweight='bold')
        
        ax.set_title('Promedio de Ventas por Tipo de Día')
        ax.set_ylabel(f'Promedio {columna_valor}')
        ax.grid(True, alpha=0.3, axis='y')
        
        # Agregar tabla con estadísticas detalladas
        tabla_data = []
        for _, row in stats_tipo_dia.iterrows():
            tabla_data.append([
                row['tipo_dia'],
                f"${row['promedio']:,.0f}",
                f"${row['desviacion']:,.0f}",
                row['conteo']
            ])
        
        # Crear tabla en el gráfico
        tabla = ax.table(cellText=tabla_data,
                        colLabels=['Tipo Día', 'Promedio', 'Desv.Std', 'Días'],
                        cellLoc='center',
                        bbox=[0, -0.5, 1, 0.3])
        tabla.auto_set_font_size(False)
        tabla.set_fontsize(9)
    
    def _calculate_advanced_metrics(self, df, columna_valor):
        """Calcula métricas avanzadas INCLUYENDO análisis de días especiales"""
        df_diario = df.groupby('fecha').agg({
            columna_valor: 'sum',
            'tipo_dia': 'first',
            'es_festivo': 'first', 
            'es_fin_semana': 'first'
        }).reset_index()
        
        serie = df_diario.set_index('fecha')[columna_valor]
        
        # Métricas básicas (usando datos diarios)
        metrics = {
            "total_ventas": float(serie.sum()),
            "promedio_diario": float(serie.mean()),
            "dias_analizados": len(serie),
            "locales_unicos": df['local'].nunique(),
            "articulos_unicos": df['articulo'].nunique()
        }
        
        # Métricas avanzadas de días especiales (usando datos DIARIOS)
        if len(df_diario) > 0:
            # Calcular totales por tipo de día (usando datos diarios)
            stats_por_tipo = df_diario.groupby('tipo_dia')[columna_valor].sum()
            total_ventas = stats_por_tipo.sum()
            
            # Calcular promedios DIARIOS por tipo de día (CORREGIDO)
            metrics.update({
                # Porcentaje de ventas por tipo de día
                "porcentaje_laboral": float((stats_por_tipo.get('laboral', 0) / total_ventas) * 100),
                "porcentaje_fin_semana": float((stats_por_tipo.get('fin_semana', 0) / total_ventas) * 100),
                "porcentaje_festivo": float((stats_por_tipo.get('festivo', 0) / total_ventas) * 100),
                
                # Promedios DIARIOS por tipo de día (CORREGIDO)
                "promedio_laboral": float(df_diario[df_diario['tipo_dia'] == 'laboral'][columna_valor].mean()),
                "promedio_fin_semana": float(df_diario[df_diario['tipo_dia'] == 'fin_semana'][columna_valor].mean()),
                "promedio_festivo": float(df_diario[df_diario['tipo_dia'] == 'festivo'][columna_valor].mean()),
                
                # Días analizados por tipo
                "dias_laborales": len(df_diario[df_diario['tipo_dia'] == 'laboral']),
                "dias_fin_semana": len(df_diario[df_diario['tipo_dia'] == 'fin_semana']),
                "dias_festivos": len(df_diario[df_diario['tipo_dia'] == 'festivo'])
            })
        
        # Métricas avanzadas existentes
        if len(serie) > 1:
            metrics.update({
                "volatilidad_diaria": float(serie.std()),
                "coeficiente_variacion": float(serie.std() / serie.mean() * 100),
                "maximo_diario": float(serie.max()),
                "minimo_diario": float(serie.min()),
                "tendencia_crecimiento": float(self._calcular_tendencia(serie)),
                "estacionalidad_fuerte": self._detectar_estacionalidad_fuerte(serie)
            })
    
        return metrics
    
    def _generate_forecasting_insights(self, df, columna_valor):
        """Genera insights automáticos INCLUYENDO días especiales"""
        df_diario = df.groupby('fecha')[columna_valor].sum().reset_index()
        serie = df_diario.set_index('fecha')[columna_valor]
        
        insights = []
        
        if len(serie) > 30:
            # Insights existentes de tendencia y estacionalidad
            tendencia = self._calcular_tendencia(serie)
            if tendencia > 0:
                insights.append("📈 **Tendencia creciente** detectada en los datos")
            elif tendencia < 0:
                insights.append("📉 **Tendencia decreciente** detectada en los datos")
            else:
                insights.append("➡️ **Tendencia estable** en los datos")
            
            # NUEVOS INSIGHTS SOBRE DÍAS ESPECIALES
            promedio_laboral = df[df['tipo_dia'] == 'laboral'][columna_valor].mean()
            promedio_fin_semana = df[df['tipo_dia'] == 'fin_semana'][columna_valor].mean()
            promedio_festivo = df[df['tipo_dia'] == 'festivo'][columna_valor].mean()
            
            # Insight sobre fines de semana
            if promedio_fin_semana > promedio_laboral * 1.2:
                insights.append("🎉 **Fines de semana fuertes**: +20% vs días laborales")
            elif promedio_fin_semana < promedio_laboral * 0.8:
                insights.append("🏢 **Días laborales fuertes**: fines de semana -20% vs laborales")
            else:
                insights.append("⚖️ **Patrón balanceado** entre días laborales y fines de semana")
            
            # Insight sobre días festivos
            if promedio_festivo > 0:
                if promedio_festivo > promedio_laboral * 1.5:
                    insights.append("🎊 **Días festivos muy fuertes**: +50% vs días laborales")
                elif promedio_festivo > promedio_laboral * 1.2:
                    insights.append("🎁 **Días festivos fuertes**: +20% vs días laborales")
                elif promedio_festivo < promedio_laboral * 0.8:
                    insights.append("📉 **Días festivos débiles**: -20% vs días laborales")
            
            # Insight sobre volatilidad por tipo de día
            cv_laboral = df[df['tipo_dia'] == 'laboral'][columna_valor].std() / promedio_laboral
            cv_fin_semana = df[df['tipo_dia'] == 'fin_semana'][columna_valor].std() / promedio_fin_semana
            
            if cv_fin_semana > cv_laboral * 1.5:
                insights.append("🎭 **Alta variabilidad fines de semana**: patrones inconsistentes")
            elif cv_fin_semana < cv_laboral * 0.7:
                insights.append("📊 **Fines de semana estables**: patrones consistentes")
        
        else:
            insights.append("ℹ️ **Datos limitados** - se recomiendan más datos para pronósticos confiables")
        
        return insights
    
    