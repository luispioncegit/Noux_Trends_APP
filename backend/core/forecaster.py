import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any, Tuple
import warnings
warnings.filterwarnings('ignore')
import json
from datetime import datetime
import joblib
import os

# Nuevas dependencias para LightGBM
try:
    import lightgbm as lgb
    LGBM_AVAILABLE = True
except ImportError:
    LGBM_AVAILABLE = False

class Forecaster:
    """
    Forecaster con modelos reales de Prophet y ETS
    """
    
    def __init__(self, df: pd.DataFrame, models_dir: str = "models"):
        self.df = df.copy()
        self.df['fecha'] = pd.to_datetime(self.df['fecha'])
        self.models_dir = models_dir
        self.trained_models = {}
        self.metricas_entrenamiento = {}
        self.modelos_entrenados = False
        
        # Crear directorio de modelos si no existe
        os.makedirs(models_dir, exist_ok=True)
    
    def entrenar_modelos(self, variables: List[str], modelos: List[str], train_size: float = 0.8) -> Dict:
        """Entrena UN modelo global con LightGBM"""
        resultados_entrenamiento = {
            'modelos_entrenados': [],
            'metricas': {},
            'fecha_entrenamiento': datetime.now().isoformat(),
            'datos_grafico': {}  # ✅ NUEVO: agregar campo para gráfico
        }
        
        if not LGBM_AVAILABLE:
            print("❌ LightGBM no disponible")
            return resultados_entrenamiento
        
        try:
            # Preparar datos para modelo global
            print("🔄 Preparando datos para modelo global...")
            df_entrenamiento = self.preparar_datos_entrenamiento(variables)
            
            if df_entrenamiento.empty:
                print("❌ No hay suficientes datos para entrenar")
                return resultados_entrenamiento
            
            # Entrenar modelo LightGBM
            print("🤖 Entrenando modelo LightGBM global...")
            modelo, metricas, datos_grafico = self._entrenar_lightgbm_global(df_entrenamiento, variables, train_size)
            
            # Guardar modelo
            modelo_key = "lightgbm_global"
            self.trained_models[modelo_key] = {
                'modelo': modelo,
                'tipo': 'LightGBM_Global',
                'metricas': metricas,
                'variables_usadas': variables,
                'fecha_entrenamiento': datetime.now().isoformat()
            }
            
            self._guardar_modelo(self.trained_models[modelo_key], modelo_key)
            
            resultados_entrenamiento['modelos_entrenados'] = [modelo_key]
            resultados_entrenamiento['metricas'][modelo_key] = metricas
            resultados_entrenamiento['datos_grafico'] = datos_grafico  # ✅ NUEVO: agregar datos del gráfico

            self.modelos_entrenados = True
            #debug
            print(f"📊 Datos gráfico preparados: {bool(datos_grafico)}")
            print(f"  - Entrenamiento: {len(datos_grafico.get('fechas_entrenamiento', []))} puntos")
            print(f"  - Validación: {len(datos_grafico.get('fechas_validacion', []))} puntos")
            
            
            print("✅ Modelo LightGBM global entrenado exitosamente")
            
        except Exception as e:
            print(f"❌ Error entrenando modelo global: {str(e)}")
        
        return resultados_entrenamiento
    

    def preparar_datos_entrenamiento(self, variables: List[str]) -> pd.DataFrame:
        """Prepara datos con features para el modelo global"""
        try:
            # Crear dataset a nivel diario por combinación de variables
            df_agrupado = self.df.groupby(['fecha'] + variables)['venta'].sum().reset_index()
            
            # Ordenar por fecha
            df_agrupado = df_agrupado.sort_values(['fecha'] + variables)
            
            # ✅ CONVERTIR VARIABLES CATEGÓRICAS A TIPO CATEGORY
            for var in variables:
                if var in df_agrupado.columns:
                    df_agrupado[var] = df_agrupado[var].astype('category')
            
            # Crear features temporales
            df_agrupado = self._crear_features_temporales(df_agrupado)
            
            # Crear lags y rolling features por combinación de variables
            df_agrupado = self._crear_features_lags(df_agrupado, variables)
            
            # Limpiar valores infinitos y NaN
            df_agrupado = df_agrupado.replace([np.inf, -np.inf], np.nan)
            
            # Eliminar filas con venta = 0 o NaN en venta
            df_agrupado = df_agrupado[(df_agrupado['venta'] > 0) & (df_agrupado['venta'].notna())]
            
            # ✅ LIMPIAR NaN EN FEATURES
            # Rellenar NaN en lags y rolling con 0 o valores apropiados
            for col in df_agrupado.columns:
                if col.startswith('lag_') or col.startswith('rolling_'):
                    df_agrupado[col] = df_agrupado[col].fillna(0)
            
            print(f"📊 Datos preparados: {len(df_agrupado)} filas, {len(variables)} variables")
            print(f"📋 Tipos de datos: {df_agrupado.dtypes.to_dict()}")
            
            return df_agrupado
                
        except Exception as e:
            print(f"❌ Error preparando datos: {str(e)}")
            return pd.DataFrame()
    
    def preparar_datos_para_variable(self, variable: str) -> pd.DataFrame:
        """Prepara datos diarios agregados para una variable categórica"""
        # Para variables categóricas reales, agrupar por fecha y la variable
        df_agrupado = self.df.groupby(['fecha', variable])['venta'].sum().reset_index()
        return df_agrupado
    
    
    def _crear_features_temporales(self, df: pd.DataFrame) -> pd.DataFrame:
        """Crea features temporales"""
        df_copy = df.copy()
        
        # Features básicas de tiempo
        df_copy['mes'] = df_copy['fecha'].dt.month
        df_copy['dia_semana'] = df_copy['fecha'].dt.dayofweek
        df_copy['dia_mes'] = df_copy['fecha'].dt.day
        df_copy['semana_ano'] = df_copy['fecha'].dt.isocalendar().week
        df_copy['trimestre'] = df_copy['fecha'].dt.quarter
        
        # Features cíclicas
        df_copy['mes_sin'] = np.sin(2 * np.pi * df_copy['mes']/12)
        df_copy['mes_cos'] = np.cos(2 * np.pi * df_copy['mes']/12)
        df_copy['dia_semana_sin'] = np.sin(2 * np.pi * df_copy['dia_semana']/7)
        df_copy['dia_semana_cos'] = np.cos(2 * np.pi * df_copy['dia_semana']/7)
        
        return df_copy
    
    def _crear_features_lags(self, df: pd.DataFrame, variables: List[str]) -> pd.DataFrame:
        """Crea features de lags y rolling statistics"""
        # Para MVP, crear lags simples a nivel global
        # En una versión avanzada, haríamos groupby por variables
        
        df_sorted = df.sort_values('fecha')
        
        # Lags simples
        for lag in [1, 2, 3, 7, 14, 30]:
            df_sorted[f'lag_{lag}'] = df_sorted.groupby(variables)['venta'].shift(lag)
        
        # Rolling statistics
        for window in [7, 14, 30]:
            df_sorted[f'rolling_mean_{window}'] = df_sorted.groupby(variables)['venta'].transform(
                lambda x: x.rolling(window=window, min_periods=1).mean()
            )
            df_sorted[f'rolling_std_{window}'] = df_sorted.groupby(variables)['venta'].transform(
                lambda x: x.rolling(window=window, min_periods=1).std()
            )
        
        return df_sorted
    


    def _entrenar_lightgbm_global(self, df: pd.DataFrame, variables: List[str], train_size: float) -> Tuple[Any, Dict]:
        """Entrena el modelo LightGBM global"""
        # Preparar features y target
        feature_columns = [
            'mes', 'dia_semana', 'dia_mes', 'semana_ano', 'trimestre',
            'mes_sin', 'mes_cos', 'dia_semana_sin', 'dia_semana_cos'
        ]
        
        # Agregar lags y rolling features
        for lag in [1, 2, 3, 7, 14, 30]:
            feature_columns.append(f'lag_{lag}')
        for window in [7, 14, 30]:
            feature_columns.extend([f'rolling_mean_{window}', f'rolling_std_{window}'])
        
        # Agregar variables categóricas
        feature_columns.extend(variables)
        
        # Filtrar columnas existentes
        feature_columns = [col for col in feature_columns if col in df.columns]
        
        # ✅ VERIFICAR TIPOS DE DATOS
        print(f"🔍 Verificando tipos de datos...")
        for col in feature_columns:
            print(f"   {col}: {df[col].dtype}")
        
        # Eliminar filas con NaN
        df_clean = df.dropna(subset=feature_columns + ['venta'])
        
        if df_clean.empty:
            raise ValueError("No hay datos suficientes después de limpieza")
        
        # Split temporal
        fecha_corte = df_clean['fecha'].quantile(train_size)
        train_mask = df_clean['fecha'] <= fecha_corte
        test_mask = df_clean['fecha'] > fecha_corte
        
        X_train = df_clean[train_mask][feature_columns]
        y_train = df_clean[train_mask]['venta']
        X_test = df_clean[test_mask][feature_columns] 
        y_test = df_clean[test_mask]['venta']
        
        # Verificar que tenemos datos suficientes
        if len(X_train) == 0 or len(X_test) == 0:
            raise ValueError("No hay suficientes datos después del split")
        
        # ✅ LightGBM maneja automáticamente columnas de tipo 'category'
        # No necesitamos especificar categorical_features si ya son tipo category
        
        # Entrenar modelo
        model = lgb.LGBMRegressor(
            n_estimators=100,
            learning_rate=0.1,
            max_depth=7,
            random_state=42,
            verbose=-1  # Sin logs
        )
        
        try:
            # LightGBM detecta automáticamente las columnas categóricas
            model.fit(
                X_train, y_train,
                eval_set=[(X_test, y_test)],
                eval_metric='l1',  # MAE
                callbacks=[
                    lgb.early_stopping(stopping_rounds=20, verbose=False),
                    lgb.log_evaluation(period=0)  # Sin logs de evaluación
                ]
            )
        except Exception as e:
            print(f"⚠️ Error con early stopping, entrenando sin él: {str(e)}")
            # Fallback: entrenar sin early stopping
            model.fit(
                X_train, y_train,
                eval_set=[(X_test, y_test)],
                verbose=False
            )
        
        # Calcular métricas
        y_pred = model.predict(X_test)
        metricas = self._calcular_metricas(y_test.values, y_pred)
        
        print(f"✅ Modelo entrenado: {len(X_train)} train, {len(X_test)} test")
        print(f"📊 Métricas - MAPE: {metricas['mape']:.2f}%, RMSE: {metricas['rmse']:.2f}")
        # ✅ NUEVO: Preparar datos para el gráfico de validación
        datos_grafico = {
            'fechas_entrenamiento': df_clean[train_mask]['fecha'].tolist(),
            'ventas_entrenamiento': y_train.tolist(),
            'fechas_validacion': df_clean[test_mask]['fecha'].tolist(), 
            'ventas_validacion': y_test.tolist(),
            'predicciones_validacion': y_pred.tolist(),
            'fecha_corte': fecha_corte.isoformat()
        }
        
        return model, metricas, datos_grafico

    
    def _calcular_metricas(self, real: np.ndarray, predicho: np.ndarray) -> Dict[str, float]:
        """Calcula métricas de calidad del modelo"""
        try:
            # Filtrar valores no cero para evitar división por cero
            mask = real != 0
            if mask.sum() > 0:
                mape = np.mean(np.abs((real[mask] - predicho[mask]) / real[mask])) * 100
            else:
                mape = 0
                
            rmse = np.sqrt(np.mean((real - predicho) ** 2))
            
            # R²
            ss_res = np.sum((real - predicho) ** 2)
            ss_tot = np.sum((real - np.mean(real)) ** 2)
            r2 = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0
            
            return {
                'mape': float(mape),
                'rmse': float(rmse),
                'r2': float(r2),
                'muestras_test': len(real)
            }
        except:
            return {'mape': 999, 'rmse': 999, 'r2': 0, 'muestras_test': 0}
    
    def _guardar_modelo(self, modelo: Dict, nombre: str):
        """Guarda el modelo entrenado en disco"""
        try:
            modelo_path = os.path.join(self.models_dir, f"{nombre}.joblib")
            joblib.dump(modelo, modelo_path)
        except Exception as e:
            print(f"Error guardando modelo {nombre}: {str(e)}")
    
    def cargar_modelos(self):
        """Carga modelos previamente entrenados"""
        try:
            for archivo in os.listdir(self.models_dir):
                if archivo.endswith('.joblib'):
                    modelo_path = os.path.join(self.models_dir, archivo)
                    modelo = joblib.load(modelo_path)
                    nombre_modelo = archivo.replace('.joblib', '')
                    self.trained_models[nombre_modelo] = modelo
            self.modelos_entrenados = True
        except Exception as e:
            print(f"Error cargando modelos: {str(e)}")
    
########## Métodos de pronóstico ##########

    def generar_pronostico(self, modelo_key: str, horizonte: int, variables: List[str] = None) -> Dict[str, Any]:
        """Genera pronóstico usando el modelo LightGBM global"""
        if modelo_key not in self.trained_models:
            return self._pronostico_fallback(horizonte)
        
        modelo_info = self.trained_models[modelo_key]
        
        try:
            if modelo_info['tipo'] == 'LightGBM_Global':
                return self._pronostico_lightgbm_global(modelo_info, horizonte, variables)
            else:
                return self._pronostico_fallback(horizonte)
                
        except Exception as e:
            print(f"Error en pronóstico {modelo_key}: {str(e)}")
            return self._pronostico_fallback(horizonte)
        

    def _pronostico_lightgbm_global(self, modelo_info: Dict, horizonte: int, variables: List[str]) -> Dict:
        """Genera pronóstico con el modelo LightGBM global"""
        modelo = modelo_info['modelo']
        variables_usadas = modelo_info.get('variables_usadas', variables or [])
        
        # Generar fechas futuras
        ultima_fecha = self.df['fecha'].max()
        fechas_futuras = pd.date_range(
            start=ultima_fecha + pd.Timedelta(days=1),
            periods=horizonte,
            freq='D'
        )
        
        # Preparar datos para predicción
        pronosticos = []
        for fecha in fechas_futuras:
            # Para cada fecha, generar pronóstico para todas las combinaciones de variables
            pronostico_fecha = self._generar_pronostico_fecha(modelo, fecha, variables_usadas)
            pronosticos.extend(pronostico_fecha)
        
        # Agrupar por fecha para el nivel total
        df_pronostico = pd.DataFrame(pronosticos)
        df_agrupado = df_pronostico.groupby('fecha').agg({
            'pronostico': 'sum'
        }).reset_index()
        
        # Calcular intervalos de confianza simples
        pronostico_array = np.array(df_agrupado['pronostico'])
        error_promedio = pronostico_array * 0.1  # 10% de error estimado
        
        # ✅ CORREGIR: Retornar la estructura que espera el frontend
        return {
            'pronostico': df_agrupado['pronostico'].tolist(),
            'intervalo_inferior': (pronostico_array - error_promedio).tolist(),  # ✅ Campo directo
            'intervalo_superior': (pronostico_array + error_promedio).tolist(),  # ✅ Campo directo
            'fechas_futuras': [f.isoformat() for f in fechas_futuras],
            'modelo_utilizado': 'LightGBM_Global',
            'metricas_modelo': modelo_info['metricas'],
            'pronosticos_desagregados': pronosticos
        }
    

    def _generar_pronostico_fecha(self, modelo, fecha: datetime, variables: List[str]) -> List[Dict]:
        """Genera pronóstico para una fecha específica y todas las combinaciones de variables"""
        pronosticos = []
        
        # Obtener todas las combinaciones únicas de variables
        combinaciones = self._obtener_combinaciones_variables(variables)
        
        # ✅ DEBUG: Ver combinaciones
        print(f"🔍 DEBUG _generar_pronostico_fecha:")
        print(f"  - Fecha: {fecha}")
        print(f"  - Combinaciones: {len(combinaciones)}")
        print(f"  - Variables: {variables}")

        for combinacion in combinaciones:
            try:
                # Preparar features para esta combinación
                features = self._preparar_features_prediccion(fecha, combinacion, variables)
                
                if features is not None:
                    # Hacer predicción
                    pronostico_valor = modelo.predict([features])[0]
                    pronostico_valor = max(pronostico_valor, 0)  # No valores negativos
                    
                    # ✅ CALCULAR INTERVALOS para cada pronóstico individual
                    error_relativo = 0.1  # 10% de error estimado
                    intervalo_inferior = pronostico_valor * (1 - error_relativo)
                    intervalo_superior = pronostico_valor * (1 + error_relativo)
                    
                    pronosticos.append({
                        'fecha': fecha.isoformat(),
                        'pronostico': float(pronostico_valor),
                        'intervalo_inferior': float(intervalo_inferior),  # ✅ AGREGAR
                        'intervalo_superior': float(intervalo_superior),  # ✅ AGREGAR
                        **combinacion
                    })
                    
            except Exception as e:
                print(f"Error en predicción para {combinacion}: {str(e)}")
                continue
        # ✅ DEBUG: Resumen de la fecha
        if pronosticos:
            total_fecha = sum(p['pronostico'] for p in pronosticos)
            print(f"  - TOTAL para {fecha}: {total_fecha:.2f} (de {len(pronosticos)} combinaciones)")
        
        return pronosticos
    

    def _obtener_combinaciones_variables(self, variables: List[str]) -> List[Dict]:
        """Obtiene todas las combinaciones únicas de variables categóricas"""
        if not variables:
            return [{}]
        
        # ✅ DEBUG: Ver qué variables estamos procesando
        print(f"🔍 DEBUG _obtener_combinaciones_variables:")
        print(f"  - Variables recibidas: {variables}")

        # Para MVP, usar las combinaciones más comunes (top 10 por variable)
        combinaciones = []
        
        for variable in variables:
            valores_unicos = self.df[variable].unique()[:5]  # Top 5 valores por variable
            if not combinaciones:
                # Primera variable
                combinaciones = [{variable: valor} for valor in valores_unicos]
            else:
                # Combinar con variables existentes
                nuevas_combinaciones = []
                for combo in combinaciones:
                    for valor in valores_unicos:
                        nueva_combo = combo.copy()
                        nueva_combo[variable] = valor
                        nuevas_combinaciones.append(nueva_combo)
                combinaciones = nuevas_combinaciones
        
        resultado = combinaciones[:20]  # Limitar a 20 combinaciones para MVP
        print(f"  - Combinaciones finales: {len(resultado)}")
    
        return resultado
    

    def _preparar_features_prediccion(self, fecha: datetime, combinacion: Dict, variables: List[str]) -> Optional[List]:
        """Prepara features para una combinación específica en una fecha futura"""
        try:
            features = []
            
            # Features temporales (mantener igual)
            features.extend([
                fecha.month,                           # mes
                fecha.dayofweek,                       # dia_semana  
                fecha.day,                             # dia_mes
                fecha.isocalendar().week,              # semana_ano
                fecha.quarter,                         # trimestre
                np.sin(2 * np.pi * fecha.month/12),    # mes_sin
                np.cos(2 * np.pi * fecha.month/12),    # mes_cos
                np.sin(2 * np.pi * fecha.dayofweek/7), # dia_semana_sin
                np.cos(2 * np.pi * fecha.dayofweek/7)  # dia_semana_cos
            ])
            
            # Lags y rolling features (mantener igual)
            lag_values = []
            for lag in [1, 2, 3, 7, 14, 30]:
                valor_lag = self._obtener_valor_historico(combinacion, lag)
                lag_values.append(valor_lag)
                features.append(valor_lag)
            
            rolling_values = []
            for window in [7, 14, 30]:
                valor_rolling = self._obtener_rolling_historico(combinacion, window)
                rolling_values.extend([valor_rolling, valor_rolling * 0.1])
                features.extend([valor_rolling, valor_rolling * 0.1])
            
            # ✅ CORREGIR: Variables categóricas - usar codificación numérica
            for variable in variables:
                if variable in combinacion:
                    # Convertir categoría a código numérico basado en los datos históricos
                    codigo = self._obtener_codigo_categoria(variable, combinacion[variable])
                    features.append(codigo)
                else:
                    features.append(-1)  # Valor por defecto para categoría faltante
            
            # ✅ DEBUG CRÍTICO: Ver valores de features clave
            if len(features) > 0:  # Solo debug para la primera combinación de cada fecha
                print(f"🔍 DEBUG Features para {combinacion} en {fecha}:")
                print(f"  - Lags (1,7,30): {lag_values[0]:.1f}, {lag_values[3]:.1f}, {lag_values[5]:.1f}")
                print(f"  - Rolling (7,14,30): {rolling_values[0]:.1f}, {rolling_values[2]:.1f}, {rolling_values[4]:.1f}")
                print(f"  - Categóricas: {[features[-2], features[-1]]}")

            return features
            
        except Exception as e:
            print(f"Error preparando features: {str(e)}")
            return None
        
    def _obtener_codigo_categoria(self, variable: str, valor: str) -> int:
        """Obtiene el código numérico para una categoría basado en datos históricos"""
        try:
            # Obtener valores únicos de la variable en los datos históricos
            valores_unicos = self.df[variable].unique()
            # Crear mapeo de valor a código numérico
            mapeo = {valor: codigo for codigo, valor in enumerate(valores_unicos)}
            return mapeo.get(valor, -1)  # -1 si no se encuentra
        except Exception as e:
            print(f"Error codificando {variable}={valor}: {str(e)}")
            return -1


    def _obtener_valor_historico(self, combinacion: Dict, lag: int) -> float:
        """Obtiene valor histórico promedio para una combinación y lag"""
        try:
            # Filtrar datos para esta combinación
            mask = pd.Series(True, index=self.df.index)
            for var, valor in combinacion.items():
                mask &= (self.df[var] == valor)
            
            df_filtrado = self.df[mask]
            
            if len(df_filtrado) == 0:
                print(f"  - lag_{lag}: NO DATOS -> promedio general {self.df['venta'].mean():.1f}")
                return self.df['venta'].mean()  # Fallback al promedio general
            
            # Obtener valor del día equivalente al lag
            fecha_referencia = self.df['fecha'].max() - pd.Timedelta(days=lag)
            valor_lag = df_filtrado[df_filtrado['fecha'] == fecha_referencia]['venta']
            
            if len(valor_lag) > 0:
                print(f"  - lag_{lag}: VALOR REAL {float(valor_lag.iloc[0]):.1f}")
                return float(valor_lag.iloc[0])
            else:
                print(f"  - lag_{lag}: NO FECHA -> promedio combinación {float(df_filtrado['venta'].mean()):.1f}")
                return float(df_filtrado['venta'].mean())  # Fallback al promedio de la combinación
                
        except:
            print(f"  - lag_{lag}: ERROR -> promedio general {float(self.df['venta'].mean()):.1f}")
            return float(self.df['venta'].mean())  # Fallback general
    
    
    def _obtener_rolling_historico(self, combinacion: Dict, window: int) -> float:
        """Obtiene rolling average histórico para una combinación"""
        try:
            # Filtrar datos para esta combinación
            mask = pd.Series(True, index=self.df.index)
            for var, valor in combinacion.items():
                mask &= (self.df[var] == valor)
            
            df_filtrado = self.df[mask]
            
            if len(df_filtrado) == 0:
                return self.df['venta'].mean()
            
            # Calcular rolling average de los últimos 'window' días
            df_reciente = df_filtrado[df_filtrado['fecha'] > (self.df['fecha'].max() - pd.Timedelta(days=window))]
            
            if len(df_reciente) > 0:
                return float(df_reciente['venta'].mean())
            else:
                return float(df_filtrado['venta'].mean())
                
        except:
            return float(self.df['venta'].mean())

    
    def _pronostico_fallback(self, horizonte: int) -> Dict:
        """Pronóstico fallback"""
        fechas_futuras = pd.date_range(
            start=pd.Timestamp.now().normalize(),
            periods=horizonte,
            freq='D'
        )
        
        promedio_ventas = self.df['venta'].mean()
        
        return {
            'pronostico': [promedio_ventas] * horizonte,
            'intervalo_confianza': {
                'inferior': [promedio_ventas * 0.7] * horizonte,
                'superior': [promedio_ventas * 1.3] * horizonte
            },
            'fechas_futuras': [f.isoformat() for f in fechas_futuras],
            'modelo_utilizado': 'Fallback',
            'metricas_modelo': {'mape': 999, 'rmse': 999, 'r2': 0}
        }
    
    ######### Métodos de pronóstico multi-nivel #########
    
    def generar_pronosticos_multi_nivel(self, variables: List[str], modelo_principal: str, horizonte: int) -> Dict[str, Any]:
        """Genera pronósticos a múltiples niveles usando el modelo global"""
        try:
            modelo_key = "lightgbm_global"
            
            # ✅ PRIMERO: Intentar cargar modelos si no hay ninguno
            if not self.trained_models:
                self.cargar_modelos()
                
            # ✅ SEGUNDO: Si el modelo específico no está, intentar cargarlo
            if modelo_key not in self.trained_models:
                self.cargar_modelos()  # Intentar cargar de nuevo
                    
            # ✅ TERCERO: Si después de cargar sigue sin estar, usar fallback
            if modelo_key not in self.trained_models:
                print(f"❌ Modelo {modelo_key} no encontrado después de cargar. Usando fallback.")
                return self._fallback_multi_nivel(variables, horizonte)
                
            # ✅ CUARTO: Generar pronóstico con el modelo cargado
            resultado = self.generar_pronostico(modelo_key, horizonte, variables)
            print(f"🔍 DEBUG: modelo_key = {modelo_key}")
            print(f"🔍 DEBUG: trained_models keys = {list(self.trained_models.keys())}")
            print(f"🔍 DEBUG: modelo_key in trained_models = {modelo_key in self.trained_models}")

            if modelo_key in self.trained_models:
                modelo_info = self.trained_models[modelo_key]
                print(f"🔍 DEBUG: modelo_info tipo = {modelo_info.get('tipo', 'NO_TIPO')}")

            # Verificar que no sea fallback
            if resultado.get('modelo_utilizado') == 'Fallback':
                print("❌ El modelo existe pero generó fallback. Revisar características.")
                return self._fallback_multi_nivel(variables, horizonte)
                
            dimension_ancla = self._seleccionar_dimension_ancla(variables)
            
            return {
                'forecast_data': resultado.get('pronosticos_desagregados', []),
                'pronostico_total': resultado,
                'modelo_utilizado': resultado.get('modelo_utilizado', 'LightGBM_Global'),
                'metricas_calidad': resultado.get('metricas_modelo', {}),
                'dimension_ancla': dimension_ancla,
                'variables_desagregadas': variables
            }
                
        except Exception as e:
            print(f"Error en pronóstico multi-nivel: {str(e)}")
            return self._fallback_multi_nivel(variables, horizonte)
        

    def _fallback_multi_nivel(self, variables: List[str], horizonte: int) -> Dict[str, Any]:
        """Fallback para pronóstico multi-nivel"""
        # Usar proporciones históricas como fallback
        pronostico_total = self._pronostico_fallback(horizonte)
        proporciones = self._calcular_proporciones_historicas(variables, horizonte)
        pronosticos_desagregados = self._desagregar_pronostico(pronostico_total, proporciones, variables, horizonte)
        
        return {
            'forecast_data': pronosticos_desagregados,
            'pronostico_total': pronostico_total,
            'modelo_utilizado': 'Fallback_Proporciones',
            'metricas_calidad': {'mape': 999, 'rmse': 999, 'r2': 0},
            'dimension_ancla': self._seleccionar_dimension_ancla(variables),
            'variables_desagregadas': variables
        }
    
    
    def _calcular_proporciones_historicas(self, variables: List[str], horizonte: int) -> Dict[str, Any]:
        """Calcula proporciones históricas para desagregación"""
        proporciones = {}
        
        # Usar los últimos 30 días para calcular proporciones
        fecha_limite = self.df['fecha'].max() - pd.Timedelta(days=30)
        df_reciente = self.df[self.df['fecha'] > fecha_limite]
        
        for variable in variables:
            if variable == 'total':
                continue
                
            # Calcular proporciones por categoría
            df_agrupado = df_reciente.groupby([variable])['venta'].sum()
            total_ventas = df_agrupado.sum()
            
            proporciones_variable = {}
            for categoria in df_agrupado.index:
                proporcion = df_agrupado[categoria] / total_ventas
                proporciones_variable[categoria] = float(proporcion)
            
            proporciones[variable] = proporciones_variable
        
        return proporciones
    
    def _desagregar_pronostico(self, pronostico_total: Dict, proporciones: Dict, 
                             variables: List[str], horizonte: int) -> Dict[str, Any]:
        """Desagrega el pronóstico total usando proporciones"""
        
        pronosticos_combinados = []
        
        # Generar todas las combinaciones posibles
        categorias_por_variable = {}
        for variable in variables:
            if variable != 'total' and variable in proporciones:
                categorias_por_variable[variable] = list(proporciones[variable].keys())
        
        # Para MVP, solo generamos algunas combinaciones principales
        combinaciones_principales = self._generar_combinaciones_principales(categorias_por_variable)
        
        # Aplicar proporciones a cada combinación
        for combinacion in combinaciones_principales:
            proporcion_total = 1.0
            
            for variable, categoria in combinacion.items():
                if variable in proporciones and categoria in proporciones[variable]:
                    proporcion_total *= proporciones[variable][categoria]
            
            # Generar pronóstico para esta combinación
            for i in range(horizonte):
                pronostico_combinado = {
                    'fecha': pronostico_total['fechas_futuras'][i],
                    'pronostico': pronostico_total['pronostico'][i] * proporcion_total,
                    'intervalo_inferior': pronostico_total['intervalo_confianza']['inferior'][i] * proporcion_total,
                    'intervalo_superior': pronostico_total['intervalo_confianza']['superior'][i] * proporcion_total,
                    'proporcion': proporcion_total
                }
                
                # Agregar información de categorías
                for variable, categoria in combinacion.items():
                    pronostico_combinado[variable] = categoria
                
                pronosticos_combinados.append(pronostico_combinado)
        
        return pronosticos_combinados
    
    def _generar_combinaciones_principales(self, categorias_por_variable: Dict[str, List]) -> List[Dict]:
        """Genera combinaciones principales basadas en volumen histórico"""
        combinaciones = []
        
        # Para MVP, usar solo las categorías principales de cada variable
        categorias_principales = {}
        for variable, categorias in categorias_por_variable.items():
            # Tomar máximo 3 categorías principales por variable
            categorias_principales[variable] = categorias[:3]
        
        # Generar combinaciones (producto cartesiano limitado)
        from itertools import product
        
        variables = list(categorias_principales.keys())
        if variables:
            # Solo generar combinaciones para las primeras 2 variables para MVP
            variables_limite = variables[:2]
            listas_categorias = [categorias_principales[var] for var in variables_limite]
            
            for combo in product(*listas_categorias):
                combinacion = {}
                for i, var in enumerate(variables_limite):
                    combinacion[var] = combo[i]
                combinaciones.append(combinacion)
        
        return combinaciones
    

    def _seleccionar_dimension_ancla(self, variables: List[str]) -> str:
        """Selecciona automáticamente la dimensión más estable como ancla"""
        if 'local' in variables:
            return 'local'
        elif 'categoria' in variables:
            return 'categoria'
        elif 'articulo' in variables:
            return 'articulo'
        else:
            return 'total'
        
    

    # MANTENER métodos existentes (sin cambios):
    def get_modelos_disponibles(self) -> Dict[str, Any]:
        """Retorna información de modelos entrenados disponibles"""
        if not self.modelos_entrenados:
            self.cargar_modelos()
        
        modelos_info = {}
        for modelo_key, modelo_info in self.trained_models.items():
            modelos_info[modelo_key] = {
                'tipo': modelo_info.get('tipo', 'Desconocido'),
                'metricas': modelo_info.get('metricas', {}),
                'variables_usadas': modelo_info.get('variables_usadas', []),
                'fecha_entrenamiento': modelo_info.get('fecha_entrenamiento', '')
            }
        
        return {
            "modelos_disponibles": list(self.trained_models.keys()),
            "modelos_info": modelos_info,
            "total_modelos": len(self.trained_models),
            "modelos_entrenados": self.modelos_entrenados
        }