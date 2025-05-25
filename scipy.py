import pandas as pd
import numpy as np

df_clients = pd.read_csv('db/db_clients.csv')
print("Original size (Clients):", len(df_clients))

df_clients['genero'] = df_clients['genero'].replace(' ', np.nan)

invalid_genres = df_clients[df_clients['genero'].isna()]['id'].tolist()

# clean invalid genre
df_clients = df_clients.dropna(subset=['genero'])

df_clients['fecha_nacimiento'] = pd.to_datetime(df_clients['fecha_nacimiento'], errors='coerce')

# register invalid births (< 1900-01-01)
invalid_births = df_clients[df_clients['fecha_nacimiento'] < pd.Timestamp('1900-01-01')]['id'].tolist()

# clean invalid births
df_clients = df_clients[df_clients['fecha_nacimiento'] >= pd.Timestamp('1900-01-01')]

print("Final size (Clients):", len(df_clients))

del_clients = invalid_genres + invalid_births
print("IDs eliminados:", del_clients)

df_transactions = pd.read_csv('db/db_transactions.csv')
print("Original size (Transactions):", len(df_transactions))

# purge invalid clients' transactions
df_transactions = df_transactions[~df_transactions['id'].isin(del_clients)]

# print("Final size (Transactions):", len(df_transactions))

df_transactions['giro_comercio'] = df_transactions['giro_comercio'].replace('4121', 'SERVICIOS DE TRANSPORTE POR AUTOMÓVIL')

df_transactions = df_transactions.dropna()

df_transactions['comercio'] = df_transactions['comercio'].replace('7ELEVEN', '7 ELEVEN')
df_transactions['comercio'] = df_transactions['comercio'].replace('TOTAL PLAY', 'TOTALPLAY')
df_transactions['comercio'] = df_transactions['comercio'].replace('APLAZ', 'APLAZO')
df_transactions['comercio'] = df_transactions['comercio'].replace('DIDIFOOD', 'DIDI FOOD')
df_transactions['comercio'] = df_transactions['comercio'].replace('MERCADOPAGO', 'MERCADO PAGO')
df_transactions['comercio'] = df_transactions['comercio'].replace('SMARTFIT', 'SMART FIT')
df_transactions['comercio'] = df_transactions['comercio'].replace('WAL-MART', 'WALMART')
df_transactions['comercio'] = df_transactions['comercio'].replace({
  'MI ATT': 'AT&T',
  'ATT': 'AT&T'
})

merged_df = df_transactions.merge(df_clients, on='id', how='right')
print("Final size (Transactions):", len(merged_df))
print(merged_df.columns)

#### SHIT STARTS HERE
merged_df['fecha'] = pd.to_datetime(merged_df['fecha'], errors='coerce')

# Agrupamos por cliente y comercio
grouped = merged_df.groupby(['id', 'comercio'])

# Función para identificar patrones periódicos
def get_transaction_pattern(group):
    group = group.sort_values('fecha')
    fechas = group['fecha']
    
    if len(fechas) < 3:
        return None  # muy pocos datos para detectar periodicidad

    # Calculamos diferencias entre fechas consecutivas
    diffs = fechas.diff().dt.days.dropna()

    # Si no hay al menos dos diferencias, no se puede calcular nada útil
    if len(diffs) < 2:
        return None

    std_diff = diffs.std()
    mean_diff = diffs.mean()

    # Heurística: baja desviación estándar sugiere periodicidad clara
    if std_diff < 3 and mean_diff > 5:
        return {
            'id': group['id'].iloc[0],
            'comercio': group['comercio'].iloc[0],
            'frecuencia_dias': round(mean_diff),
            'desviacion_dias': round(std_diff, 2),
            'monto_promedio': round(group['monto'].mean(), 2),
            'ultima_fecha': fechas.max()
        }

    return None

# Analizar todos los grupos
patterns = []
for _, group in grouped:
    pattern = get_transaction_pattern(group)
    if pattern:
        patterns.append(pattern)

# Convertimos los resultados a DataFrame
df_patterns = pd.DataFrame(patterns)

# Calcular la próxima fecha estimada de gasto
df_patterns['proxima_fecha_estimada'] = df_patterns['ultima_fecha'] + pd.to_timedelta(df_patterns['frecuencia_dias'], unit='D')

# Mostramos los resultados
print("Patrones periódicos detectados:", len(df_patterns))
print(df_patterns.head())

from collections import Counter
from scipy.fft import fft
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, classification_report
from sklearn.preprocessing import LabelEncoder
import warnings
warnings.filterwarnings('ignore')

class PeriodicSpendingPredictor:
    def __init__(self, merged_df):
        self.df = merged_df.copy()
        self.periodic_patterns = {}
        self.amount_model = None
        self.timing_model = None
        self.label_encoders = {}
        
    def prepare_data(self):
        """Prepara los datos identificando patrones periódicos"""
        print("Preparando datos y identificando patrones periódicos...")
        
        # Convertir fecha a datetime
        self.df['fecha'] = pd.to_datetime(self.df['fecha'])
        
        # Crear características temporales
        self.df['dia_semana'] = self.df['fecha'].dt.dayofweek
        self.df['dia_mes'] = self.df['fecha'].dt.day
        self.df['mes'] = self.df['fecha'].dt.month
        self.df['semana_año'] = self.df['fecha'].dt.isocalendar().week
        
        # Identificar gastos periódicos por cliente y comercio
        self._identify_periodic_patterns()
        
        return self.df
    
    def _identify_periodic_patterns(self):
        """Identifica patrones de gastos periódicos"""
        print("Identificando patrones periódicos...")
        
        periodic_transactions = []
        
        # Agrupar por cliente y comercio para encontrar patrones
        for (client_id, comercio), group in self.df.groupby(['id', 'comercio']):
            if len(group) < 3:  # Necesitamos al menos 3 transacciones
                continue
                
            group_sorted = group.sort_values('fecha')
            dates = group_sorted['fecha'].values
            amounts = group_sorted['monto'].values
            
            # Calcular diferencias entre fechas consecutivas
            date_diffs = []
            for i in range(1, len(dates)):
                diff = (pd.to_datetime(dates[i]) - pd.to_datetime(dates[i-1])).days
                date_diffs.append(diff)
            
            if not date_diffs:
                continue
                
            # Identificar periodicidad
            periodicity = self._detect_periodicity(date_diffs)
            
            if periodicity['is_periodic']:
                # Marcar transacciones como periódicas
                pattern_key = f"{client_id}_{comercio}"
                self.periodic_patterns[pattern_key] = {
                    'client_id': client_id,
                    'comercio': comercio,
                    'period_days': periodicity['period'],
                    'avg_amount': np.mean(amounts),
                    'std_amount': np.std(amounts),
                    'last_transaction': dates[-1],
                    'transaction_count': len(group),
                    'confidence': periodicity['confidence']
                }
                
                # Agregar flag de periodicidad
                mask = (self.df['id'] == client_id) & (self.df['comercio'] == comercio)
                self.df.loc[mask, 'is_periodic'] = True
                self.df.loc[mask, 'period_days'] = periodicity['period']
                
                periodic_transactions.extend(group.index.tolist())
        
        # Marcar todas las transacciones no periódicas
        self.df['is_periodic'] = self.df['is_periodic'].fillna(False)
        self.df['period_days'] = self.df['period_days'].fillna(0)
        
        print(f"Encontrados {len(self.periodic_patterns)} patrones periódicos únicos")
        print(f"Total de transacciones periódicas: {len(periodic_transactions)}")
        
    def _detect_periodicity(self, date_diffs, tolerance=0.3):
        """Detecta si existe periodicidad en las diferencias de fechas"""
        if len(date_diffs) < 2:
            return {'is_periodic': False, 'period': 0, 'confidence': 0}
        
        # Períodos comunes a detectar (en días)
        common_periods = [7, 14, 15, 30, 60, 90, 365]  # semanal, quincenal, mensual, etc.
        
        best_period = None
        best_confidence = 0
        
        for period in common_periods:
            # Calcular qué tan cerca están las diferencias del período esperado
            deviations = [abs(diff - period) / period for diff in date_diffs]
            avg_deviation = np.mean(deviations)
            
            # Confidence basado en qué tan consistente es el patrón
            confidence = max(0, 1 - avg_deviation)
            
            if confidence > best_confidence and confidence > 0.6:  # Mínimo 60% de confianza
                best_confidence = confidence
                best_period = period
        
        return {
            'is_periodic': best_period is not None,
            'period': best_period or 0,
            'confidence': best_confidence
        }
    
    def create_features(self):
        """Crea características para el modelo"""
        print("Creando características para el modelo...")
        
        features_df = self.df[self.df['is_periodic'] == True].copy()
        
        if len(features_df) == 0:
            print("No se encontraron transacciones periódicas suficientes")
            return None
        
        # Encoding de variables categóricas
        categorical_cols = ['comercio', 'giro_comercio', 'tipo_venta', 'genero', 'tipo_persona']
        
        for col in categorical_cols:
            if col in features_df.columns:
                le = LabelEncoder()
                features_df[col + '_encoded'] = le.fit_transform(features_df[col].astype(str))
                self.label_encoders[col] = le
        
        # Características del cliente
        client_features = features_df.groupby('id').agg({
            'monto': ['mean', 'std', 'count'],
            'period_days': 'first'
        }).reset_index()
        
        client_features.columns = ['id', 'avg_spending', 'std_spending', 'transaction_count', 'period_days']
        features_df = features_df.merge(client_features, on='id', suffixes=('', '_client'))
        
        return features_df
    
    def train_models(self):
        """Entrena modelos para predecir montos y timing"""
        print("Entrenando modelos...")
        
        features_df = self.create_features()
        if features_df is None:
            return False
        
        # Características para el modelo
        feature_cols = [
            'dia_semana', 'dia_mes', 'mes', 'semana_año',
            'comercio_encoded', 'giro_comercio_encoded', 'tipo_venta_encoded',
            'genero_encoded', 'tipo_persona_encoded',
            'period_days', 'avg_spending', 'std_spending', 'transaction_count'
        ]
        
        # Filtrar columnas que existen
        available_cols = [col for col in feature_cols if col in features_df.columns]
        X = features_df[available_cols]
        
        # Modelo para predecir montos
        y_amount = features_df['monto']
        X_train, X_test, y_train, y_test = train_test_split(X, y_amount, test_size=0.2, random_state=42)
        
        self.amount_model = RandomForestRegressor(n_estimators=100, random_state=42)
        self.amount_model.fit(X_train, y_train)
        
        # Evaluar modelo de montos
        y_pred = self.amount_model.predict(X_test)
        mae = mean_absolute_error(y_test, y_pred)
        print(f"MAE del modelo de montos: ${mae:.2f}")
        
        # Modelo para predecir próxima compra (días hasta siguiente transacción)
        features_df_sorted = features_df.sort_values(['id', 'comercio', 'fecha'])
        features_df_sorted['days_to_next'] = features_df_sorted.groupby(['id', 'comercio'])['fecha'].diff().dt.days.shift(-1)
        
        # Filtrar datos válidos para timing
        timing_data = features_df_sorted.dropna(subset=['days_to_next'])
        if len(timing_data) > 10:
            X_timing = timing_data[available_cols]
            y_timing = timing_data['days_to_next']
            
            X_train_t, X_test_t, y_train_t, y_test_t = train_test_split(X_timing, y_timing, test_size=0.2, random_state=42)
            
            self.timing_model = RandomForestRegressor(n_estimators=100, random_state=42)
            self.timing_model.fit(X_train_t, y_train_t)
            
            y_pred_t = self.timing_model.predict(X_test_t)
            mae_timing = mean_absolute_error(y_test_t, y_pred_t)
            print(f"MAE del modelo de timing: {mae_timing:.2f} días")
        
        self.feature_cols = available_cols
        return True
    
    def predict_next_purchases(self, client_id=None, days_ahead=30):
        """Predice próximas compras periódicas"""
        predictions = []
        
        patterns_to_predict = self.periodic_patterns
        if client_id:
            patterns_to_predict = {k: v for k, v in self.periodic_patterns.items() 
                                 if v['client_id'] == client_id}
        
        current_date = datetime.now()
        
        for pattern_key, pattern in patterns_to_predict.items():
            last_transaction_date = pd.to_datetime(pattern['last_transaction'])
            period_days = pattern['period_days']
            
            # Calcular próximas fechas probables
            days_since_last = (current_date - last_transaction_date).days
            
            if days_since_last < 0:  # Transacción futura (datos de prueba)
                continue
                
            # Próximo gasto esperado
            cycles_passed = days_since_last // period_days
            next_purchase_days = period_days - (days_since_last % period_days)
            
            if next_purchase_days <= days_ahead:
                next_purchase_date = current_date + timedelta(days=next_purchase_days)
                
                # Predecir monto si hay modelo entrenado
                predicted_amount = pattern['avg_amount']
                confidence = pattern['confidence']
                
                predictions.append({
                    'client_id': pattern['client_id'],
                    'comercio': pattern['comercio'],
                    'predicted_date': next_purchase_date,
                    'predicted_amount': predicted_amount,
                    'confidence': confidence,
                    'period_days': period_days,
                    'days_until': next_purchase_days
                })
        
        return sorted(predictions, key=lambda x: x['predicted_date'])
    
    def get_periodic_summary(self):
        """Resumen de patrones periódicos encontrados"""
        if not self.periodic_patterns:
            return "No se encontraron patrones periódicos"
        
        summary = {
            'total_patterns': len(self.periodic_patterns),
            'by_period': {},
            'by_commerce': {},
            'avg_confidence': np.mean([p['confidence'] for p in self.periodic_patterns.values()])
        }
        
        # Agrupar por período
        for pattern in self.periodic_patterns.values():
            period = pattern['period_days']
            period_name = self._get_period_name(period)
            summary['by_period'][period_name] = summary['by_period'].get(period_name, 0) + 1
        
        # Agrupar por comercio
        for pattern in self.periodic_patterns.values():
            comercio = pattern['comercio']
            summary['by_commerce'][comercio] = summary['by_commerce'].get(comercio, 0) + 1
        
        return summary
    
    def _get_period_name(self, days):
        """Convierte días a nombre de período"""
        if days == 7:
            return "Semanal"
        elif days == 14:
            return "Quincenal"
        elif days == 15:
            return "Quincenal"
        elif days == 30:
            return "Mensual"
        elif days == 60:
            return "Bimensual"
        elif days == 90:
            return "Trimestral"
        elif days == 365:
            return "Anual"
        else:
            return f"{days} días"

# Ejemplo de uso
def run_periodic_analysis(merged_df):
    """Ejecuta el análisis completo de gastos periódicos"""
    print("=== INICIANDO ANÁLISIS DE GASTOS PERIÓDICOS ===\n")
    
    # Crear predictor
    predictor = PeriodicSpendingPredictor(merged_df)
    
    # Preparar datos
    df_prepared = predictor.prepare_data()
    
    # Entrenar modelos
    success = predictor.train_models()
    
    if not success:
        print("No se pudieron entrenar los modelos por falta de datos periódicos")
        return predictor
    
    # Obtener resumen
    summary = predictor.get_periodic_summary()
    print("\n=== RESUMEN DE PATRONES PERIÓDICOS ===")
    print(f"Total de patrones encontrados: {summary['total_patterns']}")
    print(f"Confianza promedio: {summary['avg_confidence']:.2%}")
    
    print("\nDistribución por período:")
    for period, count in summary['by_period'].items():
        print(f"  {period}: {count} patrones")
    
    print("\nTop 10 comercios con más patrones periódicos:")
    top_commerce = sorted(summary['by_commerce'].items(), key=lambda x: x[1], reverse=True)[:10]
    for comercio, count in top_commerce:
        print(f"  {comercio}: {count} patrones")
    
    # Predicciones para los próximos 30 días
    print("\n=== PREDICCIONES PRÓXIMOS 30 DÍAS ===")
    predictions = predictor.predict_next_purchases(days_ahead=30)
    
    if predictions:
        print(f"Se predicen {len(predictions)} compras en los próximos 30 días")
        print("\nPrimeras 10 predicciones:")
        for i, pred in enumerate(predictions[:10]):
            print(f"{i+1}. Cliente {pred['client_id']} - {pred['comercio']}")
            print(f"   Fecha: {pred['predicted_date'].strftime('%Y-%m-%d')}")
            print(f"   Monto: ${pred['predicted_amount']:.2f}")
            print(f"   Confianza: {pred['confidence']:.2%}")
            print(f"   En {pred['days_until']} días\n")
    else:
        print("No hay predicciones para los próximos 30 días")
    
    return predictor


# # Ejecutar el análisis
predictor = run_periodic_analysis(merged_df)
