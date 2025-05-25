import pandas as pd
import numpy as np
import joblib
from datetime import datetime

FEATURES = [
    'días_desde_última',
    'monto_actual',
    'promedio_montos_previos',
    'periodicidad_media',
    'trans_previas'
]

def cargar_modelos():
    clf = joblib.load('model_clf.xgb')
    reg_days = joblib.load('model_reg_days.xgb')
    reg_amt = joblib.load('model_reg_amt.xgb')
    return clf, reg_days, reg_amt

def construir_features_de_prediccion(df_transacciones, cliente_id, comercio):
    df_transacciones['fecha'] = pd.to_datetime(df_transacciones['fecha'], errors='coerce')
    historial = df_transacciones[
        (df_transacciones['id'] == cliente_id) &
        (df_transacciones['comercio'] == comercio)
    ].sort_values('fecha')

    if len(historial) < 3:
        raise ValueError("No hay suficientes transacciones previas para predecir (mínimo 3)")

    historial = historial.reset_index(drop=True)
    historial['days_diff'] = historial['fecha'].diff().dt.days

    actual = historial.iloc[-1]
    prev = historial.iloc[:-1]

    features = {
        'días_desde_última': actual['days_diff'],
        'monto_actual': actual['monto'],
        'promedio_montos_previos': prev['monto'].mean(),
        'periodicidad_media': prev['days_diff'].mean(),
        'trans_previas': len(prev)
    }

    return pd.DataFrame([features])

def predecir_recurrencia(df_features, clf_model):
    prob = clf_model.predict_proba(df_features)[:, 1][0]
    return prob

def predecir_días_y_monto(df_features, reg_days, reg_amt):
    dias_pred = np.expm1(reg_days.predict(df_features)[0])
    monto_pred = np.expm1(reg_amt.predict(df_features)[0])
    return dias_pred, monto_pred

def predecir_transaccion(cliente_id, comercio, df_transacciones):
    try:
        clf, reg_days, reg_amt = cargar_modelos()
        features = construir_features_de_prediccion(df_transacciones, cliente_id, comercio)

        prob_recurrencia = predecir_recurrencia(features, clf)
        resultado = {
            'probabilidad_de_recompra_30d': round(prob_recurrencia, 4)
        }

        if prob_recurrencia >= 0.5:
            dias, monto = predecir_días_y_monto(features, reg_days, reg_amt)
            resultado.update({
                'dias_hasta_recompra_estimados': round(dias, 2),
                'monto_estimado_proxima_compra': round(monto, 2)
            })
        else:
            resultado.update({
                'mensaje': 'Baja probabilidad de recompra en los próximos 30 días.'
            })

        return resultado

    except Exception as e:
        return {'error': str(e)}
