import pandas as pd

import numpy as np

import joblib

from datetime import timedelta



# Cargar modelos

clf = joblib.load("model_clf.xgb")

reg_days = joblib.load("model_reg_days.xgb")

reg_amt = joblib.load("model_reg_amt.xgb")



# Cargar datos

df_clients = pd.read_csv("db_clients.csv")

df_transactions = pd.read_csv("db_transactions.csv")



# Preprocesamiento básico

df_clients['fecha_nacimiento'] = pd.to_datetime(df_clients['fecha_nacimiento'], errors='coerce')

df_transactions['fecha'] = pd.to_datetime(df_transactions['fecha'], errors='coerce')



def predict_next_purchase(client_id):

    user_tx = df_transactions[df_transactions['id'] == client_id].copy()

    if user_tx.shape[0] < 3:

        return {"error": "No hay suficientes transacciones para este cliente."}



    user_tx = user_tx.sort_values("fecha").reset_index(drop=True)

    user_tx["days_diff"] = user_tx["fecha"].diff().dt.days



    # Última transacción

    last_tx = user_tx.iloc[-1]

    prev_tx = user_tx.iloc[:-1]



    # Features

    días_desde_última = user_tx.iloc[-1]["days_diff"]

    monto_actual = last_tx["monto"]

    promedio_montos_previos = prev_tx["monto"].mean()

    periodicidad_media = prev_tx["days_diff"].mean()

    trans_previas = len(prev_tx)



    X_pred = pd.DataFrame([{

        "días_desde_última": días_desde_última,

        "monto_actual": monto_actual,

        "promedio_montos_previos": promedio_montos_previos,

        "periodicidad_media": periodicidad_media,

        "trans_previas": trans_previas

    }])



    # Clasificación

    prob = clf.predict_proba(X_pred)[0, 1]

    if prob < 0.5:

        return {"proxima_compra": False, "probabilidad": prob}



    # Regresión

    dias_estimados = reg_days.predict(X_pred)[0]

    monto_estimado = reg_amt.predict(X_pred)[0]

    fecha_estimado = last_tx["fecha"] + timedelta(days=int(dias_estimados))



    comercio_estimado = user_tx["comercio"].mode().iloc[0] if not user_tx["comercio"].mode().empty else "Desconocido"



    return {

        "proxima_compra": True,

        "probabilidad": prob,

        "fecha_estimada": fecha_estimado.date().isoformat(),

        "monto_estimado": round(monto_estimado, 2),

        "comercio_estimado": comercio_estimado

    }
