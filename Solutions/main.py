import pandas as pd
import numpy as np
from datetime import timedelta
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, mean_absolute_error
from xgboost import XGBClassifier, XGBRegressor
import joblib
import sys

def cargar_y_limpiar_clientes(path):
    df = pd.read_csv(path)
    print("Original size (Clients):", len(df))
    
    df['genero'] = df['genero'].replace(' ', np.nan)
    invalid_genres = df[df['genero'].isna()]['id'].tolist()
    df = df.dropna(subset=['genero'])

    df['fecha_nacimiento'] = pd.to_datetime(df['fecha_nacimiento'], errors='coerce')
    invalid_births = df[df['fecha_nacimiento'] < pd.Timestamp('1900-01-01')]['id'].tolist()
    df = df[df['fecha_nacimiento'] >= pd.Timestamp('1900-01-01')]

    del_ids = set(invalid_genres + invalid_births)
    print("Final size (Clients):", len(df))
    print("IDs eliminados (Clients):", del_ids)
    return df, del_ids

def cargar_y_limpiar_transacciones(path, del_ids):
    df = pd.read_csv(path)
    print("Original size (Transactions):", len(df))

    df = df[~df['id'].isin(del_ids)]
    df['giro_comercio'] = df['giro_comercio'].replace('4121', 'SERVICIOS DE TRANSPORTE POR AUTOMÓVIL')
    df = df.dropna()

    replace_map = {
        '7ELEVEN': '7 ELEVEN', 'TOTAL PLAY': 'TOTALPLAY',
        'APLAZ': 'APLAZO', 'DIDIFOOD': 'DIDI FOOD',
        'MERCADOPAGO': 'MERCADO PAGO', 'SMARTFIT': 'SMART FIT',
        'WAL-MART': 'WALMART', 'MI ATT': 'AT&T', 'ATT': 'AT&T'
    }
    df['comercio'] = df['comercio'].replace(replace_map)
    print("Cleaned size (Transactions):", len(df))
    return df

def construir_dataset(merged_df):
    merged_df['fecha'] = pd.to_datetime(merged_df['fecha'], errors='coerce')
    freq = merged_df.groupby(['id', 'comercio']).size().reset_index(name='n_tx')
    rec_pairs = freq[freq['n_tx'] >= 3][['id','comercio']]
    tx_rec = merged_df.merge(rec_pairs, on=['id','comercio'], how='inner')

    rows = []
    for (cli, com), grp in tx_rec.groupby(['id','comercio']):
        grp = grp.sort_values('fecha').reset_index(drop=True)
        grp['days_diff'] = grp['fecha'].diff().dt.days
        for i in range(2, len(grp) - 1):
            curr = grp.loc[i]
            prev = grp.loc[:i-1]
            nxt  = grp.loc[i+1]
            delta = (nxt['fecha'] - curr['fecha']).days
            rows.append({
                'días_desde_última': curr['days_diff'],
                'monto_actual': curr['monto'],
                'promedio_montos_previos': prev['monto'].mean(),
                'periodicidad_media': prev['days_diff'].mean(),
                'trans_previas': i,
                'tuvo_próxima_en_30d': int(delta <= 30),
                'días_hasta_próxima': delta,
                'monto_próximo': nxt['monto']
            })
    df = pd.DataFrame(rows)

    # 🧼 Filtrar valores extremos
    df = df[
        (df['días_hasta_próxima'] > 0) & 
        (df['días_hasta_próxima'] <= 60) &
        (df['monto_próximo'] > 0) & 
        (df['monto_próximo'] <= 10000)
    ]
    
    print("Training examples:", df.shape)
    return df

def entrenar_y_guardar_modelos(model_df):
    features = ['días_desde_última', 'monto_actual', 'promedio_montos_previos', 'periodicidad_media', 'trans_previas']
    X = model_df[features]
    y = model_df['tuvo_próxima_en_30d']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    clf = XGBClassifier(n_estimators=100, max_depth=4, random_state=42, use_label_encoder=False, eval_metric='logloss')
    clf.fit(X_train, y_train)
    y_proba = clf.predict_proba(X_test)[:,1]
    print("ROC-AUC Clasificador:", roc_auc_score(y_test, y_proba))

    # Regresión solo para positivos
    pos_mask = y_train == 1
    X_reg = X_train[pos_mask]
    y_days = model_df.loc[X_reg.index, 'días_hasta_próxima']
    y_amt  = model_df.loc[X_reg.index, 'monto_próximo']

    # ➕ Transformaciones logarítmicas
    y_days_log = np.log1p(y_days)
    y_amt_log = np.log1p(y_amt)

    reg_days = XGBRegressor(
        n_estimators=200,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.9,
        colsample_bytree=0.9,
        random_state=42
    )
    reg_days.fit(X_reg, y_days_log)

    reg_amt = XGBRegressor(
        n_estimators=200,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42
    )
    reg_amt.fit(X_reg, y_amt_log)

    # Evaluación
    test_mask = y_proba >= 0.5
    X_reg_test = X_test[test_mask]
    y_days_true = model_df.loc[X_reg_test.index, 'días_hasta_próxima']
    y_amt_true  = model_df.loc[X_reg_test.index, 'monto_próximo']

    # ➕ Invertimos la transformación logarítmica
    y_days_pred = np.expm1(reg_days.predict(X_reg_test))
    y_amt_pred  = np.expm1(reg_amt.predict(X_reg_test))

    print("MAE días:", mean_absolute_error(y_days_true, y_days_pred))
    print("MAE monto:", mean_absolute_error(y_amt_true, y_amt_pred))

    joblib.dump(clf, 'model_clf.xgb')
    joblib.dump(reg_days, 'model_reg_days.xgb')
    joblib.dump(reg_amt, 'model_reg_amt.xgb')
    print("Modelos guardados exitosamente.")

def main():
    try:
        df_clients, del_ids = cargar_y_limpiar_clientes('db_clients.csv')
        df_transactions = cargar_y_limpiar_transacciones('db_transactions.csv', del_ids)
        merged_df = df_transactions.merge(df_clients, on='id', how='inner')
        print("Merged size:", len(merged_df))

        model_df = construir_dataset(merged_df)
        if model_df.empty:
            print("No hay suficientes datos para entrenar.")
            sys.exit(0)

        entrenar_y_guardar_modelos(model_df)

    except Exception as e:
        print("Ocurrió un error:", str(e))
        sys.exit(1)

if __name__ == '__main__':
    main()
