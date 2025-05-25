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

print("Final size (Transactions):", len(df_transactions))
