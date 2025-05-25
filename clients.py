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
