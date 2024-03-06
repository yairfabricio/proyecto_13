# %% [markdown]
# # Telecomunicaciones: identificar operadores ineficaces

# %% [markdown]
# ## Abrir el archivo de datos y leer la información general

# %%
#importar librerias
import pandas as pd 

# %%
#leer el dataset 1
dt=pd.read_csv('files/datasets/input/telecom_dataset_us.csv')
print(dt)
print(dt.info())

# %%
#leer el dataset de los clientes
clients=pd.read_csv('files/datasets/input/telecom_clients_us.csv')
print(clients)
print(clients.info())

# %% [markdown]
# ## Lleva a cabo el análisis exploratorio de datos

# %% [markdown]
# ### Convierte los datos en los tipos necesarios.

# %%
#convertir date a formato fecha del dataset dt
dt['date']=pd.to_datetime(dt['date'])
#convertir columnas a tipo category
dt['direction']= dt['direction'].astype('category')
#convertir date a formato fecha del dataset clients
clients['date_start']=pd.to_datetime(clients['date_start'])
#convertir columnas a tipo category
clients['tariff_plan']=clients['tariff_plan'].astype('category')

# %% [markdown]
# ### Encuentra y elimina errores en los datos. Asegúrate de explicar qué errores encontraste y cómo los eliminaste.

# %%
#eliminar las filas nulas del datasets dt de la fila operator_id
dt.dropna(inplace=True)


dt.to_csv("files/datasets/intermediate/a01_dataset_sin_na.csv")
clients.to_csv("files/datasets/intermediate/a01_clients_sin_na.csv")