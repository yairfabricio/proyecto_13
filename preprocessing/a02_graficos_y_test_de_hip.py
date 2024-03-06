
import pandas as pd 
import seaborn as sns
import matplotlib.pyplot as plt
from statsmodels.stats.proportion import proportions_ztest
from scipy import stats as st
import numpy as np
import pickle


# %% [markdown]
# ### Une los dos datasets

dt = pd.read_csv("files/datasets/intermediate/a01_dataset_sin_na.csv") 
clients = pd.read_csv("files/datasets/intermediate/a01_clients_sin_na.csv")

# %%
df= dt.merge(clients,on='user_id')
df

# %% [markdown]
# ## Identificar operadores ineficaces

# %% [markdown]
# se considera operador ineficaz si cumple las siguientes tres caracteristicas

# %% [markdown]
# ### Se considera que un operador es ineficaz si tiene una gran cantidad de llamadas entrantes perdidas (internas y externas)

# %% [markdown]
# #### contar el numero de llamadas perdidas por operador 

# %%
loss_call=df.groupby('operator_id')['is_missed_call'].sum().reset_index()
loss_call

# %% [markdown]
# #### considerar las llamadas de duracion 0 como llamadas perdidas

# %%
#identificar las llamadas con duracion 0
loss_cero=df[df['total_call_duration']==0].groupby('operator_id').count()
loss_call_cero_duration=loss_cero['is_missed_call'].reset_index()
#unir las llamadas de duracion cero a las llamadas perdidas
loss_call= pd.concat([loss_call,loss_call_cero_duration],axis=0)
loss_call

# %% [markdown]
# #### realizar un histograma para el numero de llamadas perdidas por operador y halla el promedio

# %%
sns.histplot(data=loss_call['is_missed_call'])

# %%
loss_call.describe()

# %% [markdown]
# #### almacenar los operadores que superan la media de llamadas perdidas por operador

# %%
operator_with_missed_call=loss_call[loss_call['is_missed_call']>loss_call['is_missed_call'].mean()]
operator_with_missed_call

# %% [markdown]
# se obtuvo 358 operadores que superaron el umbral de perdidas de llamadas

# %% [markdown]
# ### Se considera que un operador es ineficaz si tiene un tiempo de espera prolongado para las llamadas entrantes

# %% [markdown]
# #### restar call_duration a total_call_duration para hallar el tiempo de espera

# %%
wait_call=df.groupby('operator_id')['total_call_duration'].sum()-df.groupby('operator_id')['call_duration'].sum()
wait_call=wait_call.reset_index()
wait_call

# %% [markdown]
# #### realizar un histograma para el tiempo de espera y halla el promedio

# %%
x_values = pd.Series(range(0, len(wait_call[0])))
plt.scatter(x_values, wait_call[0])

# %%
#filtrar los datos anomalos en wait_call
print(np.percentile(wait_call[0], [80, 95, 99]))
abnormal_users=wait_call[wait_call[0]>14141]['operator_id']
# excluir los datos anomalos del grafico
wait_call_filtered=wait_call[~wait_call['operator_id'].isin(abnormal_users)]
sns.histplot(data=wait_call_filtered[0])


# %%
wait_call_filtered[0].describe()

# %% [markdown]
# #### almacenar los operadores que superan el promedio de tiempo de espera

# %%
operator_with_wait_call=wait_call[wait_call[0]>wait_call[0].mean()]
operator_with_wait_call
oprator_with_wait_call_filtered=wait_call_filtered[wait_call_filtered[0]>wait_call_filtered[0].mean()]

# %% [markdown]
# se observa 230 operadores que superaron el umbral de tiempo de espera

# %% [markdown]
# ### Se considera que un operador es ineficaz si tiene un numero de llamadas salientes reducido

# %% [markdown]
# #### contar el numero de llamadas out para cada operador

# %%
call_out=df[df['direction']=='out'].groupby('operator_id').count()
call_out['direction']

# %% [markdown]
# ####  realizar un histograma para el numero de llamadas out para cada operador y halla el promedio

# %%
sns.histplot(data=call_out['direction'])
print(call_out['direction'].mean())

# %% [markdown]
# #### almacenar a los operadores que tienen un numero de llamadas menor que el promedio

# %%
operator_with_less_out=call_out[call_out['direction']<call_out['direction'].mean()]
operator_with_less_out=operator_with_less_out['direction'].reset_index()
operator_with_less_out

# %% [markdown]
# hay 556 operadores con deficiencia de llamadas salientes

# %% [markdown]
# ### identificar a los operadores que cumplen con las tres condiciones anteriores y almacenarlos en la variable ineficaz

# %%
conjunto1 = set(operator_with_missed_call['operator_id'])
conjunto2 = set(operator_with_wait_call['operator_id'])
conjunto3 = set(operator_with_less_out['operator_id'])

# Encontrar la intersección de los conjuntos
ineficaz = list(conjunto1.intersection(conjunto2, conjunto3))

# Ahora, ineficaz contiene los elementos presentes en las tres series
print(f'Numero de operadores ineficientes:{len(ineficaz)}')

# %% [markdown]
# ## Prueba las hipótesis estadísticas

# %% [markdown]
# ### Hay diferencia significativa entre el tamaño de las muestras de los operadores ineficaces en las tres tarifas actuales de la clientela

# %%
# filtrar los operadores ineficaces en el df para la tarifa a 
operator_ineficiente_a=df[df['tariff_plan']=='A']['operator_id'].isin(ineficaz).astype(int)
operator_ineficiente_a
#filtrar los operadors ineficientes en el df para la tarifa b
operator_ineficiente_b=df[df['tariff_plan']=='B']['operator_id'].isin(ineficaz).astype(int)
#filtrar los operadors ineficientes en el df para la tarifa c
operator_ineficiente_c=df[df['tariff_plan']=='C']['operator_id'].isin(ineficaz).astype(int)
#funcion para hacer las pruebas z
def realizar_prueba_z_e_imprimir_resultado(grupo_1, grupo_2, alpha=0.05):
    # Realizar la prueba de diferencia de proporciones:
    z_stat, p_value = proportions_ztest([grupo_1.sum(), grupo_2.sum()], [len(grupo_1), len(grupo_2)])


    if p_value < alpha:
        print("Hay una diferencia estadísticamente significativa.")
    else:
        print("No hay evidencia suficiente para afirmar que hay una diferencia estadísticamente significativa.")
#realizar la primera prueba
realizar_prueba_z_e_imprimir_resultado(operator_ineficiente_a,operator_ineficiente_b)
#realizar la segunda prueba
realizar_prueba_z_e_imprimir_resultado(operator_ineficiente_b,operator_ineficiente_c)
#realizar la tercera prueba
realizar_prueba_z_e_imprimir_resultado(operator_ineficiente_c,operator_ineficiente_a)

# %% [markdown]
# - H0: las proporciones de operadores ineficentes entre los grupos de las tarifas son iguales
# - H1: las proporciones de operadores ineficientes entre los grupos son diferentes

# %% [markdown]
# En este caso rechazamos la hipotesis nula ya que en los tres tipos de tarifa las proporciones de los operadores ineficientes tuvieron una diferencia estadísticamente significativa.

# %% [markdown]
# ### Las llamadas entrantes perdidas promedio de los operadores eficaces y los operadores ineficaces son diferentes

# %% [markdown]
# - H0: el promedio de las llamadas entrantes promedio de los operadores eficaces y los operadores ineficaces son iguales
# - H1: el promedio de las llamadas entrantes promedio de los operadores eficaces y los operadores ineficaces son diferentes

# %%
missed_call_operator_eficaz=df[~df['operator_id'].isin(ineficaz)]['is_missed_call'].astype(int)
missed_call_operator_ineficaz=df[df['operator_id'].isin(ineficaz)]['is_missed_call'].astype(int)
alpha = 0.05
results = st.mannwhitneyu(missed_call_operator_eficaz,missed_call_operator_ineficaz)
print('valor p:',results.pvalue)
if results.pvalue < alpha:
    print("Rechazamos la hipótesis nula: la diferencia es estadísticamente significativa")
else:
    print("No podemos rechazar la hipótesis nula:  no podemos sacar conclusiones sobre la diferencia")

# %% [markdown]
# ### el tiempo de espera de los operadores eficaces y los ineficaces son diferentes

# %% [markdown]
# - H0: el tiempo de espera entre los operadores eficaces y los ineficaces es igual
# - H1: el tiempo de espera entre los operadores eficaces y los ineficaces son diferentes

# %%
wait_time_eficaz=wait_call[~wait_call['operator_id'].isin(ineficaz)][0]
wait_time_ineficaz=wait_call[wait_call['operator_id'].isin(ineficaz)][0]
alpha = 0.05
results = st.mannwhitneyu(wait_time_eficaz,wait_time_ineficaz)
print('valor p:',results.pvalue)
if results.pvalue < alpha:
    print("Rechazamos la hipótesis nula :la diferencia es estadísticamente significativa")
else:
    print("No podemos rechazar la hipótesis nula:  no podemos sacar conclusiones sobre la diferencia")

# %% [markdown]
# #### el tiempo de espera de los operadores eficaces y los ineficaces son diferentes utilizando los datos filtrados

# %% [markdown]
# - H0: el tiempo de espera entre los operadores eficaces y los ineficaces es igual
# - H1: el tiempo de espera entre los operadores eficaces y los ineficaces son diferentes

# %%
wait_time_eficaz_filtered=wait_call_filtered[~wait_call_filtered['operator_id'].isin(ineficaz)][0]
wait_time_ineficaz_filtered=wait_call_filtered[wait_call_filtered['operator_id'].isin(ineficaz)][0]
alpha = 0.05
results = st.mannwhitneyu(wait_time_eficaz_filtered,wait_time_ineficaz_filtered)
print('valor p:',results.pvalue)
if results.pvalue < alpha:
    print("Rechazamos la hipótesis nula :la diferencia es estadísticamente significativa")
else:
    print("No podemos rechazar la hipótesis nula:  no podemos sacar conclusiones sobre la diferencia")

# %% [markdown]
# ### el numero de llamadas salientes de los operadares eficaces y los ineficaces son diferentes

# %% [markdown]
# - H0: el numero de llamadas salientes de los operadores eficaces e ineficaces son los mismos
# - H1: el numero de llamadas salientes de los operadores eficaces e ineficaces son diferentes

# %%
less_out_eficaz=operator_with_less_out[~operator_with_less_out['operator_id'].isin(ineficaz)]['direction']
less_out_ineficaz=operator_with_less_out[operator_with_less_out['operator_id'].isin(ineficaz)]['direction']
alpha = 0.05
results = st.mannwhitneyu(less_out_eficaz,less_out_ineficaz)
print('valor p:',results.pvalue)
if results.pvalue < alpha:
    print("Rechazamos la hipótesis nula:la diferencia es estadísticamente significativa")
else:
    print("No podemos rechazar la hipótesis nula:  no podemos sacar conclusiones sobre la diferencia")


results.to_csv("files/datasets/output/a02_clients_sin_na.csv")

results

# %% [markdown]
# ## Escribe una conclusión general

# %% [markdown]
# - Hay una diferencia entre los operadores ineficaces con respecto a las muestras tomadas de cada tarifa de la clientela, por lo que habria que analizar en cada especifico como es el comportamiento de ciertos operadores en cada plan para determinar una posible causa
# - De las tres pruebas estidisticas hechas acerca de las condiciones tiene que cumplir un operador para que se le considere ineficaz comparado con un operador eficaz, en todas incluida la prueba filtrada con los datos atipicos en el caso de tiempo de espera prolongado dieron que ambos grupos tienen una diferencia significativa hablando de las medias por lo tanto podemos concluir que confirmamos los criterios que se toman en cuenta para identificar los operadores ineficaces.

# %% [markdown]
# 

pickle.dump(results, "files/datasets/output/a02_results_hip_test.pkl")


