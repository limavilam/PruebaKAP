import pandas as pd 
import numpy as np
import missingno as msno
import matplotlib.pyplot as plt
import seaborn as sns
import os

usuarios_url = os.path.expanduser("~/Downloads/PruebaTecnica/Usuarios.csv")
usuarios_df = pd.read_csv(usuarios_url)

"""
Extract:  se realiza la lectura de los datos desde su origen (en este caso, un archivo CSV) y se obtiene una descripción general del conjunto de datos,
incluyendo el número de columnas, nombres de las variables, cantidad de valores nulos y tipos de datos. Posteriormente, se analiza la presencia de datos 
faltantes en cada columna y se realizan ajustes necesarios, como la corrección de nombres de columnas y la conversión de tipos de datos en caso que lo requiera. 
Además, dado que el dataset contiene variables numéricas, se extrae información estadística descriptiva, lo que permite una mejor comprensión de la estructura 
de los datos antes de proceder con su transformación.
"""

#La columna de ingreso_mensual aparece con ;;;, se van a reemplazar. 
usuarios_df.rename(columns={"ingreso_mensual;;;": "ingreso_mensual"}, inplace=True)
print(usuarios_df.head())
usuarios_df.info()

usuarios_df['ingreso_mensual'] = usuarios_df['ingreso_mensual'].astype(str).str.replace(';;;', '', regex=False)

# Conversión de tipo de datos
usuarios_df['ingreso_mensual'] = pd.to_numeric(usuarios_df['ingreso_mensual'], errors='coerce')
print ("Cambio de tipo de dato")
print(usuarios_df.info())

#Obtención de estadísticas descriptivas 
print(usuarios_df.describe(include='all'))

#Obteniendo sumatoria de valores nulos
print ("Sumatoria de valores nulos")
print(usuarios_df.isnull().sum())

#Visualización de datos faltantes de las columnas (edad, ingreso_mensual,pais y nombre)
#msno.matrix(usuarios_df)
#plt.show()

"""
Transform: Ese normalizan los nombres de los países convirtiéndolos a mayúsculas para evitar inconsistencias, se identifican y tratan valores atípicos 
en ingreso_mensual utilizando el Rango Intercuartílico (IQR). Posteriorme, se imputan los valores nulos en variables numéricas, como edad e ingreso_mensual,
utilizando la media, mientras que en variables categóricas, la columna país se rellena con la moda y la columna nombre con "Sin Nombre". 
Finalmente, se valida mediante la sumatoria y visualización que no queden valores nulos. 
"""

#Normalizar los nombres de los paises en mayúsculas
usuarios_df['pais'] = usuarios_df['pais'].str.upper()
print(usuarios_df['pais'])

# Determinar valores atípicos con IQR
q1 = usuarios_df['ingreso_mensual'].quantile(0.25)
q3 = usuarios_df['ingreso_mensual'].quantile(0.75)
iqr = q3 - q1
lim_inf = q1 - 1.5 * iqr
lim_sup = q3 + 1.5 * iqr
print(f"Límite inferior: {lim_inf:.2f}")
print(f"Límite superior: {lim_sup:.2f}")

# Visualización de valores atípicos
plt.figure(figsize=(8, 6))
sns.boxplot(data=usuarios_df, y='ingreso_mensual', width=0.3) 
plt.axhline(y=lim_inf, color='red', linestyle='--', label='Límite Inferior')
plt.axhline(y=lim_sup, color='blue', linestyle='--', label='Límite Superior')
plt.legend()
plt.title('Boxplot de Ingreso Mensual con Límites de Outliers')
plt.ylabel('Ingreso Mensual')
plt.show()

usuarios_mean_edad = round(usuarios_df['edad'].mean(),1)
usuarios_df['edad'].infer_objects(copy=False).fillna(usuarios_mean_edad, inplace=True)

usuarios_mean_salario = round(usuarios_df['ingreso_mensual'].mean(),2)
usuarios_df['ingreso_mensual'].infer_objects(copy=False).fillna(usuarios_mean_salario, inplace=True)

#Visualización de datos completos de edad, ingreso mensual (Variables Numéricas)
#msno.matrix(usuarios_df)
#plt.show()

#Rellenando la columna de pais
usuarios_df['pais'] = usuarios_df['pais'].fillna(usuarios_df['pais'].mode()[0])

#Rellenando la columna de nombre
usuarios_df['nombre'] = usuarios_df['nombre'].fillna("Sin Nombre")

#Obteniendo sumatoria de valores nulos
print ("Sumatoria de valores nulos después de la transformación")
print(usuarios_df.isnull().sum())

#Visualización de datos completos de variables numéricas y categóricas
msno.matrix(usuarios_df)
plt.show()

#Visualización del dataset transformado
print("visualización de dataset transformado")
print(usuarios_df.head())


"""
Load: En esta fase se almacena la información transformada en un nuevo archivo CSV, asegurando que los datos estén limpios y estructurados para su 
posterior uso en modelos de Machine Learning o análisis
"""

usuarios_df.to_csv("usuarios_limpios.csv", index=False)
print("Datos limpios guardados correctamente.")


