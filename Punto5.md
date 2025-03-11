#  (Desafío de Integración) Diseña un pipeline de principio a fin (puede ser descrito  forma conceptual o con fragmentos de código) 

## Obtención y carga de datos

Lo primero que debemos hacer es tener claro cual es el conjunto de datos con el que se va a trabajar, una vez identificado se debe validar su tamaño, si el tamaño del dataset es pequeño  (menor a 100 MB) podemos almacenarlo en el mismo respositorio de Github donde estará el codigo fuente, en otro caso podemos considerar otros espacios de almacenamineto como un blob storage, podria ser Amazon S3, Google Cloud Storage o Azure Blob storage, esto debido a su capacidad para almacenar grandes volúmenes de datos de forma segura, con alta durabilidad (99.999999999% según Google y Amazon) y disponibilidad global.

Para este ejemplo vamos a asumir que el dataset es pequeño y que podemos alojarlo en Github, también asumiremos que el dataset es un archivo CSV el cual representa un dataset de un problema de clasificación binaria, para esto simplemente debemos añadir dicho archivo a nuestro repositorio, agregarlo, commitearlo y pushearlo. 

El archivo se alojará en una carpeta dentro del repositorio, esta ruta podría ser: 

```
/data/input/dataset.csv
```


## Fase ETL (Extract, Transform, Load)

La lógica de este pipeline es la siguiente:

1. Extraer los datos de la fuente
2. Transformar los datos
3. Guardar los datos en el destino

Para ello vamos a crear un archivo python que se encargue de realizar esto, este archivo se guardará en la carpeta /src/etl.py

```python
import pandas as pd

# Extraer los datos de la fuente    
df = pd.read_csv('/data/input/dataset.csv')

# Transformar los datos
## Eliminacion de valoes nulos
df = df.dropna()
## Eliminacion de datos duplicados si es necesario

## Determinación de datos incosistentes (pof ejemplo, valores de un rango de fechas)

## Determinación de valores corruptos (por ejemplo, valores nulos en columnas que no deberian tenerlos)

## Selección de caracteristicas

## Discretización de datos o recodificación de datos

# Guardar los datos en el destino
df.to_csv('/data/output/dataset_processed.csv', index=False)
``` 

## Análisis de datos

Una vez que tenemos los datos procesados, podemos proceder a realizar un análisis de los datos, para ello vamos a crear un archivo python que se encargue de realizar esto, este archivo se guardará en la carpeta /src/analysis.py, este archivo se encargara de realizar un análisis de los datos mediante estadisticas descriptiva e inferencial,los resultados del análisis pueden ser guardados en un archivo de texto junto a las graficas generadas, la ruta de este archivo y la carpeta donde se guardarán las graficas será:

```
/data/output/analysis.txt
/data/output/plots/
``` 

Para realizar la fase de ETL yel análisis de datos vamos a utilizar la libreria pandas, numpy, matplotlib y seaborn, para ello vamos a instalar las dependencias necesarias, para esto vamos a crear un archivo requirements.txt y añadir las dependencias necesarias, la ruta de este archivo será:

```
requirements.txt
``` 
el cual contendrá las siguientes dependencias:

```
pandas
numpy
matplotlib
seaborn
```

## Creación y entrenamiento del modelo

Una vez que tenemos los datos procesados y analizados, podemos proceder a crear y entrenar el modelo, para ello vamos a crear un archivo python que se encargue de realizar esto, la ruta de este archivo será:

```
/src/model.py
```

Vamos a utilizar la libreria scikit-learn para crear y entrenar el modelo, para ello debemos agregar las dependencias necesarias al archivo requirements.txt, el cual quedará de la siguiente manera:

```
pandas
numpy
matplotlib
seaborn
scikit-learn
```

Asumamos que el modelo que el modelo que mejor se ajusta a nuestro problema es un modelo de bosques aleatorios, para ello vamos a utilizar la clase RandomForestClassifier de la libreria scikit-learn, el cual quedará de la siguiente manera:

```python
from sklearn.ensemble import RandomForestClassifier 

# Crear el modelo
model = RandomForestClassifier()

# Asumamos que el dataset tiene una columna de etiquetas (y) y el resto de columnas son las caracteristicas (X) 
X = df.drop('y', axis=1)
y = df['y']

# Dividir el dataset en conjuntos de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Entrenar el modelo
model.fit(X_train, y_train) 
``` 

## Evaluación del modelo (Generación de reportes y métricas)

Una vez que el modelo está entrenado, podemos proceder a evaluarlo, para ello se puede utilizar la clase metrics de la libreria scikit-learn, el cual quedará de la siguiente manera:

```python
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Evaluar el modelo
y_pred = model.predict(X_test)

# Generar reportes y métricas
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
auc = roc_auc_score(y_test, y_pred)
confusion_matrix = confusion_matrix(y_test, y_pred)

classification_report = classification_report(y_test, y_pred)

# Guardar los resultados en un archivo de texto
with open('/data/output/metrics.txt', 'w') as f:
    f.write(f"Accuracy: {accuracy}\n")
    f.write(f"Precision: {precision}\n")
    f.write(f"Recall: {recall}\n")
    f.write(f"F1 Score: {f1}\n")
    f.write(f"AUC: {auc}\n")
    f.write(f"Confusion Matrix: {confusion_matrix}\n")
    f.write(f"Classification Report: {classification_report}\n")


 # Finalmente podemos guardar el modelo en un archivo binario para su posterior uso, la ruta de este archivo será:
model.save('/data/output/model.pkl')
```

 ## Creación de un pipeline de CI/CD

Una vez que tenemos todo el proceso listo, podemos proceder a crear un pipeline de CI/CD, para ello vamos a utilizar [Github Actions](https://docs.github.com/en/actions/writing-workflows/quickstart), con Github Actions podemos crear un pipeline de CI/CD que se encargue de ejecutar el proceso de ETL, análisis de datos, entrenamiento del modelo y evaluación del modelo, para ello vamos a crear un archivo .yml que se encargue de realizar esto, la ruta de este archivo será:

```
.github/workflows/pipeline.yml
```


Hasta este momento la estructura de nuestro repositorio sería la siguiente:

```
.
├── .github
│   └── workflows
│       └── pipeline.yml
├── data
│   ├── input
│   │   └── dataset.csv
│   └── output
│       ├── dataset_processed.csv
│       ├── model.pkl
│       ├── analysis.txt
│       └── plots
├── src
│   ├── etl.py
│   ├── analysis.py
│   ├── model.py
│   └── server.py
└── requirements.txt
```

El archivo pipeline.yml podría quedar de la siguiente manera:

```yaml 
name: Pipeline de CI/CD

on:
  push:
    branches:
      - main

jobs:
  build:
    runs-on: ubuntu-latest
    steps:
      - name: Clonar repositorio
        uses: actions/checkout@v3   
      - name: Configurar Python
        uses: actions/setup-python@v3
        with:
          python-version: '3.9'
      - name: Instalar dependencias
        run: |
          python -m pip install --upgrade pip
          pip install -r requirements.txt
      - name: Ejecutar ETL
        run: |
          python src/etl.py
      - name: Ejecutar análisis de datos
        run: |
          python src/analysis.py
      - name: Entrenar modelo
        run: |
          python src/model.py
      - name: Desplegar modelo
        run: |
          python src/deployment.py
```     

## Despliegue del modelo

Una vez que tenemos el modelo entrenado y evaluado, podemos proceder a desplegar el modelo, para utilizar el modelo podemos crear una API que nos permitira hacer predicciones basados en el request que reciba, podemos utilizar FastAPI para la creación del endpoint, este debe ser un POST methods para recibir un request body con los datos que se desean predecir, este archivo podría quedar de la siguiente manera:

```
/src/server.py
```

```python
from fastapi import FastAPI
import joblib
import pandas as pd

app = FastAPI()
model = joblib.load("model/model.pkl")

@app.post("/predict/")
async def predict(data: dict):
    df = pd.DataFrame([data])
    prediction = model.predict(df)
    return {"prediction": int(prediction[0])}

```  

## Esta aplicación se puede desplegar en un contenedor de Docker, para ello vamos a crear un archivo Dockerfile que se encargue de realizar esto, la ruta de este archivo será:

```
Dockerfile
```     

```dockerfile
FROM python:3.9
WORKDIR /app
COPY . /app
RUN pip install -r requirements.txt
CMD ["uvicorn", "src.server:app", "--host", "0.0.0.0", "--port", "8080"]

``` 

## Despliegue de la aplicación en la nube   

Finalmente para desplegar la aplicación en la nube podemos utilizar algun proveedor de cloud como [Azure](https://docs.microsoft.com/en-us/azure/app-service/), [AWS](https://docs.aws.amazon.com/lambda/latest/dg/services-apigateway.html), [Google Cloud](https://cloud.google.com/run), etc.







