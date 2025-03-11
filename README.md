README - Evaluación de Desarrollo Práctico

Este documento describe la implementación de las seis tareas prácticas que evalúan habilidades en programación, manejo de datos, construcción y validación de modelos, así como buenas prácticas en DevOps.

1. Regresión Lineal Simple

Objetivo

Implementar un modelo de regresión lineal simple a partir de un conjunto de datos y calcular el Mean Absolute Error (MAE) como métrica principal.

Pasos

Cargar el conjunto de datos.

Separar las variables independientes y dependientes.

Entrenar un modelo de regresión lineal.

Calcular y mostrar el MAE.

2. Clasificación con Árbol de Decisión

Objetivo

Entrenar un árbol de decisión para predecir si un cliente comprará un producto bancario.

Datos

Edad, Ingreso Mensual, Estado Civil, Compra
25, 2000, 0, 0
30, 3000, 1, 1
22, 1500, 0, 0
45, 4500, 1, 1
35, 4000, 0, 1
28, 2500, 1, 0
42, 5000, 0, 1
40, 3800, 1, 1
23, 2000, 0, 0
37, 4200, 1, 1

Pasos

Cargar los datos.

Separar las variables independientes (X) y la variable objetivo (y).

Entrenar un árbol de decisión.

Evaluar la precisión del modelo.

3. Proceso ETL

Objetivo

Diseñar un proceso ETL para limpiar y transformar un archivo CSV de usuarios con datos incompletos.

Pasos

Extract: Cargar el archivo CSV.

Transform:

Manejo de valores nulos en "edad" e "ingreso mensual".

Normalización de datos.

Load: Guardar los datos transformados en un nuevo CSV listo para el modelo de Machine Learning.

4. Clasificación de MNIST con Red Neuronal

Objetivo

Entrenar una red neuronal en Keras/TensorFlow o PyTorch para clasificar dígitos manuscritos (MNIST).

Pasos

Cargar la base de datos MNIST.

Construir una red neuronal con al menos una capa oculta.

Entrenar el modelo y evaluar su accuracy.

Asegurar reproducibilidad con:

Control de versiones de datos.

Seeds aleatorias.

Uso de entornos virtuales.

5. Pipeline CI/CD con Integración Continua

Objetivo

Diseñar un pipeline donde:

Se sube un dataset a un repositorio Git.

Se configura un workflow CI/CD con GitHub Actions o Jenkins para:

Ejecutar el ETL.

Entrenar un modelo.

Generar reportes de métricas.

Desplegar el modelo en la nube.

Tecnologías y Herramientas

GitHub Actions/Jenkins.

Docker para despliegue.

Cloud services (AWS, Azure, GCP).

6. Pipeline NLP con Named Entity Recognition (NER)

Objetivo

Procesar artículos de prensa en español con spaCy y realizar Named Entity Recognition (NER).

Pasos

Limpieza y normalización:

Eliminación de caracteres no deseados.

Tokenización y lematización.

Aplicar NER:

Identificar entidades de tipo Personas, Organizaciones y Lugares.

Extracción de oraciones relevantes:

Guardar en CSV con columnas: "oración", "entidades encontradas", "tipo de entidad".

Evaluación de calidad:

Métricas de precisión, recall y F1-score.

Requisitos

Python 3.x

Bibliotecas necesarias:

numpy, pandas, scikit-learn, tensorflow, torch, spacy, matplotlib

Docker y GitHub Actions/Jenkins (para CI/CD)

Ejecución

Cada ejercicio se puede ejecutar de manera independiente en un entorno Python con las bibliotecas adecuadas instaladas.
