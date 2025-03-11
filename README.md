# Sección 2: Preguntas de Desarrollo Práctico

Este repositorio contiene soluciones y recursos relacionados con 6 tareas prácticas diseñadas para evaluar habilidades en programación, manejo de datos, construcción y validación de modelos, así como buenas prácticas de DevOps.

---

## Tareas

### **1. Regresión Lineal Simple**
**Objetivo:** Implementar una regresión lineal simple con un pequeño conjunto de datos y calcular el **Mean Absolute Error (MAE)** como métrica principal.

**Instrucciones:**
- Crear un modelo de regresión lineal.
- Entrenar el modelo con datos de ejemplo.
- Calcular y mostrar el MAE.

**Archivos Relacionados:** 
- [`regresion_lineal.py`](./regresion_lineal.py)

---

### **2. Árbol de Decisión para Clasificación**
**Objetivo:** Construir un modelo de Árbol de Decisión para predecir si un cliente comprará o no un producto bancario, utilizando el conjunto de datos proporcionado.

**Enunciado:**
- Variables: Edad, ingreso mensual, estado civil (0: soltero, 1: casado), compra (0: no compra, 1: compra).
- **Tareas:** Entrenar el modelo y mostrar la **exactitud (accuracy)** en el conjunto de datos.

**Archivos Relacionados:** 
- [`decision_tree_classifier.py`](./decision_tree_classifier.py)

**Conjunto de Datos:**
```plaintext
[
 [25, 2000, 0, 0],
 [30, 3000, 1, 1],
 [22, 1500, 0, 0],
 [45, 4500, 1, 1],
 [35, 4000, 0, 1],
 [28, 2500, 1, 0],
 [42, 5000, 0, 1],
 [40, 3800, 1, 1],
 [23, 2000, 0, 0],
 [37, 4200, 1, 1]
]
