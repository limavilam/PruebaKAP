import numpy as np
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.tree import plot_tree

# Datos: edad, ingreso mensual, estado civil (0: soltero, 1: casado), compra (0: no compra, 1: compra)
Datos = np.array([
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
])

X = Datos[:, :-1]
#print(X)
y = Datos[:, -1]
#print(y)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = DecisionTreeClassifier (random_state=42)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy del modelo: {accuracy:.2f}')

plt.figure(figsize=(10, 6))
plot_tree(model, feature_names=['Edad', 'Ingreso', 'Estado Civil'], class_names=['No Compra', 'Compra'], filled=True)
plt.show()



