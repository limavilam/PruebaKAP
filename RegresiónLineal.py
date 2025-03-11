import numpy as np
import matplotlib.pyplot as plt

X = np.array([2.5, 3.8, 4.6, 6.8, 7.2, 9.5])  
y = np.array([2.1, 2.9, 4.2, 5.8, 7.1, 8.5])  

def linear_regression(X, y):
    
    length_dataset= len(X)
    summation_x = np.sum(X)
    summation_y = np.sum(y)
    summation_xy = np.sum(X * y)
    summation_x2 = np.sum(X ** 2)
    
    slope = (length_dataset * summation_xy - summation_x * summation_y) / (length_dataset * summation_x2 - summation_x ** 2)
    intercept = (summation_y - slope * summation_x) / length_dataset
    
    return slope, intercept

def predict(X, slope, intercept):
    predict_calculation = slope * X + intercept
    return predict_calculation

def mean_absolute_error(y_true, y_pred):
    mae = np.mean(np.abs(y_true - y_pred))
    return mae

slope, intercept = linear_regression(X, y)

y_pred = predict(X, slope, intercept)

mae_result = mean_absolute_error(y, y_pred)
print(f'Mean Absolute Error (MAE): {mae_result}')

plt.scatter(X, y, color='pink', label='Datos reales')
plt.plot(X, y_pred, color='green', linestyle='--', label='Regresión Lineal')
plt.xlabel('X')
plt.ylabel('y')
plt.title('Regresión Lineal')
plt.legend()
plt.show()