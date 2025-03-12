from itertools import product
import numpy as np

def brute_force_regression(X, y, param_range):
    best_error = float('inf')
    best_params = None
    for params in product(param_range, repeat=X.shape[1]):
        y_pred = X.dot(params)
        error = np.mean((y - y_pred) ** 2)
        if error < best_error:
            best_error = error
            best_params = params
    return best_params

# Exemple d'utilisation
X = np.array([[1, 2], [1, 3], [1, 4]])
y = np.array([5, 7, 9])
param_range = np.linspace(-10, 10, 100)
best_params = brute_force_regression(X, y, param_range)
print("Meilleurs paramètres :", best_params)