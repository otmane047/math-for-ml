import numpy as np

def gradient_descent(X, y, learning_rate=0.01, n_iterations=1000):
    n_samples = len(y)
    theta = np.zeros(X.shape[1])  # Initialisation des paramètres
    for _ in range(n_iterations):
        gradient = (2/n_samples) * X.T.dot(X.dot(theta) - y)  # Calcul du gradient
        theta -= learning_rate * gradient  # Mise à jour des paramètres
    return theta

# Exemple d'utilisation
X = np.array([[1, 2], [1, 3], [1, 4]])
y = np.array([5, 7, 9])
theta = gradient_descent(X, y)
print("Paramètres optimaux :", theta)
