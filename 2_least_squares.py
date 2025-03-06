import numpy as np

def least_squares(X, y):
    theta = np.linalg.inv(X.T.dot(X)).dot(X.T).dot(y)  # Solution analytique
    return theta

# Exemple d'utilisation
X = np.array([[1, 2], [1, 3], [1, 4]])
y = np.array([5, 7, 9])
theta = least_squares(X, y)
print("Paramètres optimaux :", theta)
