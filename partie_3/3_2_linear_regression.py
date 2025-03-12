import numpy as np
from sklearn.linear_model import LinearRegression


def linear_regression(X, y):
    m = LinearRegression()
    m.fit(X, y)
    return m


# Exemple d'utilisation
X = np.array([[1, 2], [1, 3], [1, 4]])
y = np.array([5, 7, 9])
my_model = linear_regression(X, y)
print("Coefficients :", my_model.coef_)