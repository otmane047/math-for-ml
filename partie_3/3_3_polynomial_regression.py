import numpy as np
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression

def polynomial_regression(X, y, degree):
    poly_features = PolynomialFeatures(degree=degree)
    X_poly = poly_features.fit_transform(X)
    m = LinearRegression()
    m.fit(X_poly, y)
    return m


# Exemple d'utilisation
X = np.array([[1], [2], [3]])
y = np.array([1, 4, 9])
my_model = polynomial_regression(degree=2)
print("Coefficients :", my_model.coef_)