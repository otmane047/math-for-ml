import numpy as np
from sklearn.linear_model import LogisticRegression

def logistic_regression(X, y):
    m = LogisticRegression()
    m.fit(X, y)
    return m

# Exemple d'utilisation
X = np.array([[1], [2], [3]])
y = np.array([0, 0, 1])
my_model = logistic_regression(X, y)
print("Prédiction :", my_model.predict([[4]]))