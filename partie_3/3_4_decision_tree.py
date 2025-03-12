import numpy as np
from sklearn.tree import DecisionTreeRegressor

def decision_tree(X, y):
    m = DecisionTreeRegressor()
    m.fit(X, y)
    return m

# Exemple d'utilisation
X = np.array([[1], [2], [3]])
y = np.array([1, 4, 9])
my_model = decision_tree(X, y)
print("Prédiction :", my_model.predict([[4]]))