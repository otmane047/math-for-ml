import numpy as np
from sklearn.neighbors import KNeighborsClassifier

def knn(X, y, n_neighbors=5):
    m = KNeighborsClassifier(n_neighbors=n_neighbors)
    m.fit(X, y)
    return m

# Exemple d'utilisation
X = np.array([[1], [2], [3]])
y = np.array([0, 0, 1])
my_model = knn(X, y, n_neighbors=2)
my_model.fit(X, y)
print("Prédiction :", my_model.predict([[4]]))