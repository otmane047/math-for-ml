import numpy as np
from sklearn.cluster import MeanShift

def mean_shift_clustering(X):
    m = MeanShift()
    m.fit(X)
    return m


# Exemple d'utilisation
X = np.array([[1, 2], [1, 4], [1, 0], [4, 2], [4, 4], [4, 0]])
my_model = mean_shift_clustering(X)
print("Labels :", my_model.labels_)