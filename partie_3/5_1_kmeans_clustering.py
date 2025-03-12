import numpy as np
from sklearn.cluster import KMeans

def kmeans_clustering(X, n_clusters=3):
    m = KMeans(n_clusters=n_clusters)
    m.fit(X)
    return m


# Exemple d'utilisation
X = np.array([[1, 2], [1, 4], [1, 0], [4, 2], [4, 4], [4, 0]])
my_model = kmeans_clustering(X, n_clusters=2)
my_model.fit(X)
print("Labels :", my_model.labels_)