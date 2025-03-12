# Importation des bibliothèques nécessaires
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, learning_curve
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt

# 1. Génération de données synthétiques
np.random.seed(42)  # Pour la reproductibilité
n_samples = 1000  # Nombre d'échantillons

# Caractéristiques (features)
budget_pub = np.random.uniform(1000, 10000, n_samples)  # Budget publicitaire
saison = np.random.choice([1, 2, 3, 4], n_samples)  # Saison (1: Hiver, 2: Printemps, 3: Été, 4: Automne)
promotion = np.random.randint(0, 2, n_samples)  # Promotion (0: Non, 1: Oui)

# Génération des ventes (cible)
ventes = 5000 + 0.5 * budget_pub + 1000 * (saison == 3) + 2000 * promotion + np.random.normal(0, 500, n_samples)

# Création d'un DataFrame
data = pd.DataFrame({
    'budget_pub': budget_pub,
    'saison': saison,
    'promotion': promotion,
    'ventes': ventes
})

# 2. Préparation des données
# Séparation des caractéristiques (X) et de la cible (y)
X = data[['budget_pub', 'saison', 'promotion']]
y = data['ventes']

# Division des données en ensembles d'entraînement et de test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 3. Création et entraînement du modèle
model = LinearRegression()
model.fit(X_train, y_train)

# 4. Prédictions sur l'ensemble de test
y_pred = model.predict(X_test)

# 5. Évaluation du modèle
# Calcul de l'Erreur Quadratique Moyenne (MSE)
mse = mean_squared_error(y_test, y_pred)
print(f"Erreur Quadratique Moyenne (MSE) : {mse:.2f}")

# Calcul de la Racine Carrée de l'MSE (RMSE)
rmse = np.sqrt(mse)
print(f"Racine Carrée de l'Erreur Quadratique Moyenne (RMSE) : {rmse:.2f}")

# Calcul de l'Erreur Absolue Moyenne (MAE)
mae = mean_absolute_error(y_test, y_pred)
print(f"Erreur Absolue Moyenne (MAE) : {mae:.2f}")

# Calcul du Coefficient de Détermination (R²)
r2 = r2_score(y_test, y_pred)
print(f"Coefficient de Détermination (R²) : {r2:.2f}")

# 6. Visualisation des performances
# Courbe d'apprentissage
train_sizes, train_scores, test_scores = learning_curve(model, X, y, cv=5, scoring='neg_mean_squared_error')

# Conversion des scores négatifs en MSE
train_scores_mean = -np.mean(train_scores, axis=1)
test_scores_mean = -np.mean(test_scores, axis=1)

# Tracé de la courbe d'apprentissage
plt.figure()
plt.plot(train_sizes, train_scores_mean, 'o-', color="r", label="Score d'entraînement")
plt.plot(train_sizes, test_scores_mean, 'o-', color="g", label="Score de validation")
plt.xlabel("Taille de l'ensemble d'entraînement")
plt.ylabel("MSE")
plt.title("Courbe d'apprentissage")
plt.legend(loc="best")
plt.show()

# 7. Prédiction sur une nouvelle observation
new_observation = np.array([[5000, 3, 1]])  # Budget publicitaire = 5000, Saison = Été, Promotion = Oui
prediction = model.predict(new_observation)
print(f"Prédiction pour la nouvelle observation : {prediction[0]:.2f}")