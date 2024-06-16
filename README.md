# Projet Pokedex

# Introduction
Projet de classification supervisée des pokémons à base de deep learning

# Bibliothèques et Dépendances
Les bibliothèques suivantes sont nécessaires pour exécuter ce projet :
- `numpy`
- `matplotlib`
- `tensorflow`
- `keras`
- `scipy`
- `os`

La version python utilisée doit être compatible avec Tensorflow. La version utilisée pour ce projet est python 3.8.

# Fichiers Python
- cnn_pokemon.py est un CNN assez simple
- classifieur_resnet_pokemon.py utilise le transfer learning avec ResNet-101

# Dataset
Le dossier Data contient les données utilisées pour l'entraînement et la validation (20 x 100 images).
Le dossier Test contient les données utilisées pour la phase de test (100 images).

# Path
Il est conseillé, pour éviter des problèmes, de mettre tous les chemins en dur dans le code. Exemple : data_dir = r"C:\Users\Marc\Pictures\Pokemon\Data".
