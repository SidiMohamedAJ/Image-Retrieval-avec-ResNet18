# Recherche d'images similaires avec ResNet18

Ce projet vise à rechercher des images similaires dans une base de données à l'aide du modèle ResNet18 pré-entraîné.

## Extraction des caractéristiques

Nous utilisons le modèle ResNet18 pré-entraîné pour extraire les caractéristiques des images de la base de données. Les images sont prétraitées à l'aide de transformations standard telles que le redimensionnement et la normalisation, puis nous utilisons la couche avgpool du modèle ResNet18 pour extraire les caractéristiques.

Les caractéristiques extraites sont ensuite stockées dans un fichier numpy (all_vecs.npy) pour une utilisation ultérieure.

## Recherche d'images similaires en ligne
Lors de la recherche d'images similaires en ligne, l'utilisateur sélectionne une image à l'aide de l'interface Streamlit. L'image sélectionnée est alors prétraitée et ses caractéristiques sont extraites à l'aide du même processus que celui utilisé pour les images de la base de données.

Ensuite, nous calculons la distance euclidienne entre les caractéristiques de l'image requête et celles de toutes les images de la base de données. Les indices des images les plus similaires sont obtenus en classant les distances calculées.

Les images correspondantes sont affichées en sortie pour l'utilisateur, lui permettant de visualiser les images les plus similaires à celle qu'il a sélectionnée.

## Comment exécuter le code
- Extraction des caractéristiques :
Exécutez le script "database_feature_extraction.py" pour extraire les caractéristiques des images de la base de données.
- Recherche d'images similaires en ligne :
Exécutez le script "streamlit run main.py" pour rechercher des images similaires en ligne.


## Dépendances
- Python 3.x
- PyTorch
- Torchvision
- NumPy
- Streamlit
