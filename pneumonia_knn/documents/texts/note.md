Quel est ma compréhension par rapport à KNN et ses implémentation ?

# K-Nearest Neighbors.

## L'importance de flatten()
Dans le pre processing il y a une étape important qui permet de transformer l'image en 1 dimention.

Une image a beaucoup d'informations, mais ce qui est mis en évidence dans le pixel étudié qui a été "flattened" c'est la couleur (RGB) et la position.

Image A : [0.1, 0.5, 0.3, 0.8, 0.2, 0.6, ...]
Image B : [0.2, 0.4, 0.3, 0.9, 0.1, 0.5, ...]
            ↕    ↕    ↕    ↕    ↕    ↕
           R00  G00  B00  R01  G01  B01

### Explication 
C'est un tableau 1D ou vecteur séquentiel, c'est ce qui est donné lorsque l'image a été flatten(). 
Sur chaque image on retrouve l'information de chaque pixel. 
Flatten() ne calcule rien elle fait en sorte de mettre les informations à la suite. 

Les valeurs : D'abord R(ed)[valeur] ensuite G(reen)[valeur] et B(lue)[valeur] et ça recommence. Chaque triple correspond à l'information couleur d'un pixel.
Pourquoi en float ? Parce que Normalize fait une opération mathématique qui centre et réduit les valeurs des pixels pour les mettre à la même échelle.
 
Les index : Le type de couleur (R, G ou B) avec la position du pixel (03 par exemple), ça donne R03 par exemple. 

### Pourquoi

KNN ne sait pas travailler avec des images 2D ou plus. Parce qu'il fait la comparaison sur un espace 1D en calculant une distance.


## L'importance des implémentations

### PCA (Principal Component Analysis)
             R00   G00   B00 ...
Image 1  →  0.1   0.5   0.3
Image 2  →  0.2   0.4   0.3
Image 3  →  0.8   0.5   0.9
Image 4  →  0.1   0.4   0.3
              ↓     ↓     ↓
          variance variance variance
          élevée   faible   élevée
          → garde  → vire  → garde

### LDA ( Linear Discriminant Analysis)


## Limite : Perte de structure spacial

Il y a principalement 3 étapes dans les modèles d'algo des machine learner :

# Le training : 
Dans cette étape, KNN est souvent associé à un "lazy learner" parce que concrétement ce qu'elle fait c'est de mémoriser les images. Contrairement à un modèle classique comme un réseau de neuronnes, KNN ne calcule rien, elle retient juste les images et les labels.

# L'évaluation :
Dans cette étape, 