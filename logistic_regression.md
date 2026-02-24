# Logistic Regression – Modèle Linéaire de Référence

## Pourquoi ce modèle ?

La Logistic Regression est un modèle fondamental en Machine Learning pour la classification binaire.  
Dans ce projet de détection de pneumonie à partir de radiographies thoraciques, elle est utilisée comme **modèle baseline** afin de comparer ses performances avec :

- Random Forest (modèle ensemble non linéaire)
- EfficientNet (Deep Learning)
- KNN

Elle permet d’évaluer si des modèles plus complexes apportent un gain réel de performance.

---

# Outils utilisés

## scikit-learn

Nous utilisons **scikit-learn** car :

- Implémentation stable et optimisée de Logistic Regression
- Calcul automatique des métriques (Accuracy, Recall, F1-score, ROC-AUC, Precision)
- Pipeline clair et reproductible
- Standard académique en Machine Learning

## StandardScaler

La Logistic Regression est un modèle linéaire sensible à l’échelle des variables.  
Nous utilisons donc un **StandardScaler** pour :

- Centrer les données (moyenne = 0)
- Réduire la variance (écart-type = 1)
- Stabiliser l’optimisation

## PCA (Principal Component Analysis)

Le PCA est utilisé pour :

- Réduire la dimension des données
- Éliminer les redondances entre pixels
- Améliorer la généralisation
- Réduire le risque de surapprentissage

---

# Pipeline de traitement

## 1. Préprocessing des images

Les images suivent le pipeline suivant :

Image RGB  
→ Conversion en niveaux de gris  
→ Redimensionnement en 64×64  
→ Aplatissement (flatten)  
→ Standardisation  
→ Entraînement Logistic Regression

### Dimension des données

- Image 64×64 = 4,096 pixels
- Chaque pixel devient une variable
- Entrée finale = vecteur de 4,096 features

---

# Modèle 1 : Logistic Regression – Pixels bruts

## Configuration

```python
LogisticRegression(
    max_iter=5000,
    solver="saga",
    class_weight="balanced",
    n_jobs=-1
)
```

### Paramètres importants

- **max_iter=5000**  
  Permet au modèle de converger malgré le grand nombre de variables.

- **solver="saga"**  
  Optimiseur robuste pour haute dimension.

- **class_weight="balanced"**  
  Corrige le déséquilibre entre classes NORMAL et PNEUMONIA.

---

## Caractéristiques

| Élément            | Valeur                          |
| ------------------ | ------------------------------- |
| Nombre de features | 4,096                           |
| Type de données    | Pixels bruts                    |
| Avantage           | Simple et rapide                |
| Inconvénient       | Sensible au bruit et variations |

---

# Modèle 2 : Logistic Regression + PCA

## Pourquoi PCA ?

Les pixels voisins sont fortement corrélés.  
La dimension élevée (4,096 variables) peut provoquer :

- Surapprentissage
- Bruit inutile
- Instabilité du modèle

Le PCA transforme les données vers un nouvel espace :

- Variables décorrélées
- Dimension réduite
- Conservation de 95% de la variance

---

## Pipeline PCA

Pixels (4,096)  
→ StandardScaler  
→ PCA (95% variance conservée)  
→ Logistic Regression

---

## Avantages du PCA

- Réduction dimensionnelle importante
- Apprentissage plus rapide
- Moins de bruit
- Meilleure généralisation

---

# Fonctionnement mathématique de la Logistic Regression

## Étape 1 : Combinaison linéaire

z = w1x1 + w2x2 + ... + wnxn + b

Chaque pixel contribue avec un poids appris.

---

## Étape 2 : Fonction sigmoïde

σ(z) = 1 / (1 + e^(-z))

Cette fonction transforme la sortie en probabilité entre 0 et 1.

---

## Étape 3 : Décision

Si probabilité ≥ 0.5 → Pneumonia  
Sinon → Normal

---

# Métriques d’évaluation

## Accuracy

Accuracy = (TP + TN) / Total

Mesure globale de performance.  
Peut être trompeuse si classes déséquilibrées.

---

## Precision

Precision = TP / (TP + FP)

Parmi les patients diagnostiqués Pneumonia, combien sont réellement malades ?

---

## Recall (Sensibilité)

Recall = TP / (TP + FN)

Parmi les patients réellement atteints, combien sont détectés ?

⚠️ En médecine, le Recall est prioritaire.

---

## F1-score

F1 = 2 × (Precision × Recall) / (Precision + Recall)

Compromis entre Precision et Recall.

---

## ROC-AUC

Mesure indépendante du seuil de décision.

- 1.0 = parfait
- 0.5 = aléatoire

Très pertinent en diagnostic médical.

---

# Matrice de confusion

|               | Pred Normal | Pred Pneumonia |
| ------------- | ----------- | -------------- |
| **Normal**    | TN          | FP             |
| **Pneumonia** | FN          | TP             |

- **TP** : Pneumonie correctement détectée
- **TN** : Patient sain correctement identifié
- **FP** : Faux positif (moins grave)
- **FN** : Faux négatif (grave en médecine)

---

# Considérations médicales

Priorité absolue :

Faux négatif >> Faux positif

Manquer une pneumonie est plus grave qu’un surdiagnostic.

---

# Recommandation

La version **Logistic Regression + PCA** est recommandée :

- Meilleure généralisation
- Moins sensible au bruit
- Plus stable
- Bon compromis performance / simplicité

---

# Conclusion

La Logistic Regression constitue un baseline solide, rapide et interprétable.

Elle permet :

- D’évaluer la complexité nécessaire du modèle
- De comparer efficacement avec Random Forest
- De mesurer le gain réel apporté par EfficientNet

C’est un modèle simple mais rigoureux, adapté aux contraintes académiques et médicales.
