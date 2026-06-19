# 🫁 Zoidberg 2.0 - Classification de radiographies thoraciques

## Description

Ce projet vise à classifier automatiquement des radiographies thoraciques en trois catégories :

- Normal
- Pneumonia Bacteria
- Pneumonia Virus

L'objectif est d'évaluer et comparer différentes approches de Machine Learning et de Deep Learning pour la détection de pneumonies à partir d'images médicales.

---

## Problématiques

Ce projet cherche à répondre aux questions suivantes :

- Les réseaux de neurones convolutifs (CNN) sont-ils plus performants que les méthodes classiques de Machine Learning pour la classification de radiographies thoraciques ?
- Les pneumonies bactériennes sont-elles plus faciles à détecter que les pneumonies virales ?
- Quel est l'impact du Transfer Learning sur les performances d'un CNN médical ?

---

## Dataset

Le dataset est composé de radiographies thoraciques réparties en trois classes :

| Classe | Description |
|----------|-------------|
| Normal | Radiographies sans pneumonie |
| Pneumonia Bacteria | Pneumonies d'origine bactérienne |
| Pneumonia Virus | Pneumonies d'origine virale |


## Prétraitement des données

Les images sont :

- Converties en RGB
- Redimensionnées en 224 × 224 pixels
- Normalisées
- Chargées via des DataLoaders PyTorch

### Data Augmentation

Uniquement sur l'ensemble d'entraînement :

- Rotations
- Translations
- Zoom
- Symétries horizontales
- Déformations géométriques

---

## Modèles étudiés

### Logistic Regression

- Pixels aplatis
- Standardisation
- PCA (95 % de variance conservée)
- Régularisation L2

### Random Forest

- Extraction de caractéristiques HOG
- SMOTE
- Class Weight équilibré

### Support Vector Machine (SVM)

- Pixels aplatis
- Standardisation
- PCA
- Noyau RBF

### K-Nearest Neighbors (KNN)

- PCA
- SMOTE
- Distance Manhattan

### CNN - EfficientNetB0

- Transfer Learning ImageNet
- Optimiseur Adam
- Cross Entropy Loss

---

## Méthodologie

- Validation croisée stratifiée
- Même découpage des folds pour tous les modèles
- Sélection des hyperparamètres basée sur le F1-score macro

### Métriques utilisées

- Accuracy
- Precision
- Recall
- F1-score
- ROC-AUC

---


## Conclusions

- Le CNN EfficientNetB0 obtient les meilleures performances sur toutes les métriques.
- Le Transfer Learning améliore significativement les performances du CNN.
- Les pneumonies bactériennes sont les plus faciles à détecter.
- La principale difficulté concerne la distinction entre radiographies normales et pneumonies virales.
- Les méthodes classiques obtiennent des résultats corrects mais restent limitées par leur représentation simplifiée des images.

---

## Interprétabilité

Les méthodes suivantes ont été utilisées pour interpréter les prédictions :

- Feature Importance
- Permutation Importance
- Heatmaps
- Grad-CAM

Ces techniques permettent de visualiser les régions des radiographies influençant les décisions des modèles.

---

## Technologies utilisées

- Python
- PyTorch
- Scikit-Learn
- NumPy
- Pandas
- Matplotlib
- OpenCV
- HuggingFace Datasets

---

## Structure du projet

```text
├── data/
├── notebooks/
├── models/
├── src/
│   ├── preprocessing/
│   ├── training/
│   ├── evaluation/
│   └── visualization/
├── results/
├── reports/
└── README.md
```

---

## Auteurs

Projet réalisé dans le cadre d'un projet académique d'Intelligence Artificielle appliquée à l'imagerie médicale.
