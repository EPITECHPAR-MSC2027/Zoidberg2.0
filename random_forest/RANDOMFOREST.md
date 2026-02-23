# Random Forest - Modèle clé de comparaison

## Pourquoi on l'utilise

Random Forest est un excellent compromis:
- **Robuste**: Peu sensible au surapprentissage
- **Performance**: Très bon rapport performance/complexité sans réglage complexe
- **Interprétabilité**: Importance des variables bien définie
- **Flexibilité**: Fonctionne bien avec ou sans extraction de features
- **Scalabilité**: Parallélisation efficace avec multi-CPU

C'est un modèle de référence solide en Machine Learning classique et médical.

## Outils choisis

- **scikit-learn**: Implémentation stable, performante et standard de Random Forest
- **scikit-image**: Extraction de features visuelles avancées (HOG)

---

# Deux approches comparées

Zoidberg 2.0 implémente **deux modèles** pour comparer les stratégies de préprocessing:

## Modèle 1: Random Forest SANS HOG (Pixels bruts)

**Fichier**: `app.py` chargement du modèle `random_forest_model.pkl`

### Préprocessing
```
Image couleur (RGB) → Niveaux de gris → Redimensionnement 128×128 
→ Aplatissement (16,384 pixels) → Entrée directe au RF
```

### Caractéristiques
- **Entrée**: 16,384 features (128×128 pixels aplatis)
- **Type features**: Chaque pixel est une feature brute
- **Avantages**:
  - Simple et rapide
  - Capture les patterns bruts
  - Pas de perte d'information par abstraction
- **Inconvénients**:
  - Haute dimensionnalité (risque de surapprentissage)
  - Bruit et variations mineures affectent le modèle
  - Moins robuste aux rotations/translations
  - Sensible aux changements d'intensité lumineuse

### Configuration Random Forest
```python
RandomForestClassifier(
    n_estimators=300,           # 300 arbres
    class_weight="balanced",    # Équilibrage des classes
    random_state=42,            # Reproductibilité
    n_jobs=-1                   # Tous les CPU
)
```

### Seuil de décision
- **Seuil standard**: 0.5
- **Optimisation**: Via indice Youden sur validation set

---

## Modèle 2: Random Forest + HOG (Features extraites)

**Fichier**: `random_forest.py` génère `random_forest_hog_model.pkl`

### Préprocessing
```
Image couleur (RGB) → Niveaux de gris → Redimensionnement 128×128
→ Extraction HOG features → vecteur 2,916 features → Entrée au RF
```

### HOG (Histogram of Oriented Gradients)
Extraction de descripteurs de gradients orientés - capture les contours et structures:

**Configuration HOG**:
```python
hog(
    image,
    orientations=9,              # 9 directions de gradients
    pixels_per_cell=(8, 8),      # Grille 8×8 pixels
    cells_per_block=(2, 2),      # Blocs 2×2 cellules
    block_norm="L2-Hys",         # Normalisation L2-Hys
    visualize=False
)
```

**Résultat**: 
- Image 128×128 → Grid de 16×16 cellules
- Chaque bloc 2×2 cellules avec 9 orientations = 4×9 = 36 valeurs
- Total: (14×14) × 36 = **2,916 features** (vs 16,384 pixels)

### Caractéristiques
- **Entrée**: 2,916 features (réduction dimensionnelle 5.6×)
- **Type features**: Gradients orientés et contours détectés
- **Avantages**:
  - Dimension réduite → apprentissage plus rapide
  - **Robuste aux variations**: Insensible aux changements d'intensité mineurs
  - **Sémantique**: Capture les structures pulmonaires (contours, cavités)
  - **Moins de bruit**: Features semantiques au lieu de pixels bruts
  - **Meilleur transfert**: Features transportables à d'autres radiographies
  - **Données médicales**: Utilisé en imagerie depuis 10+ ans
- **Inconvénients**:
  - Moins de détails fins
  - Plus lent à calculer (extraction HOG)
  - Perte d'information texturelle très fine

### Configuration Random Forest
```python
RandomForestClassifier(
    n_estimators=500,                    # 500 arbres (plus pour compenser réduction features)
    max_depth=None,                      # Arbre complet (pas de limitation profondeur)
    min_samples_split=5,                 # Régularisation légère
    class_weight="balanced_subsample",   # Équilibrage avancé par subsample
    random_state=42,
    n_jobs=-1
)
```

### Seuil optimisé
- **Youden Threshold**: ~0.903 (trouvé sur validation set)
- **Raison**: En médecine, priorité au **Recall** (détecter toutes pneumonies)
- **Interprétation**: Seuil **plus haut** = moins de faux positifs mais plus strict
  - Threshold 0.5: Diagnostic "pneumonia" si ≥50% d'arbres votent OUI
  - Threshold 0.903: Diagnostic "pneumonia" si ≥90.3% d'arbres votent OUI

---

## Comparaison détaillée

| Aspect | Sans HOG | Avec HOG |
|--------|----------|----------|
| **Features** | 16,384 (pixels) | 2,916 (gradients) |
| **Réduction dimension** | - | 5.6× moins |
| **Type données** | Pixelaires bruts | Sémantiques |
| **Arbres** | 300 | 500 |
| **Temps extraction** | Nul | +3-5s par image |
| **Temps train** | Rapide | 2-3× plus lent |
| **Robustesse lumière** | Faible | Excellente |
| **Détails fins** | Oui | Limités  |
| **Interprétabilité** | Pixels directs | Structures visuelles |
| **Surapprentissage** | Risqué (haute dim) | Réduit (bonne dim) |
| **Performance médical** | Bonne | Optimisée |
| **Seuil optimal** | 0.5 | 0.903 |

---

## Considérations pour le diagnostic médical

### Priorité: RAPPEL (Recall = Sensibilité)
```
Coût erreur: Manquer une pneumonie >> Faux positif
```

- **Faux négatif (erreur grave)**: Patient a pneumonie mais diagnostic = normal
- **Faux positif (moins grave)**: Patient normal mais diagnostic = pneumonia (suivi clinique supplémentaire)

### Indice Youden (Optimalité)
$$\text{Youden} = \text{Sensibilité} + \text{Spécificité} - 1 = \text{TPR} - \text{FPR}$$

- Maximise la somme des vrais positifs et vrais négatifs
- Trouve le point optimal indépendant de la prévalence
- Pour HOG: ~0.903 (plutôt que 0.5)

### Recommandation pour production
**Utiliser le modèle HOG + Seuil Youden (0.903):**
- Features sémantiques plus stables entre radiographies différentes
- Seuil optimisé pour diagnostic médical (priorité rappel)
- Robuste aux variations d'acquisition (différents appareils, protocoles)
- Meilleur compromis sensibilité/spécificité

---

## Fonctionnement du Random Forest

### Étape 1: Construire N arbres (N=300 ou 500)
- Pour chaque arbre i:
  - Créer bootstrapped sample (tirage aléatoire avec remise) des données
  - Entraîner arbre de décision sur ce sample
  - À chaque split, considérer sous-ensemble aléatoire de features

### Étape 2: Ajouter diversité
- Chaque arbre voit données différentes (bootstrap)
- Chaque split considère features différentes
- Résultat: Forêt pas correlée = réduction overfitting

### Étape 3: Prédire (Classification)
- **Classification**: Chaque arbre prédit classe, **vote majoritaire**
- **Probabilités**: Fraction d'arbres votant pour chaque classe
  - Exemple: 280/300 arbres votent "PNEUMONIA" = 93.3% probabilité
  - Avec threshold 0.5: Classé PNEUMONIA 
  - Avec threshold 0.903: Classé NORMAL (trop strict)

---

## Métriques & Évaluation

### Accuracy
$$\text{Accuracy} = \frac{TP + TN}{TP + TN + FP + FN}$$
**Trompeur** si classes déséquilibrées (ex: 95% normaux)

### Precision (Précision) 
$$\text{Precision} = \frac{TP}{TP + FP}$$
"Sur les patients diagnostiqués pneumonia, combien avaient vraiment?"

### Recall (Rappel = Sensibilité)
$$\text{Recall} = \frac{TP}{TP + FN}$$
**Critique en médecine**: "Sur les patients pneumonia, combien avons-nous détecté?"

### ROC-AUC (Area Under Curve)
- Trace TPR vs FPR pour tous les thresholds
- Indépendant du seuil choisi
- **1.0** = Modèle parfait, **0.5** = Aléatoire
- Plus pertinent pour diagnostic médical que Accuracy

### Confusion Matrix
```
                Prédiction
              NORMAL | PNEUMONIA
    ──────────────────────────────
NORMAL       |  TN   |    FP    |
PNEUMONIA    |  FN   |    TP    |
```

- **TP** (True Positive): Patient pneumonia, diagnostiqué pneumonia
- **TN** (True Negative): Patient normal, diagnostiqué normal
- **FP** (False Positive): Patient normal, diagnostiqué pneumonia (surdiagnostic)
- **FN** (False Negative): Patient pneumonia, diagnostiqué normal (manqué grave!)