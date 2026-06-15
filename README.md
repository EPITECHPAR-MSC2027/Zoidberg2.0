## pneumonia_knn

Classification de radiographies thoraciques en trois classes — **NORMAL**, **BACTERIA**, **VIRUS** — à l'aide d'un classifieur K plus proches voisins (KNN), précédé d'une réduction de dimension par PCA et d'un rééquilibrage des classes par SMOTE.

### Aperçu

Le pipeline complet est `StandardScaler → PCA → SMOTE → KNN`, encapsulé dans un unique objet afin d'éviter toute fuite de données. Les hyperparamètres sont sélectionnés par `GridSearchCV` (validation croisée sur le jeu d'entraînement uniquement), avec le **F1-score** comme critère de sélection, choix adapté au déséquilibre des classes.

Le modèle est évalué sur deux ensembles indépendants :

- **Validation** — contrôle intermédiaire.
- **Test** — évaluation finale (référence des performances annoncées).

### Prérequis

- Python 3.10+
- Dépendances principales : `scikit-learn`, `imbalanced-learn` (SMOTE), `numpy`, `pandas`, `matplotlib`, `joblib`.

Les dépendances sont listées dans le `requirements.txt` situé à la racine du projet (`Zoidberg2.0/`). Installation :

```bash
pip install -r requirements.txt
```

### Utilisation

1. Placer les datasets sérialisés dans `model/dataset/` (voir l'arborescence ci-dessous).
2. Ajuster `run_index` en tête de notebook pour nommer le run courant.
3. Exécuter le notebook `knn_corrige_12.ipynb` de haut en bas.

Le notebook charge le `GridSearchCV` pré-entraîné (`model/hyperparameter/`), effectue les prédictions, puis génère pour chaque contexte (finetuning, validation, test, récapitulatif global) les visualisations associées au format **PNG + JSON**.

> Les résultats sont écrits **hors** de `pneumonia_knn`, à la racine du projet, dans `results/k_nearest_neighbors/run_{index}/` (voir arborescence).

> **⚠️ Avertissement — Téléchargement asynchrone**
>
> Lors du premier lancement, le notebook télécharge le dataset et le modèle pré-entraîné (GridSearch). **Ce téléchargement ne bloque pas l'exécution** : les cellules suivantes peuvent démarrer avant la fin du chargement, ce qui provoque une erreur (par exemple le PCA ou le dataset pas encore disponible).
>
> **Solution : attendre que le téléchargement soit terminé, puis réexécuter la cellule concernée** (ou relancer le notebook entier). Une fois les fichiers présents en local, le problème ne se reproduit plus.

Un `Makefile` est fourni pour les commandes courantes du projet.

### Arborescence

> Les fichiers `.pkl` (datasets et GridSearch) ne sont pas versionnés : ils sont générés/fournis séparément.

```text
pneumonia_knn/
├── knn_corrige_12.ipynb        # Notebook principal (pipeline + évaluation + visualisations)
├── Makefile
└── model/
    ├── dataset/                # Datasets sérialisés (non versionnés)
    │   ├── dataset_train.pkl
    │   ├── dataset_val.pkl
    │   └── dataset_test.pkl
    ├── hyperparameter/         # GridSearch pré-entraîné (non versionné)
    │   └── knn_pca_grid_search.pkl
    └── run/                    # (réservé)
```

Résultats générés à l'exécution, à la racine du projet :

```text
Zoidberg2.0/                    # Racine du projet
├── requirements.txt            # Dépendances Python
├── pneumonia_knn/              # (ce dossier)
└── results/
    └── k_nearest_neighbors/
        └── run_{index}/
            ├── finetuning/     # Tableau + barplot du GridSearchCV (PNG + JSON)
            ├── validation/     # Matrice de confusion, vrais positifs, report, etc.
            ├── test/           # Idem, sur le jeu de test
            └── global/         # Récapitulatif global (rappel par classe, accuracy, F1 macro)
```

### Sorties générées

Pour chaque ensemble évalué (validation et test) :

- Matrice de confusion (PNG + JSON)
- Vrais positifs par classe (PNG + JSON)
- Classification report (PNG + JSON)
- Scatter des prédictions en projection 2D (PNG)
- Feature importance et heatmap pixel (PNG + JSON)

Le dossier `finetuning/` contient les résultats de la recherche d'hyperparamètres, et `global/` la synthèse comparée validation / test.
