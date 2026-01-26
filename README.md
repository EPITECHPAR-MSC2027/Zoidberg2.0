T-DEV-810-PAR_8/
├── src/
│   ├── __init__.py
│   ├── data/
│   │   ├── __init__.py
│   │   └── loader.py           # load_dataset()
│   ├── preprocessing/
│   │   ├── __init__.py
│   │   └── processor.py        # dataset_to_arrays()
│   ├── models/
│   │   ├── KNN
│   │   │   ├── __init__.py
│   │   │   ├── knn.py              # KNN
        │   ├── evaluation/
        │   │   ├── __init__.py
        │   │   └── metrics.py          # accuracy, confusion matrix, etc.
        │   └── visualization/
        │       ├── __init__.py
        │       └── plots.py            # Tous les graphiques
├── notebooks/
│   └── analysis.ipynb
├── configs/
│   ├── knn_config.yaml
│   ├── rf_config.yaml
│   └── svm_config.yaml
├── main.py                      # 1 seul point d'entrée
├── requirements.txt
├── README.md
└── .gitignore
# T-DEV-810-PAR_8