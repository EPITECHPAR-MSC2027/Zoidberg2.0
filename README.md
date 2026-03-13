# T-DEV-810-PAR_8

## KNN
### Prérequis — Activation du environnement virtuel
#### 1. Installer Python
Vérifie que Python est installé (version 3.10 ou supérieure recommandée) :
```bash
python --version
```
Téléchargement : https://www.python.org/downloads/

#### 2. Cloner le projet
```bash
git clone https://github.com/votre-repo/Zoidberg2.0.git
cd Zoidberg2.0
```

#### 3. Créer le virtual environment
```bash
python -m venv .venv
```

#### 4. Activer le virtual environment
**Windows (CMD / Cmder) :**
```bash
.venv\Scripts\activate
```
**Windows (PowerShell) :**
```bash
.venv\Scripts\Activate.ps1
```
**macOS / Linux :**
```bash
source .venv/bin/activate
```
Une fois activé, tu verras `(.venv)` apparaître au début de ta ligne de commande.

#### 5. Installer les dépendances
```bash
pip install -r requirements.txt
```

#### 6. Désactiver le virtual environment
```bash
deactivate
```

## Notes
- Toujours activer le venv **avant** de lancer Jupyter ou d'exécuter des scripts

### Structure du dossier
pneumonia_knn
├───documents
│   ├───images
│   └───texts
├───notebooks
│   ├───process
│   │   ├───dimentiality_reduction
│   │   └───standard
│   └───visualisation
├───pipeline_ci
└───utils