# T-DEV-810-PAR_8

## KNN
### Prérequis — Activation du environnement virtuel

#### Make
##### 1. Installer Make
``` bash
winget install GnuWin32.Make
```
##### 2. Environnement variable
Si en faisant "make --version" il n'existe pas, il faut ajouter make au PATH.
``` bash
set PATH=%PATH%;C:\Program Files (x86)\GnuWin32\bin
```
#### Python
##### 1. Installer Python
Vérifie que Python est installé (version 3.10 ou supérieure recommandée) :
```bash
python --version
```
Téléchargement : https://www.python.org/downloads/
##### 2. Créer le virtual environment
```bash
python -m venv .venv
```
##### 3. Activer le virtual environment
Toujours activer le venv **avant** de lancer Jupyter ou d'exécuter des scripts
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

##### 4. Installer les dépendances
```bash
pip install -r requirements.txt
```
##### 5. Désactiver le virtual environment
```bash
deactivate
```
#### Git
##### 1. Cloner le projet
```bash
git clone https://github.com/votre-repo/Zoidberg2.0.git
cd Zoidberg2.0
```

## Sauvegarder les données potentiellement lourds - GitHub Releases
- Create a new release
```bash
gh release create v1.0 --title "KNN model" --notes "This release stores all large files such as model weights and pipeline artifacts."
```
- Sauvegarder un artifact dans le github release
```bash
gh release upload ${ReleaseName} knn_pca_lda_grid_search.pkl 
```

### Structure du dossier
C:.
├───documents
│   ├───images
│   ├───model
│   │   ├───dataset
│   │   └───hyperparameter
│   └───texts
├───notebooks
│   ├───process
│   └───utils
├───powerpoint
└───results
    └───fine_tuning