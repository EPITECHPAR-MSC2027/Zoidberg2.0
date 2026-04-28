.PHONY: help install run test clean lint format

# Variables
ifeq ($(OS),Windows_NT)
    PYTHON := C:\Tools\python\python.exe
    PIP := C:\Tools\python\Scripts\pip.exe
else
    PYTHON := python3
    PIP := pip3
endif

VENV := .venv

KNN-help:
	@echo "============================================================"
	@echo "         T-DEV-810 - Analyse X-ray KNN                      "
	@echo "============================================================"
	@echo ""
	@echo "Commandes disponibles:"
	@echo ""
	@echo "  make KNN-install  - Installer les dependances"
	@echo "  make KNN-run      - Lancer le modele KNN"
	@echo "  make KNN-clean    - Nettoyer les fichiers generes"
	@echo "  make KNN-lint     - Verifier la qualite du code"
	@echo "  make venv         - Creer l'environnement virtuel"
	@echo "  make help         - Afficher cette aide"
	@echo ""

# Installation des dependances
KNN-install:
	@echo "Installation des dependances..."
	$(PIP) install -r ../requirements.txt
	@echo "Installation terminee"

# Lancer le modele KNN
KNN-run:
	@echo "Lancement du KNN..."
	$(PYTHON) -m jupyter execute ./notebooks/main.ipynb
	@echo "KNN termine"

# Nettoyer les fichiers generes
KNN-clean:
	@echo "Nettoyage..."
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete
	find . -type f -name ".pytest_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name ".coverage" -delete
	@echo "Nettoyage termine"

# Verifier la qualite du code (flake8)
KNN-lint:
	@echo "Verification du code..."
	$(PYTHON) -m flake8 pneumonia_knn/ --max-line-length=100
	@echo "Verification terminee"

# Creer l'environnement virtuel
venv:
	@echo "Creation de l'environnement virtuel..."
	$(PYTHON) -m venv $(VENV)
	@echo "Environnement cree. Activez-le avec:"
	@echo "   Windows: .venv\\Scripts\\activate"
	@echo "   Linux/Mac: source .venv/bin/activate"

activate:
	@echo "Comment activer l'environnement virtuel..."
	@echo "   Windows: ..\\.venv\\Scripts\\activate.bat"
	@echo "   Linux/Mac: source .venv/bin/activate"

deactivate:
	@echo "Comment desactiver l'environnement virtuel..."
	@echo "   Windows: ..\\.venv\\Scripts\\deactivate"
	@echo "   Linux/Mac: deactivate"