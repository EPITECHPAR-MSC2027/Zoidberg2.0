.PHONY: help install run test clean lint format

# Variables
PYTHON := python
PIP := pip
VENV := .venv

help:
	@echo "╔════════════════════════════════════════════════════════════╗"
	@echo "║         T-DEV-810 - Analyse X-ray KNN                      ║"
	@echo "╚════════════════════════════════════════════════════════════╝"
	@echo ""
	@echo "Commandes disponibles:"
	@echo ""
	@echo "  make install      - Installer les dépendances"
	@echo "  make run          - Lancer le projet principal"
	@echo "  make run-knn      - Lancer le modèle KNN"
	@echo "  make test         - Lancer les tests"
	@echo "  make clean        - Nettoyer les fichiers générés"
	@echo "  make lint         - Vérifier la qualité du code"
	@echo "  make format       - Formater le code"
	@echo "  make help         - Afficher cette aide"
	@echo ""

# Installation des dépendances
install:
	@echo "📦 Installation des dépendances..."
	$(PIP) install -r requirements.txt
	@echo "✅ Installation terminée"

# Lancer le projet principal
run:
	@echo "🚀 Lancement du projet..."
	$(PYTHON) main.py

# Lancer le modèle KNN
run-knn:
	@echo "🤖 Lancement du KNN..."
	$(PYTHON) src/models/KNN/knn.py

# Lancer les tests
test:
	@echo "🧪 Lancement des tests..."
	$(PYTHON) -m pytest tests/ -v

# Nettoyer les fichiers générés
clean:
	@echo "🧹 Nettoyage..."
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete
	find . -type f -name ".pytest_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name ".coverage" -delete
	@echo "✅ Nettoyage terminé"

# Vérifier la qualité du code (flake8)
lint:
	@echo "🔍 Vérification du code..."
	$(PYTHON) -m flake8 src/ --max-line-length=100
	@echo "✅ Vérification terminée"

# Formater le code (black)
format:
	@echo "📝 Formatage du code..."
	$(PYTHON) -m black src/ --line-length=100
	@echo "✅ Formatage terminé"

# Créer l'environnement virtuel
venv:
	@echo "📦 Création de l'environnement virtuel..."
	$(PYTHON) -m venv $(VENV)
	@echo "✅ Environnement créé. Activez-le avec:"
	@echo "   Windows: .venv\\Scripts\\activate"
	@echo "   Linux/Mac: source .venv/bin/activate"

# Afficher les dépendances
freeze:
	@echo "📋 Dépendances actuelles:"
	$(PIP) freeze

# Exécuter toutes les tâches importantes
all: clean lint test run
	@echo "✅ Toutes les tâches terminées !"