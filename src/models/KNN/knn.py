from src.data import (
    setup_huggingface_auth,
    load_chest_xray_dataset,
)
from src.preprocessing import (
    dataset_to_arrays,
    standardize_data
)
from src.models.KNN.dimensionality.classic import (classic_KNN)
from src.models.KNN.dimensionality.pca import (implementation_with_PCA)
from src.models.KNN.dimensionality.lda import (implementation_with_LDA)

dataset_name = "PAR8/chest-xray-pneumonia"

def launchKNN():
    print("""
      ╔════════════════════════════════════════════════════════════╗
      ║         T-DEV-810 - Analyse X-ray KNN                      ║
      ╚════════════════════════════════════════════════════════════╝""")
    # Authentification et charger le dataset
    print(f"🔐 Tentative Autentification Hugging Face   ...")
    setup_huggingface_auth()
    print("✅ Authentification Hugging Face réussie")

    print(f"📊 Chargement du dataset : {dataset_name}   ...")
    dataset = load_chest_xray_dataset(dataset_name)
    print("✅ Chargement du dataset : Réussi")

    # Utiliser les splits
    train_data = dataset['train']
    val_data = dataset['validation']
    test_data = dataset['test']

    print("🔧 Processing (Redimention, Converstion et Standardisation) en cours...")
    # Convertir images en arrays numpy
    X_train, y_train = dataset_to_arrays(train_data)
    X_test, y_test = dataset_to_arrays(test_data)

    # Standardisation
    X_train_scaled, X_test_scaled = standardize_data(X_train, X_test)

    print("✅ Processing : Réussi")

    # KNN without implementation
    print("🤖 Entraînement du modèle KNN   ...")
    result_knn = classic_KNN(X_train_scaled, y_train, X_test_scaled, y_test)
    print(f"Accuracy: {result_knn}")
    print("✅ KNN entraîné")

    # KNN with PCA implementation
    print("📉 Réduction avec PCA...")
    result_pca = implementation_with_PCA(X_train_scaled,X_test_scaled, y_train, y_test)
    print(f"Accuracy: {result_pca}")
    print("✅ PCA appliquée")

    # KNN with LDA implementation
    print("📉 Réduction avec LDA...")
    result_lda = implementation_with_LDA(y_train, X_train_scaled, X_test_scaled, y_test)
    print(f"Accuracy: {result_lda}")
    print("✅ LDA appliquée")


if __name__ == "__main__":
    # Testing
    print("""
      ╔════════════════════════════════════════════════════════════╗
      ║         TESTING : T-DEV-810 - Analyse X-ray KNN                      ║
      ╚════════════════════════════════════════════════════════════╝""")
    launchKNN()