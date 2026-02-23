import numpy as np
import joblib
import matplotlib.pyplot as plt

from datasets import load_dataset
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    classification_report,
    roc_auc_score,
    confusion_matrix,
    roc_curve,
    auc
)
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from skimage.feature import hog
from PIL import Image

# =========================

def load_hf_dataset():
    print("Téléchargement du dataset...")
    return load_dataset("hf-vision/chest-xray-pneumonia")

# =========================

def preprocess_split(split, image_size=(128, 128)):
    features = []
    labels = []

    for example in split:
        img = example["image"].resize(image_size).convert("L")
        img_array = np.array(img)

        hog_features = hog(
            img_array,
            orientations=9,
            pixels_per_cell=(8, 8),
            cells_per_block=(2, 2),
            block_norm="L2-Hys",
            visualize=False
        )

        features.append(hog_features)
        labels.append(example["label"])

    return np.array(features), np.array(labels)

# =========================

def train_model(X_train, y_train):
    model = RandomForestClassifier(
        n_estimators=500,
        max_depth=None,
        min_samples_split=5,
        class_weight="balanced_subsample",
        random_state=42,
        n_jobs=-1
    )
    model.fit(X_train, y_train)
    return model

# =========================

def evaluate(model, X, y, name="Dataset", threshold=0.903):
    proba = model.predict_proba(X)[:, 1]
    y_pred = (proba > threshold).astype(int)

    print(f"\n=== Résultats sur {name} ===")
    print(f"Seuil utilisé: {threshold:.3f}")
    print(classification_report(y, y_pred))
    print("ROC-AUC:", roc_auc_score(y, proba))
    print("Confusion matrix:\n", confusion_matrix(y, y_pred))


def find_best_threshold(model, X, y):
    proba = model.predict_proba(X)[:, 1]
    fpr, tpr, thresholds = roc_curve(y, proba)

    j_scores = tpr - fpr
    best_index = j_scores.argmax()
    best_threshold = thresholds[best_index]

    print(f"\nMeilleur seuil (Youden): {best_threshold:.3f}")
    return best_threshold

def plot_metrics_bar(model, X, y, threshold=0.5):

    proba = model.predict_proba(X)[:, 1]
    y_pred = (proba > threshold).astype(int)

    accuracy = accuracy_score(y, y_pred) * 100
    precision = precision_score(y, y_pred) * 100
    recall = recall_score(y, y_pred) * 100
    f1 = f1_score(y, y_pred) * 100

    scores = [accuracy, precision, recall, f1]
    labels = ["Accuracy", "Precision", "Recall", "F1-score"]

    plt.figure()
    bars = plt.bar(labels, scores)

    plt.ylim(0, 100)
    plt.ylabel("Score (%)")
    plt.title("Model Performance (%)")

    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2,
                 height,
                 f"{height:.1f}%",
                 ha='center', va='bottom')

    plt.show()

# =========================

if __name__ == "__main__":

    dataset = load_hf_dataset()

    print("Préparation des données (HOG features)...")
    X_train, y_train = preprocess_split(dataset["train"])
    X_val, y_val = preprocess_split(dataset["validation"])
    X_test, y_test = preprocess_split(dataset["test"])

    print("Entraînement du modèle Random Forest + HOG...")
    model = train_model(X_train, y_train)

    evaluate(model, X_test, y_test, "Test (Seuil 0.5)", threshold=0.5)

    best_threshold = find_best_threshold(model, X_val, y_val)
    evaluate(model, X_test, y_test, "Test (Seuil Optimal)", threshold=best_threshold)
    plot_metrics_bar(model, X_test, y_test, threshold=best_threshold)

    joblib.dump(model, "random_forest_hog_model.pkl")
    print("\nModèle HOG + Random Forest sauvegardé")