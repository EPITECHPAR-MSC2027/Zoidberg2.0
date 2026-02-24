import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)
from sklearn.decomposition import PCA
import joblib


# ==========================================================
# 1. CONFIGURATION
# ==========================================================

DATASET_DIR = "chest_Xray"
IMG_SIZE = (64, 64)
CLASS_NAMES = ["NORMAL", "PNEUMONIA"]


# ==========================================================
# 2. DATA LOADING + PREPROCESSING
# ==========================================================

def load_split(split_dir):
    """
    Charge un dossier (train/val/test),
    convertit les images en niveaux de gris,
    redimensionne, normalise et transforme en vecteurs.
    """
    X, y = [], []
    counts = {cls: 0 for cls in CLASS_NAMES}
    errors = 0

    for label, cls in enumerate(CLASS_NAMES):
        cls_dir = os.path.join(split_dir, cls)

        for fname in os.listdir(cls_dir):
            if fname.startswith("."):
                continue

            fpath = os.path.join(cls_dir, fname)

            try:
                img = Image.open(fpath).convert("L")
                img = img.resize(IMG_SIZE)

                arr = np.asarray(img, dtype=np.float32) / 255.0
                X.append(arr.flatten())
                y.append(label)
                counts[cls] += 1

            except Exception:
                errors += 1

    X = np.array(X, dtype=np.float32)
    y = np.array(y, dtype=np.int64)

    print(f"\nLoaded: {split_dir}")
    print("Counts:", counts, "| errors:", errors)
    print("Shape:", X.shape)

    return X, y


# Chargement des splits
train_dir = os.path.join(DATASET_DIR, "train")
val_dir   = os.path.join(DATASET_DIR, "val")
test_dir  = os.path.join(DATASET_DIR, "test")

X_train, y_train = load_split(train_dir)
X_val, y_val     = load_split(val_dir)
X_test, y_test   = load_split(test_dir)


# ==========================================================
# 3. NORMALISATION
# ==========================================================

scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_val_s   = scaler.transform(X_val)
X_test_s  = scaler.transform(X_test)


# ==========================================================
# 4. ENTRAINEMENT – LOGISTIC REGRESSION
# ==========================================================

print("\nTraining Logistic Regression...")

model = LogisticRegression(
    max_iter=5000,
    solver="saga",
    class_weight="balanced",
    n_jobs=-1
)

model.fit(X_train_s, y_train)


# ==========================================================
# 5. EVALUATION COMPLETE
# ==========================================================

def evaluate_model(name, model, Xs, y_true):

    y_pred = model.predict(Xs)
    y_proba = model.predict_proba(Xs)[:, 1]

    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    auc = roc_auc_score(y_true, y_proba)

    print(f"\n===== {name} =====")
    print("Confusion Matrix:\n", confusion_matrix(y_true, y_pred))
    print("\nClassification Report:\n",
          classification_report(y_true, y_pred, target_names=CLASS_NAMES))

    print(f"Accuracy : {acc:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall   : {rec:.4f}")
    print(f"F1-score : {f1:.4f}")
    print(f"ROC-AUC  : {auc:.4f}")

    return {
        "Accuracy": acc,
        "Precision": prec,
        "Recall": rec,
        "F1-score": f1,
        "ROC-AUC": auc
    }


scores_val = evaluate_model("VALIDATION", model, X_val_s, y_val)
scores_test = evaluate_model("TEST", model, X_test_s, y_test)


# ==========================================================
# 6. GRAPH BAR (%)
# ==========================================================

def plot_scores(scores, title):

    labels = list(scores.keys())
    values = [scores[k] * 100 for k in labels]

    plt.figure(figsize=(8,5))
    bars = plt.bar(labels, values)

    plt.ylim(0, 100)
    plt.ylabel("Score (%)")
    plt.title(title)
    plt.grid(axis="y", alpha=0.3)

    for bar, val in zip(bars, values):
        plt.text(bar.get_x() + bar.get_width()/2,
                 val + 1,
                 f"{val:.1f}%",
                 ha="center")

    plt.tight_layout()
    plt.show()


plot_scores(scores_test, "Logistic Regression - Test Set (%)")


# ==========================================================
# 7. VERSION AVEC PCA
# ==========================================================

print("\nApplying PCA...")

pca = PCA(n_components=0.95, random_state=42)

X_train_pca = pca.fit_transform(X_train_s)
X_val_pca   = pca.transform(X_val_s)
X_test_pca  = pca.transform(X_test_s)

print("Dimensions before:", X_train_s.shape[1])
print("Dimensions after PCA:", X_train_pca.shape[1])

model_pca = LogisticRegression(
    max_iter=5000,
    solver="saga",
    class_weight="balanced",
    n_jobs=-1
)

model_pca.fit(X_train_pca, y_train)

scores_test_pca = evaluate_model("TEST (PCA)", model_pca, X_test_pca, y_test)

plot_scores(scores_test_pca, "Logistic Regression + PCA - Test (%)")


# ==========================================================
# 8. SAUVEGARDE
# ==========================================================

joblib.dump(model, "logreg_pneumonia.pkl")
joblib.dump(model_pca, "logreg_pneumonia_pca.pkl")
joblib.dump(scaler, "scaler.pkl")
joblib.dump(pca, "pca.pkl")

print("\nModels and preprocessing saved successfully!")