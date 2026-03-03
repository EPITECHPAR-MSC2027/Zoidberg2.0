import numpy as np
import joblib
import matplotlib.pyplot as plt
import os
import cv2

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
from skimage.filters import sobel, laplace
from PIL import Image
from sklearn.preprocessing import StandardScaler
from huggingface_hub import login
from dotenv import load_dotenv
import albumentations as A
load_dotenv()

# =========================

def load_hf_dataset():
    print("Téléchargement du dataset...")
    hf_token = os.getenv("KEY_HUGGING_FACE")
    if not hf_token:
        raise ValueError("La clé API Hugging Face (HF_TOKEN) n'est pas définie!")
    login(token=hf_token)

    print("Loading dataset...")
    dataset = load_dataset("PAR8/chest-xray-pneumonia", token=hf_token)

    print(f"\n  Dataset loaded!")
    print(f"  Train: {len(dataset['train'])} images")
    print(f"  Validation: {len(dataset['validation'])} images")
    print(f"  Test: {len(dataset['test'])} images")

    return dataset

# =========================

# Define medical-safe augmentation pipelines
train_transform = A.Compose([
    A.Resize(128, 128),
    A.Rotate(limit=10, border_mode=cv2.BORDER_CONSTANT, p=0.7),
    A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.1, rotate_limit=0, 
                       p=0.7, border_mode=cv2.BORDER_CONSTANT),
    A.HorizontalFlip(p=0.5),
    A.ElasticTransform(alpha=1, sigma=50, border_mode=cv2.BORDER_CONSTANT, p=0.2),
    A.OneOf([
        A.GridDistortion(num_steps=5, distort_limit=0.2, 
                        border_mode=cv2.BORDER_CONSTANT, p=1.0),
        A.OpticalDistortion(distort_limit=0.3, border_mode=cv2.BORDER_CONSTANT, p=1.0),
    ], p=0.3),
    A.OneOf([
        A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=1.0),
        A.RandomGamma(gamma_limit=(80, 120), p=1.0),
        A.CLAHE(clip_limit=4.0, tile_grid_size=(8, 8), p=1.0),
    ], p=0.9),
    A.OneOf([
        A.GaussianBlur(blur_limit=(3, 7), p=1.0),
        A.MedianBlur(blur_limit=5, p=1.0),
        A.MotionBlur(blur_limit=7, p=1.0),
    ], p=0.4),
])

val_test_transform = A.Compose([
    A.Resize(128, 128),
])

print("✓ Augmentation pipelines loaded")

# =========================

def preprocess_split(split, apply_augmentation=False, image_size=(128, 128)):
    features = []
    labels = []

    for idx, example in enumerate(split):
        if idx % 500 == 0:
            print(f"  Processing image {idx}/{len(split)}...")
        
        # Convert PIL to RGB numpy array
        img = example["image"].convert("RGB")
        img_array = np.array(img)
        
        # Apply augmentation if training
        if apply_augmentation:
            transformed = train_transform(image=img_array)
            img_array = transformed["image"]
        else:
            transformed = val_test_transform(image=img_array)
            img_array = transformed["image"]
        
        # Convert to grayscale for feature extraction
        img_gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY).astype(float) / 255.0

        # HOG features (main feature)
        hog_features = hog(
            img_gray,
            orientations=8,
            pixels_per_cell=(8, 8),
            cells_per_block=(2, 2),
            block_norm="L2-Hys",
            visualize=False
        )

        # Downsampled edge features (sample every 2 pixels)
        edges_sobel = sobel(img_gray)[::2, ::2]
        edges_flat = edges_sobel.flatten()

        # Histogram features (global statistics)
        hist, _ = np.histogram(img_gray, bins=16)
        hist_features = hist / np.sum(hist)  # Normalize

        # Combine features
        combined = np.concatenate([hog_features, edges_flat, hist_features])
        features.append(combined)
        labels.append(example["label"])

    # Normalize features
    features_array = np.array(features)
    print(f"  Feature shape: {features_array.shape}")
    scaler = StandardScaler()
    features_normalized = scaler.fit_transform(features_array)
    
    return features_normalized, np.array(labels)

# =========================

def train_model(X_train, y_train):
    # Optimized hyperparameters
    model = RandomForestClassifier(
        n_estimators=1000,
        max_depth=25,
        min_samples_split=4,
        min_samples_leaf=2,
        class_weight="balanced_subsample",
        criterion="gini",
        random_state=42,
        n_jobs=-1,
        verbose=1
    )
    model.fit(X_train, y_train)
    print("\n✓ Modèle entraîné avec hyperparamètres optimisés")
    return model

# =========================

def evaluate(model, X, y, name="Dataset", threshold=0.903):
    proba_full = model.predict_proba(X)
    y_pred = np.argmax(proba_full, axis=1)

    print(f"\n=== Résultats sur {name} ===")
    print(f"Seuil utilisé: {threshold:.3f}")
    print(classification_report(y, y_pred))
    
    # ROC-AUC pour multi-classe si > 2 classes
    n_classes = len(np.unique(y))
    if n_classes > 2:
        print("ROC-AUC:", roc_auc_score(y, proba_full, multi_class='ovr'))
    else:
        print("ROC-AUC:", roc_auc_score(y, proba_full[:, 1]))
    print("Confusion matrix:\n", confusion_matrix(y, y_pred))


def find_best_threshold(model, X, y):
    proba_full = model.predict_proba(X)
    
    # Pour binaire: utiliser seuil sur classe 1
    n_classes = len(np.unique(y))
    if n_classes == 2:
        proba = proba_full[:, 1]
        fpr, tpr, thresholds = roc_curve(y, proba)
        j_scores = tpr - fpr
        best_index = j_scores.argmax()
        best_threshold = thresholds[best_index]
        print(f"\nMeilleur seuil (Youden): {best_threshold:.3f}")
        return best_threshold
    else:
        print(f"\nDéjà optimal pour {n_classes} classes (utilise argmax)")
        return 0.5

def plot_metrics_bar(model, X, y, threshold=0.5):

    proba_full = model.predict_proba(X)
    y_pred = np.argmax(proba_full, axis=1)

    accuracy = accuracy_score(y, y_pred) * 100
    precision = precision_score(y, y_pred, average='weighted') * 100
    recall = recall_score(y, y_pred, average='weighted') * 100
    f1 = f1_score(y, y_pred, average='weighted') * 100

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

    print("\nPréparation des données avec augmentation on-the-fly...")
    print("Train set (avec augmentation):")
    X_train, y_train = preprocess_split(dataset["train"], apply_augmentation=True)
    
    print("Validation set:")
    X_val, y_val = preprocess_split(dataset["validation"], apply_augmentation=False)
    
    print("Test set:")
    X_test, y_test = preprocess_split(dataset["test"], apply_augmentation=False)

    print("\nEntraînement du modèle Random Forest + HOG + Augmentation...")
    model = train_model(X_train, y_train)

    print("\n" + "="*60)
    evaluate(model, X_test, y_test, "Test (Seuil 0.5)", threshold=0.5)

    best_threshold = find_best_threshold(model, X_val, y_val)
    evaluate(model, X_test, y_test, "Test (Seuil Optimal)", threshold=best_threshold)
    plot_metrics_bar(model, X_test, y_test, threshold=best_threshold)

    joblib.dump(model, "random_forest_hog_model_augmented.pkl")
    print("\n✓ Modèle HOG + Random Forest + Augmentation sauvegardé")