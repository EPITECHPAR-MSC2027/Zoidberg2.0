import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import joblib
import base64
import io

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from skimage.feature import hog
from flask import Flask, render_template, request
from datasets import load_dataset
from PIL import Image
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
    roc_curve,
)

app = Flask(__name__)

# =========================
# Load models
# =========================

model_pixel = joblib.load("random_forest_model.pkl")
model_hog = joblib.load("random_forest_hog_model.pkl")

# =========================
# Load dataset once
# =========================

dataset = load_dataset("hf-vision/chest-xray-pneumonia")

def preprocess_pixel_dataset(split):
    X, y = [], []
    for example in split:
        img = example["image"].resize((128, 128)).convert("L")
        X.append(np.array(img).flatten())
        y.append(example["label"])
    return np.array(X), np.array(y)

def preprocess_hog_dataset(split):
    X, y = [], []
    for example in split:
        img = example["image"].resize((128, 128)).convert("L")
        img_array = np.array(img)
        features = hog(
            img_array,
            orientations=9,
            pixels_per_cell=(8, 8),
            cells_per_block=(2, 2),
            block_norm="L2-Hys",
            visualize=False
        )
        X.append(features)
        y.append(example["label"])
    return np.array(X), np.array(y)

print("Préparation dataset test...")

X_test_pixel, y_test = preprocess_pixel_dataset(dataset["test"])
X_test_hog, _ = preprocess_hog_dataset(dataset["test"])

# =========================
# Evaluation function
# =========================

def compute_metrics(model, X, y):

    proba = model.predict_proba(X)[:, 1]
    y_pred = (proba > 0.5).astype(int)

    report = classification_report(y, y_pred)
    cm = confusion_matrix(y, y_pred)
    roc_auc = roc_auc_score(y, proba)

    # -------- ROC Curve --------
    fpr, tpr, _ = roc_curve(y, proba)
    fig1 = plt.figure()
    plt.plot(fpr, tpr)
    plt.plot([0, 1], [0, 1], linestyle="--")
    plt.title("ROC Curve")
    roc_img = plot_to_base64(fig1)
    plt.close(fig1)

    # -------- Threshold vs Recall --------
    thresholds = np.linspace(0.1, 0.9, 30)
    recalls = []

    for t in thresholds:
        y_temp = (proba > t).astype(int)
        tp = ((y_temp == 1) & (y == 1)).sum()
        fn = ((y_temp == 0) & (y == 1)).sum()
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        recalls.append(recall)

    fig2 = plt.figure()
    plt.plot(thresholds, recalls)
    plt.xlabel("Threshold")
    plt.ylabel("Recall (Pneumonia)")
    plt.title("Threshold vs Recall")
    thr_img = plot_to_base64(fig2)
    plt.close(fig2)

    # -------- BAR METRICS --------
    accuracy = accuracy_score(y, y_pred) * 100
    precision = precision_score(y, y_pred) * 100
    recall_score_val = recall_score(y, y_pred) * 100
    f1 = f1_score(y, y_pred) * 100

    scores = [accuracy, precision, recall_score_val, f1]
    labels = ["Accuracy", "Precision", "Recall", "F1-score"]

    fig3 = plt.figure()
    bars = plt.bar(labels, scores)

    plt.ylim(0, 100)
    plt.ylabel("Score (%)")
    plt.title("Model Performance (%)")

    for bar in bars:
        height = bar.get_height()
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            height,
            f"{height:.1f}%",
            ha="center",
            va="bottom"
        )

    metrics_img = plot_to_base64(fig3)
    plt.close(fig3)

    return report, cm, roc_auc, roc_img, thr_img, metrics_img

def plot_to_base64(fig):
    img = io.BytesIO()
    fig.savefig(img, format="png")
    img.seek(0)
    return base64.b64encode(img.getvalue()).decode()

print("Calcul métriques...")
metrics_pixel = compute_metrics(model_pixel, X_test_pixel, y_test)
metrics_hog = compute_metrics(model_hog, X_test_hog, y_test)

# =========================
# Preprocessing single image
# =========================

def preprocess_pixel(image):
    img = image.convert("L").resize((128, 128))
    return np.array(img).flatten().reshape(1, -1)

def preprocess_hog(image):
    img = image.convert("L").resize((128, 128))
    img_array = np.array(img)
    features = hog(
        img_array,
        orientations=9,
        pixels_per_cell=(8, 8),
        cells_per_block=(2, 2),
        block_norm="L2-Hys",
        visualize=False
    )
    return features.reshape(1, -1)

# =========================
# Routes
# =========================

@app.route("/")
def home():

    report, cm, roc_auc, roc_img, thr_img, metrics_img = metrics_pixel

    return render_template(
        "dashboard.html",
        report=report,
        cm=cm,
        metrics_img=metrics_img,
        thr_img=thr_img,
        roc_auc=roc_auc,
        roc_img=roc_img,
        selected_model="pixel"
    )

@app.route("/predict", methods=["POST"])
def predict():

    file = request.files["file"]
    model_choice = request.form.get("model_choice", "pixel")

    image = Image.open(file.stream)

    if model_choice == "hog":
        X = preprocess_hog(image)
        model = model_hog
        report, cm, roc_auc, roc_img, thr_img, metrics_img = metrics_hog
    else:
        X = preprocess_pixel(image)
        model = model_pixel
        report, cm, roc_auc, roc_img, thr_img, metrics_img = metrics_pixel

    prediction = model.predict(X)[0]
    probability = model.predict_proba(X)[0][1]
    result = "PNEUMONIA" if prediction == 1 else "NORMAL"

    return render_template(
        "dashboard.html",
        prediction=result,
        probability=round(probability * 100, 2),
        selected_model=model_choice,
        report=report,
        metrics_img=metrics_img,
        thr_img=thr_img,
        cm=cm,
        roc_auc=roc_auc,
        roc_img=roc_img
    )

if __name__ == "__main__":
    app.run(debug=True)