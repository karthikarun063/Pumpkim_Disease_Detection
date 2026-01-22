import os
import cv2
import time
import numpy as np
import matplotlib.pyplot as plt

from skimage.feature import local_binary_pattern
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score
)

# =============================
# CONFIGURATION
# =============================
DATASET_PATH = "dataset"
IMG_SIZE = 128
TEST_SIZE = 0.3
RANDOM_STATE = 42

# =============================
# PREPROCESSING
# =============================
def preprocess(img):
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    img = cv2.GaussianBlur(img, (5, 5), 0)
    return img

def pumpkin_mask(img):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    lower = np.array([30, 40, 40])
    upper = np.array([85, 255, 255])
    return cv2.inRange(hsv, lower, upper)

# =============================
# FEATURE EXTRACTION
# =============================
def extract_features(img):
    mask = pumpkin_mask(img)

    hist = cv2.calcHist([mask], [0], None, [128], [0, 256])
    hist = hist.flatten()

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    lbp = local_binary_pattern(gray, P=8, R=1, method="uniform")
    lbp_hist, _ = np.histogram(lbp, bins=10, range=(0, 10))

    return np.concatenate([hist, lbp_hist])

# =============================
# LOAD DATASET
# =============================
def load_dataset():
    X, y = [], []

    for label, folder in enumerate(["healthy", "diseased"]):
        folder_path = os.path.join(DATASET_PATH, folder)

        for file in os.listdir(folder_path):
            img_path = os.path.join(folder_path, file)
            img = cv2.imread(img_path)

            if img is None:
                continue

            img = preprocess(img)
            features = extract_features(img)

            X.append(features)
            y.append(label)

    return np.array(X), np.array(y)

# =============================
# MODEL PIPELINE FUNCTION
# =============================
def run_model(model_name, model, X_train, X_test, y_train, y_test):
    print(f"[INFO] Training {model_name}...")
    start_time = time.time()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:,1] if hasattr(model, "predict_proba") else None
    end_time = time.time()

    avg_inference_time = (end_time - start_time) / len(X_test) * 1000

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_prob) if y_prob is not None else 0

    cm = confusion_matrix(y_test, y_pred)
    tn, fp, fn, tp = cm.ravel()
    specificity = tn / (tn + fp)
    sensitivity = recall

    traditional_sprays = len(y_test)
    intelligent_sprays = np.sum(y_pred)
    spray_reduction = (1 - intelligent_sprays / traditional_sprays) * 100

    print(f"\n=== {model_name} RESULTS ===")
    print("Accuracy        :", round(accuracy * 100, 2), "%")
    print("Precision       :", round(precision * 100, 2), "%")
    print("Recall          :", round(recall * 100, 2), "%")
    print("Specificity     :", round(specificity * 100, 2), "%")
    print("F1-Score        :", round(f1 * 100, 2), "%")
    print("ROC-AUC Score   :", round(roc_auc, 3))
    print("Avg Inference Time (ms):", round(avg_inference_time, 3))
    print("Spray Reduction (%) :", round(spray_reduction, 2))
    
    return {
        "model": model_name,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "roc_auc": roc_auc,
        "spray_reduction": spray_reduction
    }

# =============================
# MAIN DRIVER
# =============================
X, y = load_dataset()
print("[INFO] Total samples:", len(X))

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Define models
models = [
    ("Logistic Regression", LogisticRegression(max_iter=1000, class_weight='balanced', solver='liblinear')),
    ("SVM Linear", SVC(kernel='linear', class_weight='balanced', probability=True)),
    ("SVM RBF", SVC(kernel='rbf', class_weight='balanced', probability=True))
]

results = []

for name, model in models:
    metrics = run_model(name, model, X_train, X_test, y_train, y_test)
    results.append(metrics)

# =============================
# COMPARISON GRAPHS
# =============================
metrics_names = ["accuracy", "precision", "recall", "f1", "spray_reduction"]
for metric in metrics_names:
    plt.figure(figsize=(6,4))
    values = [r[metric]*100 if metric!="spray_reduction" else r[metric] for r in results]
    names = [r["model"] for r in results]
    plt.bar(names, values, color=["skyblue", "orange", "green"])
    plt.title(f"Comparison of {metric.capitalize()}")
    plt.ylabel(metric.capitalize() + (" (%)" if metric!="spray_reduction" else ""))
    plt.ylim(0, 110)
    plt.tight_layout()
    plt.show()
