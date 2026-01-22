import os
import cv2
import time
import numpy as np
import matplotlib.pyplot as plt

from skimage.feature import local_binary_pattern
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    accuracy_score
)

# =============================
# CONFIGURATION
# =============================
DATASET_PATH = "dataset"   # dataset/healthy , dataset/diseased
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

    # Color histogram (leaf region only)
    hist = cv2.calcHist([mask], [0], None, [128], [0, 256])
    hist = hist.flatten()

    # Texture features (LBP)
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
# MAIN PIPELINE
# =============================
print("[INFO] Loading dataset...")
X, y = load_dataset()
print("[INFO] Total samples:", len(X))

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=TEST_SIZE,
    random_state=RANDOM_STATE,
    stratify=y
)

print("[INFO] Training SVM model...")
model = SVC(
    kernel='linear',
    class_weight='balanced',
    probability=True
)
model.fit(X_train, y_train)

# =============================
# INFERENCE & TIMING
# =============================
start_time = time.time()
y_pred = model.predict(X_test)
end_time = time.time()

avg_inference_time = (end_time - start_time) / len(X_test) * 1000

# =============================
# PERFORMANCE METRICS
# =============================
accuracy = accuracy_score(y_test, y_pred)
cm = confusion_matrix(y_test, y_pred)

print("\n=== CLASSIFICATION REPORT ===")
print(classification_report(y_test, y_pred))
print("Accuracy:", round(accuracy * 100, 2), "%")
print("Average Inference Time:", round(avg_inference_time, 2), "ms")

# =============================
# CONFUSION MATRIX PLOT (WITH VALUES)
# =============================
plt.figure(figsize=(6, 5))
plt.imshow(cm)
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.colorbar()

classes = ["Healthy", "Diseased"]
plt.xticks([0, 1], classes)
plt.yticks([0, 1], classes)

# Annotate values inside the matrix
for i in range(cm.shape[0]):
    for j in range(cm.shape[1]):
        plt.text(
            j, i,
            cm[i, j],
            ha="center",
            va="center",
            color="white" if cm[i, j] > cm.max() / 2 else "black",
            fontsize=12,
            fontweight="bold"
        )

plt.tight_layout()
plt.show()

# =============================
# SPRAY SIMULATION
# =============================
traditional_sprays = len(y_test)        # Spray all plants
intelligent_sprays = np.sum(y_pred)     # Spray only diseased plants

spray_reduction = (1 - intelligent_sprays / traditional_sprays) * 100

print("\n=== SPRAY SIMULATION RESULTS ===")
print("Traditional Sprays:", traditional_sprays)
print("Intelligent Sprays:", intelligent_sprays)
print("Spray Reduction (%):", round(spray_reduction, 2))

# =============================
# BAR CHART COMPARISON
# =============================
plt.figure(figsize=(6, 4))
plt.bar(
    ["Traditional Spraying", "Intelligent Spraying"],
    [traditional_sprays, intelligent_sprays]
)
plt.title("Pesticide Spray Comparison")
plt.ylabel("Number of Sprays")
plt.tight_layout()
plt.show()
