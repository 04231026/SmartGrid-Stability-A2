
from pathlib import Path
import random
import joblib
import numpy as np
import pandas as pd
import tensorflow as tf

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
    roc_auc_score
)

# =====================================
# 1. KONFIGURASI
# =====================================

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "data" / "Data_for_UCI_named.csv"
OUTPUT_DIR = BASE_DIR / "artifacts"
OUTPUT_DIR.mkdir(exist_ok=True)

FEATURES = [
    "tau1", "tau2", "tau3", "tau4",
    "p1", "p2", "p3", "p4",
    "g1", "g2", "g3", "g4"
]

TARGET = "stabf"

# =====================================
# 2. MEMBACA DATASET
# =====================================

print("\nMEMBACA DATASET...")

df = pd.read_csv(DATA_PATH)

X = df[FEATURES].copy()
y_raw = df[TARGET].copy()

encoder = LabelEncoder()
y = encoder.fit_transform(y_raw)

print("Jumlah data:", len(df))
print("Fitur:", len(FEATURES))
print("Kelas:", list(encoder.classes_))

# =====================================
# 3. PEMBAGIAN DATA 70:15:15
# =====================================

X_train, X_temp, y_train, y_temp = train_test_split(
    X, y,
    test_size=0.30,
    random_state=SEED,
    stratify=y
)

X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp,
    test_size=0.50,
    random_state=SEED,
    stratify=y_temp
)

print("\nPEMBAGIAN DATA")
print("Training:", len(X_train))
print("Validation:", len(X_val))
print("Testing:", len(X_test))

# =====================================
# 4. NORMALISASI DATA
# =====================================

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)
X_test_scaled = scaler.transform(X_test)

joblib.dump(scaler, OUTPUT_DIR / "baseline_scaler.pkl")
joblib.dump(encoder, OUTPUT_DIR / "label_encoder.pkl")

# =====================================
# 5. CLASS WEIGHT
# =====================================

classes = np.unique(y_train)

weights = compute_class_weight(
    class_weight="balanced",
    classes=classes,
    y=y_train
)

class_weights = dict(zip(classes, weights))

print("\nCLASS WEIGHTS:", class_weights)

# =====================================
# 6. MEMBANGUN NEURAL NETWORK
# =====================================

model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(len(FEATURES),)),
    tf.keras.layers.Dense(64, activation="relu"),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(32, activation="relu"),
    tf.keras.layers.Dense(1, activation="sigmoid")
])

model.compile(
    optimizer="adam",
    loss="binary_crossentropy",
    metrics=["accuracy"]
)

model.summary()

# =====================================
# 7. TRAINING
# =====================================

print("\nMEMULAI TRAINING...")

early_stop = tf.keras.callbacks.EarlyStopping(
    monitor="val_loss",
    patience=10,
    restore_best_weights=True
)

history = model.fit(
    X_train_scaled,
    y_train,
    validation_data=(X_val_scaled, y_val),
    epochs=100,
    batch_size=32,
    class_weight=class_weights,
    callbacks=[early_stop],
    verbose=1
)

# =====================================
# 8. EVALUASI DATA TEST
# =====================================

probabilities = model.predict(
    X_test_scaled
).ravel()

predictions = (probabilities >= 0.5).astype(int)

accuracy = accuracy_score(y_test, predictions)
precision = precision_score(y_test, predictions, zero_division=0)
recall = recall_score(y_test, predictions, zero_division=0)
f1 = f1_score(y_test, predictions, zero_division=0)
auc = roc_auc_score(y_test, probabilities)

print("\n========== HASIL BASELINE NN ==========")
print(f"Accuracy  : {accuracy:.4f}")
print(f"Precision : {precision:.4f}")
print(f"Recall    : {recall:.4f}")
print(f"F1-Score  : {f1:.4f}")
print(f"ROC-AUC   : {auc:.4f}")

print("\nCLASSIFICATION REPORT")
print(classification_report(
    y_test,
    predictions,
    target_names=encoder.classes_
))

print("\nCONFUSION MATRIX")
print(confusion_matrix(y_test, predictions))

# =====================================
# 9. SIMPAN MODEL DAN HASIL
# =====================================

model.save(OUTPUT_DIR / "baseline_model.keras")

results = pd.DataFrame([{
    "accuracy": accuracy,
    "precision": precision,
    "recall": recall,
    "f1_score": f1,
    "roc_auc": auc
}])

results.to_csv(
    OUTPUT_DIR / "baseline_metrics.csv",
    index=False
)

pd.DataFrame(history.history).to_csv(
    OUTPUT_DIR / "baseline_training_history.csv",
    index=False
)

print("\n======================================")
print("BASELINE NEURAL NETWORK SELESAI")
print("Model dan hasil tersimpan di artifacts/")
print("======================================")
