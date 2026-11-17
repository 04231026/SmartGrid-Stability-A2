
import os
import json
import numpy as np
import pandas as pd
import tensorflow as tf

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report
)
from sklearn.utils.class_weight import compute_class_weight

import joblib

# ==================================================
# 1. KONFIGURASI
# ==================================================

SEED = 42

np.random.seed(SEED)
tf.random.set_seed(SEED)

DATA_PATH = "data/Data_for_UCI_named.csv"
FEATURE_PATH = "artifacts/ga_selected_features.csv"
ARTIFACT_DIR = "artifacts"

os.makedirs(ARTIFACT_DIR, exist_ok=True)

# ==================================================
# 2. MEMUAT DATASET DAN FITUR GA
# ==================================================

print("\n=== GA + NEURAL NETWORK ===")

df = pd.read_csv(DATA_PATH)

selected_features = pd.read_csv(
    FEATURE_PATH
)["feature"].tolist()

print("\nFitur hasil Genetic Algorithm:")
print(selected_features)

X = df[selected_features].copy()
y = df["stabf"].copy()

encoder = LabelEncoder()
y = encoder.fit_transform(y)

# ==================================================
# 3. MEMBAGI DATASET
# ==================================================

X_train, X_temp, y_train, y_temp = train_test_split(
    X,
    y,
    test_size=0.30,
    random_state=SEED,
    stratify=y
)

X_val, X_test, y_val, y_test = train_test_split(
    X_temp,
    y_temp,
    test_size=0.50,
    random_state=SEED,
    stratify=y_temp
)

print("\nJumlah data:")
print("Training   :", len(X_train))
print("Validation :", len(X_val))
print("Testing    :", len(X_test))

# ==================================================
# 4. NORMALISASI DATA
# ==================================================

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)
X_test_scaled = scaler.transform(X_test)

# ==================================================
# 5. CLASS WEIGHT
# ==================================================

classes = np.unique(y_train)

weights = compute_class_weight(
    class_weight="balanced",
    classes=classes,
    y=y_train
)

class_weights = {
    int(classes[i]): float(weights[i])
    for i in range(len(classes))
}

# ==================================================
# 6. MEMBANGUN MODEL NEURAL NETWORK
# ==================================================

model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(len(selected_features),)),

    tf.keras.layers.Dense(64, activation="relu"),

    tf.keras.layers.Dropout(0.2),

    tf.keras.layers.Dense(32, activation="relu"),

    tf.keras.layers.Dense(1, activation="sigmoid")
])

model.compile(
    optimizer=tf.keras.optimizers.Adam(),
    loss="binary_crossentropy",
    metrics=["accuracy"]
)

model.summary()

# ==================================================
# 7. EARLY STOPPING
# ==================================================

early_stop = tf.keras.callbacks.EarlyStopping(
    monitor="val_loss",
    patience=10,
    restore_best_weights=True
)

# ==================================================
# 8. TRAINING MODEL
# ==================================================

print("\nMemulai training GA + NN...")

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

# ==================================================
# 9. PREDIKSI DATA TEST
# ==================================================

probabilities = model.predict(
    X_test_scaled,
    verbose=0
).ravel()

predictions = (probabilities >= 0.5).astype(int)

# ==================================================
# 10. EVALUASI MODEL
# ==================================================

accuracy = accuracy_score(y_test, predictions)

precision = precision_score(
    y_test,
    predictions,
    zero_division=0
)

recall = recall_score(
    y_test,
    predictions,
    zero_division=0
)

f1 = f1_score(
    y_test,
    predictions,
    zero_division=0
)

roc_auc = roc_auc_score(
    y_test,
    probabilities
)

cm = confusion_matrix(y_test, predictions)

print("\n=== HASIL EVALUASI GA + NN ===")

print(f"Accuracy  : {accuracy * 100:.2f}%")
print(f"Precision : {precision * 100:.2f}%")
print(f"Recall    : {recall * 100:.2f}%")
print(f"F1-Score  : {f1 * 100:.2f}%")
print(f"ROC-AUC   : {roc_auc * 100:.2f}%")

print("\nConfusion Matrix:")
print(cm)

print("\nClassification Report:")
print(
    classification_report(
        y_test,
        predictions,
        target_names=encoder.classes_
    )
)

# ==================================================
# 11. MENYIMPAN MODEL DAN SCALER
# ==================================================

model.save(
    os.path.join(ARTIFACT_DIR, "ga_nn_model.keras")
)

joblib.dump(
    scaler,
    os.path.join(ARTIFACT_DIR, "ga_nn_scaler.pkl")
)

joblib.dump(
    encoder,
    os.path.join(ARTIFACT_DIR, "ga_nn_label_encoder.pkl")
)

# ==================================================
# 12. MENYIMPAN METRIK
# ==================================================

metrics = pd.DataFrame([{
    "Model": "GA + NN",
    "Number_of_Features": len(selected_features),
    "Accuracy": accuracy,
    "Precision": precision,
    "Recall": recall,
    "F1_Score": f1,
    "ROC_AUC": roc_auc
}])

metrics.to_csv(
    os.path.join(ARTIFACT_DIR, "ga_nn_metrics.csv"),
    index=False
)

# ==================================================
# 13. MENYIMPAN CONFUSION MATRIX
# ==================================================

pd.DataFrame(
    cm,
    index=encoder.classes_,
    columns=encoder.classes_
).to_csv(
    os.path.join(ARTIFACT_DIR, "ga_nn_confusion_matrix.csv")
)

# ==================================================
# 14. MENYIMPAN HISTORY TRAINING
# ==================================================

pd.DataFrame(history.history).to_csv(
    os.path.join(ARTIFACT_DIR, "ga_nn_training_history.csv"),
    index=False
)

# ==================================================
# 15. MENYIMPAN RINGKASAN HASIL
# ==================================================

summary = {
    "model": "GA + Neural Network",
    "selected_features": selected_features,
    "number_of_features": len(selected_features),
    "accuracy": float(accuracy),
    "precision": float(precision),
    "recall": float(recall),
    "f1_score": float(f1),
    "roc_auc": float(roc_auc),
    "seed": SEED
}

with open(
    os.path.join(ARTIFACT_DIR, "ga_nn_summary.json"),
    "w"
) as file:
    json.dump(summary, file, indent=4)

print("\n================================")
print("GA + NN BERHASIL DIJALANKAN")
print("================================")

print("\nSeluruh hasil tersimpan di folder artifacts/")
