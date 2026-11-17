
import os
import json
import random
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
    confusion_matrix
)
from sklearn.utils.class_weight import compute_class_weight

import joblib

# ==================================================
# 1. KONFIGURASI
# ==================================================

SEED = 42

POPULATION_SIZE = 8
GENERATIONS = 8

W = 0.7
C1 = 1.5
C2 = 1.5

random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

DATA_PATH = "data/Data_for_UCI_named.csv"
ARTIFACT_DIR = "artifacts"

os.makedirs(ARTIFACT_DIR, exist_ok=True)

FEATURES = [
    "tau1", "tau2", "tau3", "tau4",
    "p1", "p2", "p3", "p4",
    "g1", "g2", "g3", "g4"
]

# ==================================================
# 2. MEMUAT DATASET
# ==================================================

print("\n=== PARTICLE SWARM OPTIMIZATION + NN ===")

df = pd.read_csv(DATA_PATH)

X = df[FEATURES].copy()
y = df["stabf"].copy()

encoder = LabelEncoder()
y = encoder.fit_transform(y)

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

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)
X_test_scaled = scaler.transform(X_test)

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

print("Data training:", len(X_train))
print("Data validation:", len(X_val))
print("Data testing:", len(X_test))

# ==================================================
# 3. RENTANG PARAMETER PSO
# ==================================================

# Posisi partikel:
# [jumlah neuron layer 1,
#  jumlah neuron layer 2,
#  learning rate,
#  dropout]

LOWER = np.array([32, 16, 0.0001, 0.10])
UPPER = np.array([128, 64, 0.0100, 0.40])

def decode_particle(position):

    return {
        "neurons_1": int(round(position[0] / 8) * 8),
        "neurons_2": int(round(position[1] / 8) * 8),
        "learning_rate": float(position[2]),
        "dropout": float(position[3])
    }

# ==================================================
# 4. FUNGSI EVALUASI FITNESS
# ==================================================

fitness_cache = {}

def evaluate_particle(position):

    params = decode_particle(position)

    key = tuple(round(float(x), 6) for x in position)

    if key in fitness_cache:
        return fitness_cache[key]

    tf.keras.backend.clear_session()
    tf.keras.utils.set_random_seed(SEED)

    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(len(FEATURES),)),

        tf.keras.layers.Dense(
            params["neurons_1"],
            activation="relu"
        ),

        tf.keras.layers.Dropout(params["dropout"]),

        tf.keras.layers.Dense(
            params["neurons_2"],
            activation="relu"
        ),

        tf.keras.layers.Dense(1, activation="sigmoid")
    ])

    model.compile(
        optimizer=tf.keras.optimizers.Adam(
            learning_rate=params["learning_rate"]
        ),
        loss="binary_crossentropy",
        metrics=["accuracy"]
    )

    early_stop = tf.keras.callbacks.EarlyStopping(
        monitor="val_loss",
        patience=5,
        restore_best_weights=True
    )

    history = model.fit(
        X_train_scaled,
        y_train,
        validation_data=(X_val_scaled, y_val),
        epochs=40,
        batch_size=32,
        class_weight=class_weights,
        callbacks=[early_stop],
        verbose=0
    )

    probabilities = model.predict(
        X_val_scaled,
        verbose=0
    ).ravel()

    predictions = (probabilities >= 0.5).astype(int)

    fitness = f1_score(
        y_val,
        predictions,
        zero_division=0
    )

    fitness_cache[key] = fitness

    return fitness

# ==================================================
# 5. INISIALISASI PARTIKEL
# ==================================================

positions = np.random.uniform(
    LOWER,
    UPPER,
    size=(POPULATION_SIZE, len(LOWER))
)

velocities = np.zeros_like(positions)

personal_best_positions = positions.copy()

personal_best_scores = np.array([
    evaluate_particle(p)
    for p in positions
])

best_index = np.argmax(personal_best_scores)

global_best_position = personal_best_positions[
    best_index
].copy()

global_best_score = personal_best_scores[best_index]

history = []

print("\nInisialisasi PSO selesai.")

# ==================================================
# 6. PROSES ITERASI PSO
# ==================================================

for generation in range(GENERATIONS):

    for i in range(POPULATION_SIZE):

        r1 = np.random.random(len(LOWER))
        r2 = np.random.random(len(LOWER))

        velocities[i] = (
            W * velocities[i]
            + C1 * r1 * (
                personal_best_positions[i] - positions[i]
            )
            + C2 * r2 * (
                global_best_position - positions[i]
            )
        )

        positions[i] += velocities[i]

        positions[i] = np.clip(
            positions[i],
            LOWER,
            UPPER
        )

        score = evaluate_particle(positions[i])

        if score > personal_best_scores[i]:

            personal_best_scores[i] = score

            personal_best_positions[i] = positions[i].copy()

        if score > global_best_score:

            global_best_score = score

            global_best_position = positions[i].copy()

    best_params = decode_particle(global_best_position)

    history.append({
        "generation": generation + 1,
        "best_fitness": float(global_best_score),
        "neurons_1": best_params["neurons_1"],
        "neurons_2": best_params["neurons_2"],
        "learning_rate": best_params["learning_rate"],
        "dropout": best_params["dropout"]
    })

    print(
        f"Iterasi {generation + 1:02d} | "
        f"Fitness: {global_best_score:.5f} | "
        f"Parameter: {best_params}"
    )

# ==================================================
# 7. MENYIMPAN HASIL OPTIMASI
# ==================================================

best_params = decode_particle(global_best_position)

pd.DataFrame(history).to_csv(
    os.path.join(ARTIFACT_DIR, "pso_fitness_history.csv"),
    index=False
)

summary = {
    "model": "PSO + Neural Network",
    "best_fitness": float(global_best_score),
    "best_parameters": best_params,
    "population_size": POPULATION_SIZE,
    "generations": GENERATIONS,
    "seed": SEED
}

with open(
    os.path.join(ARTIFACT_DIR, "pso_summary.json"),
    "w"
) as file:
    json.dump(summary, file, indent=4)

# ==================================================
# 8. TRAINING FINAL MODEL
# ==================================================

print("\nMemulai training final PSO + NN...")

tf.keras.backend.clear_session()
tf.keras.utils.set_random_seed(SEED)

final_model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(len(FEATURES),)),

    tf.keras.layers.Dense(
        best_params["neurons_1"],
        activation="relu"
    ),

    tf.keras.layers.Dropout(best_params["dropout"]),

    tf.keras.layers.Dense(
        best_params["neurons_2"],
        activation="relu"
    ),

    tf.keras.layers.Dense(1, activation="sigmoid")
])

final_model.compile(
    optimizer=tf.keras.optimizers.Adam(
        learning_rate=best_params["learning_rate"]
    ),
    loss="binary_crossentropy",
    metrics=["accuracy"]
)

early_stop = tf.keras.callbacks.EarlyStopping(
    monitor="val_loss",
    patience=10,
    restore_best_weights=True
)

final_history = final_model.fit(
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
# 9. EVALUASI DATA TEST
# ==================================================

probabilities = final_model.predict(
    X_test_scaled,
    verbose=0
).ravel()

predictions = (probabilities >= 0.5).astype(int)

accuracy = accuracy_score(y_test, predictions)

precision = precision_score(
    y_test, predictions, zero_division=0
)

recall = recall_score(
    y_test, predictions, zero_division=0
)

f1 = f1_score(
    y_test, predictions, zero_division=0
)

roc_auc = roc_auc_score(y_test, probabilities)

cm = confusion_matrix(y_test, predictions)

print("\n=== HASIL PSO + NN ===")
print(f"Accuracy  : {accuracy * 100:.2f}%")
print(f"Precision : {precision * 100:.2f}%")
print(f"Recall    : {recall * 100:.2f}%")
print(f"F1-Score  : {f1 * 100:.2f}%")
print(f"ROC-AUC   : {roc_auc * 100:.2f}%")

print("\nConfusion Matrix:")
print(cm)

# ==================================================
# 10. MENYIMPAN MODEL DAN ARTIFACT
# ==================================================

final_model.save(
    os.path.join(ARTIFACT_DIR, "pso_nn_model.keras")
)

joblib.dump(
    scaler,
    os.path.join(ARTIFACT_DIR, "pso_nn_scaler.pkl")
)

joblib.dump(
    encoder,
    os.path.join(ARTIFACT_DIR, "pso_nn_label_encoder.pkl")
)

pd.DataFrame([{
    "Model": "PSO + NN",
    "Number_of_Features": len(FEATURES),
    "Accuracy": accuracy,
    "Precision": precision,
    "Recall": recall,
    "F1_Score": f1,
    "ROC_AUC": roc_auc
}]).to_csv(
    os.path.join(ARTIFACT_DIR, "pso_nn_metrics.csv"),
    index=False
)

pd.DataFrame(
    cm,
    index=encoder.classes_,
    columns=encoder.classes_
).to_csv(
    os.path.join(ARTIFACT_DIR, "pso_nn_confusion_matrix.csv")
)

pd.DataFrame(final_history.history).to_csv(
    os.path.join(ARTIFACT_DIR, "pso_nn_training_history.csv"),
    index=False
)

summary.update({
    "accuracy": float(accuracy),
    "precision": float(precision),
    "recall": float(recall),
    "f1_score": float(f1),
    "roc_auc": float(roc_auc)
})

with open(
    os.path.join(ARTIFACT_DIR, "pso_nn_summary.json"),
    "w"
) as file:
    json.dump(summary, file, indent=4)

print("\n================================")
print("PSO + NN BERHASIL DIJALANKAN")
print("================================")
print("Seluruh hasil tersimpan di artifacts/")
