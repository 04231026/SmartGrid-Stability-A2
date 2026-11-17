
import os
import json
import random
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
    roc_auc_score,
    confusion_matrix
)

import joblib


# ==========================================================
# 1. KONFIGURASI
# ==========================================================

SEED = 42

DATA_PATH = "data/Data_for_UCI_named.csv"
FEATURE_PATH = "artifacts/ga_selected_features.csv"

ARTIFACT_DIR = "artifacts"
os.makedirs(ARTIFACT_DIR, exist_ok=True)

np.random.seed(SEED)
random.seed(SEED)
tf.keras.utils.set_random_seed(SEED)

print("=" * 60)
print("MODEL GA + PSO + NEURAL NETWORK")
print("=" * 60)


# ==========================================================
# 2. MEMUAT DATA DAN FITUR HASIL GA
# ==========================================================

df = pd.read_csv(DATA_PATH)

feature_df = pd.read_csv(FEATURE_PATH)

if "feature" in feature_df.columns:
    selected_features = feature_df["feature"].tolist()
else:
    selected_features = feature_df.iloc[:, 0].tolist()

selected_features = [
    feature for feature in selected_features
    if feature in df.columns
]

if len(selected_features) == 0:
    raise ValueError("Fitur hasil GA tidak ditemukan!")

X = df[selected_features].copy()
y = df["stabf"].copy()

print("\nFitur hasil seleksi GA:")
print(selected_features)

print("\nJumlah fitur:", len(selected_features))


# ==========================================================
# 3. ENCODING DAN PEMBAGIAN DATA
# ==========================================================

label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

X_train, X_temp, y_train, y_temp = train_test_split(
    X,
    y_encoded,
    test_size=0.30,
    random_state=SEED,
    stratify=y_encoded
)

X_val, X_test, y_val, y_test = train_test_split(
    X_temp,
    y_temp,
    test_size=0.50,
    random_state=SEED,
    stratify=y_temp
)

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)
X_test_scaled = scaler.transform(X_test)

print("\nPembagian data:")
print("Training :", len(X_train))
print("Validasi :", len(X_val))
print("Testing  :", len(X_test))


# ==========================================================
# 4. CLASS WEIGHT
# ==========================================================

classes = np.unique(y_train)

weights = compute_class_weight(
    class_weight="balanced",
    classes=classes,
    y=y_train
)

class_weights = {
    int(cls): float(weight)
    for cls, weight in zip(classes, weights)
}


# ==========================================================
# 5. FUNGSI MEMBANGUN NEURAL NETWORK
# ==========================================================

def build_model(neurons_1, neurons_2, learning_rate, dropout):

    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(X_train_scaled.shape[1],)),

        tf.keras.layers.Dense(
            neurons_1,
            activation="relu"
        ),

        tf.keras.layers.Dropout(dropout),

        tf.keras.layers.Dense(
            neurons_2,
            activation="relu"
        ),

        tf.keras.layers.Dense(
            1,
            activation="sigmoid"
        )
    ])

    optimizer = tf.keras.optimizers.Adam(
        learning_rate=learning_rate
    )

    model.compile(
        optimizer=optimizer,
        loss="binary_crossentropy",
        metrics=["accuracy"]
    )

    return model


# ==========================================================
# 6. DECODE PARTICLE PSO
# ==========================================================

def decode_particle(position):

    neurons_1 = int(round(position[0] / 8) * 8)
    neurons_2 = int(round(position[1] / 8) * 8)

    neurons_1 = int(np.clip(neurons_1, 32, 128))
    neurons_2 = int(np.clip(neurons_2, 16, 64))

    learning_rate = float(position[2])
    dropout = float(position[3])

    return (
        neurons_1,
        neurons_2,
        learning_rate,
        dropout
    )


# ==========================================================
# 7. FUNGSI FITNESS PSO
# ==========================================================

fitness_cache = {}


def evaluate_particle(position):

    parameters = decode_particle(position)

    cache_key = tuple(
        round(float(value), 6)
        for value in parameters
    )

    if cache_key in fitness_cache:
        return fitness_cache[cache_key]

    tf.keras.backend.clear_session()
    tf.keras.utils.set_random_seed(SEED)

    model = build_model(*parameters)

    early_stop = tf.keras.callbacks.EarlyStopping(
        monitor="val_loss",
        patience=5,
        restore_best_weights=True
    )

    model.fit(
        X_train_scaled,
        y_train,
        validation_data=(X_val_scaled, y_val),
        epochs=40,
        batch_size=32,
        class_weight=class_weights,
        callbacks=[early_stop],
        verbose=0
    )

    predictions = model.predict(
        X_val_scaled,
        verbose=0
    ).ravel()

    predicted_classes = (
        predictions >= 0.5
    ).astype(int)

    fitness = f1_score(
        y_val,
        predicted_classes,
        zero_division=0
    )

    fitness_cache[cache_key] = fitness

    return fitness


# ==========================================================
# 8. PARAMETER PSO
# ==========================================================

POPULATION_SIZE = 8
GENERATIONS = 8

W = 0.7
C1 = 1.5
C2 = 1.5

LOWER_BOUNDS = np.array([
    32,
    16,
    0.0001,
    0.10
])

UPPER_BOUNDS = np.array([
    128,
    64,
    0.01,
    0.40
])

DIMENSIONS = 4

positions = np.random.uniform(
    LOWER_BOUNDS,
    UPPER_BOUNDS,
    (POPULATION_SIZE, DIMENSIONS)
)

velocities = np.zeros_like(positions)

personal_best_positions = positions.copy()

personal_best_scores = np.array([
    evaluate_particle(position)
    for position in positions
])

best_index = np.argmax(personal_best_scores)

global_best_position = personal_best_positions[
    best_index
].copy()

global_best_score = personal_best_scores[
    best_index
]

print("\nInisialisasi PSO selesai.")
print("Mulai optimasi PSO...")


# ==========================================================
# 9. PROSES OPTIMASI PSO
# ==========================================================

pso_history = []

for generation in range(GENERATIONS):

    for i in range(POPULATION_SIZE):

        r1 = np.random.random(DIMENSIONS)
        r2 = np.random.random(DIMENSIONS)

        velocities[i] = (
            W * velocities[i]
            + C1 * r1 * (
                personal_best_positions[i] - positions[i]
            )
            + C2 * r2 * (
                global_best_position - positions[i]
            )
        )

        positions[i] = positions[i] + velocities[i]

        positions[i] = np.clip(
            positions[i],
            LOWER_BOUNDS,
            UPPER_BOUNDS
        )

        score = evaluate_particle(positions[i])

        if score > personal_best_scores[i]:

            personal_best_scores[i] = score

            personal_best_positions[i] = (
                positions[i].copy()
            )

        if score > global_best_score:

            global_best_score = score

            global_best_position = (
                positions[i].copy()
            )

    pso_history.append({
        "generation": generation + 1,
        "best_fitness": float(global_best_score)
    })

    print(
        f"Generasi {generation + 1}/{GENERATIONS} "
        f"| Best Fitness: {global_best_score:.5f}"
    )


# ==========================================================
# 10. MENYIMPAN HASIL OPTIMASI PSO
# ==========================================================

best_parameters = decode_particle(
    global_best_position
)

pso_history_df = pd.DataFrame(pso_history)

pso_history_df.to_csv(
    os.path.join(
        ARTIFACT_DIR,
        "ga_pso_fitness_history.csv"
    ),
    index=False
)

optimization_summary = {
    "model": "GA + PSO + Neural Network",
    "selected_features": selected_features,
    "number_of_features": len(selected_features),
    "best_fitness": float(global_best_score),
    "best_parameters": {
        "neurons_1": best_parameters[0],
        "neurons_2": best_parameters[1],
        "learning_rate": best_parameters[2],
        "dropout": best_parameters[3]
    },
    "population_size": POPULATION_SIZE,
    "generations": GENERATIONS,
    "seed": SEED
}

with open(
    os.path.join(ARTIFACT_DIR, "ga_pso_summary.json"),
    "w"
) as file:

    json.dump(
        optimization_summary,
        file,
        indent=4
    )

print("\nOptimasi PSO selesai!")
print("Parameter terbaik:", best_parameters)


# ==========================================================
# 11. TRAINING MODEL AKHIR
# ==========================================================

tf.keras.backend.clear_session()
tf.keras.utils.set_random_seed(SEED)

final_model = build_model(*best_parameters)

early_stop = tf.keras.callbacks.EarlyStopping(
    monitor="val_loss",
    patience=10,
    restore_best_weights=True
)

history = final_model.fit(
    X_train_scaled,
    y_train,
    validation_data=(X_val_scaled, y_val),
    epochs=100,
    batch_size=32,
    class_weight=class_weights,
    callbacks=[early_stop],
    verbose=1
)


# ==========================================================
# 12. EVALUASI DATA TESTING
# ==========================================================

y_probability = final_model.predict(
    X_test_scaled,
    verbose=0
).ravel()

y_prediction = (
    y_probability >= 0.5
).astype(int)

accuracy = accuracy_score(
    y_test,
    y_prediction
)

precision = precision_score(
    y_test,
    y_prediction,
    zero_division=0
)

recall = recall_score(
    y_test,
    y_prediction,
    zero_division=0
)

f1 = f1_score(
    y_test,
    y_prediction,
    zero_division=0
)

roc_auc = roc_auc_score(
    y_test,
    y_probability
)


# ==========================================================
# 13. MENYIMPAN HASIL EVALUASI
# ==========================================================

metrics = {
    "Model": "GA + PSO + NN",
    "Number_of_Features": len(selected_features),
    "Accuracy": accuracy,
    "Precision": precision,
    "Recall": recall,
    "F1_Score": f1,
    "ROC_AUC": roc_auc
}

metrics_df = pd.DataFrame([metrics])

metrics_df.to_csv(
    os.path.join(
        ARTIFACT_DIR,
        "ga_pso_nn_metrics.csv"
    ),
    index=False
)

cm = confusion_matrix(
    y_test,
    y_prediction,
    labels=[0, 1]
)

cm_df = pd.DataFrame(
    cm,
    index=["stable", "unstable"],
    columns=["stable", "unstable"]
)

cm_df.to_csv(
    os.path.join(
        ARTIFACT_DIR,
        "ga_pso_nn_confusion_matrix.csv"
    )
)

history_df = pd.DataFrame(history.history)

history_df.to_csv(
    os.path.join(
        ARTIFACT_DIR,
        "ga_pso_nn_training_history.csv"
    ),
    index=False
)


# ==========================================================
# 14. MENYIMPAN MODEL DAN PREPROCESSING
# ==========================================================

final_model.save(
    os.path.join(
        ARTIFACT_DIR,
        "ga_pso_nn_model.keras"
    )
)

joblib.dump(
    scaler,
    os.path.join(
        ARTIFACT_DIR,
        "ga_pso_nn_scaler.pkl"
    )
)

joblib.dump(
    label_encoder,
    os.path.join(
        ARTIFACT_DIR,
        "ga_pso_nn_label_encoder.pkl"
    )
)


# ==========================================================
# 15. RINGKASAN HASIL AKHIR
# ==========================================================

final_summary = {
    "model": "GA + PSO + Neural Network",
    "selected_features": selected_features,
    "number_of_features": len(selected_features),
    "best_parameters": optimization_summary["best_parameters"],
    "best_validation_fitness": float(global_best_score),
    "accuracy": float(accuracy),
    "precision": float(precision),
    "recall": float(recall),
    "f1_score": float(f1),
    "roc_auc": float(roc_auc),
    "seed": SEED
}

with open(
    os.path.join(
        ARTIFACT_DIR,
        "ga_pso_nn_summary.json"
    ),
    "w"
) as file:

    json.dump(
        final_summary,
        file,
        indent=4
    )


# ==========================================================
# 16. OUTPUT
# ==========================================================

print("\n" + "=" * 60)
print("GA + PSO + NN BERHASIL DIJALANKAN")
print("=" * 60)

print("\nHASIL EVALUASI:")
print(f"Accuracy  : {accuracy * 100:.2f}%")
print(f"Precision : {precision * 100:.2f}%")
print(f"Recall    : {recall * 100:.2f}%")
print(f"F1-Score  : {f1 * 100:.2f}%")
print(f"ROC-AUC   : {roc_auc * 100:.2f}%")

print("\nConfusion Matrix:")
print(cm)

print("\nSemua hasil tersimpan di folder artifacts.")
