
import os
import json
import joblib
import numpy as np
import pandas as pd
import tensorflow as tf

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from fuzzy_system import fuzzy_prediction


# =====================================================
# 1. KONFIGURASI
# =====================================================

SEED = 42

DATA_PATH = "data/Data_for_UCI_named.csv"
ARTIFACT_DIR = "artifacts"

os.makedirs(ARTIFACT_DIR, exist_ok=True)

FEATURES_ALL = [
    "tau1", "tau2", "tau3", "tau4",
    "p1", "p2", "p3", "p4",
    "g1", "g2", "g3", "g4"
]

print("=" * 60)
print("INTEGRASI GA + PSO + NEURAL NETWORK + FUZZY")
print("=" * 60)


# =====================================================
# 2. MEMUAT DATA
# =====================================================

df = pd.read_csv(DATA_PATH)

X = df[FEATURES_ALL].copy()
y = df["stabf"].copy()

encoder = LabelEncoder()
y_encoded = encoder.fit_transform(y)

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

print("\nJumlah data testing:", len(X_test))


# =====================================================
# 3. MEMUAT FITUR HASIL GA
# =====================================================

feature_df = pd.read_csv(
    os.path.join(ARTIFACT_DIR, "ga_selected_features.csv")
)

if "feature" in feature_df.columns:
    selected_features = feature_df["feature"].tolist()
else:
    selected_features = feature_df.iloc[:, 0].tolist()

selected_features = [
    f for f in selected_features
    if f in FEATURES_ALL
]

if len(selected_features) == 0:
    raise ValueError("Fitur GA tidak ditemukan!")

print("\nFitur hasil GA:", selected_features)


# =====================================================
# 4. MEMUAT EMPAT MODEL DAN SCALER
# =====================================================

model_files = {
    "Baseline NN": "baseline_model.keras",
    "GA + NN": "ga_nn_model.keras",
    "PSO + NN": "pso_nn_model.keras",
    "GA + PSO + NN": "ga_pso_nn_model.keras"
}

scaler_files = {
    "Baseline NN": "baseline_scaler.pkl",
    "GA + NN": "ga_nn_scaler.pkl",
    "PSO + NN": "pso_nn_scaler.pkl",
    "GA + PSO + NN": "ga_pso_nn_scaler.pkl"
}

models = {}
scalers = {}

for name in model_files:

    model_path = os.path.join(
        ARTIFACT_DIR,
        model_files[name]
    )

    scaler_path = os.path.join(
        ARTIFACT_DIR,
        scaler_files[name]
    )

    models[name] = tf.keras.models.load_model(
        model_path,
        compile=False
    )

    scalers[name] = joblib.load(scaler_path)

    print(f"Model berhasil dimuat: {name}")


# =====================================================
# 5. MENYIAPKAN DATA UNTUK SETIAP MODEL
# =====================================================

X_test_all = X_test[FEATURES_ALL].copy()
X_test_ga = X_test[selected_features].copy()


# =====================================================
# 6. PREDIKSI EMPAT MODEL
# =====================================================

probabilities = {}

# Baseline NN
baseline_scaled = scalers["Baseline NN"].transform(
    X_test_all
)

probabilities["Baseline NN"] = models[
    "Baseline NN"
].predict(
    baseline_scaled,
    verbose=0
).ravel()


# GA + NN
ga_scaled = scalers["GA + NN"].transform(
    X_test_ga
)

probabilities["GA + NN"] = models[
    "GA + NN"
].predict(
    ga_scaled,
    verbose=0
).ravel()


# PSO + NN
pso_scaled = scalers["PSO + NN"].transform(
    X_test_all
)

probabilities["PSO + NN"] = models[
    "PSO + NN"
].predict(
    pso_scaled,
    verbose=0
).ravel()


# GA + PSO + NN
ga_pso_scaled = scalers["GA + PSO + NN"].transform(
    X_test_ga
)

probabilities["GA + PSO + NN"] = models[
    "GA + PSO + NN"
].predict(
    ga_pso_scaled,
    verbose=0
).ravel()


# =====================================================
# 7. AGREGASI PROBABILITAS DAN KESEPAKATAN
# =====================================================

probability_matrix = np.column_stack([
    probabilities["Baseline NN"],
    probabilities["GA + NN"],
    probabilities["PSO + NN"],
    probabilities["GA + PSO + NN"]
])

# Rata-rata probabilitas tidak stabil
average_probability = np.mean(
    probability_matrix,
    axis=1
)

# Prediksi kelas dari masing-masing model
binary_predictions = (
    probability_matrix >= 0.5
).astype(int)

# Kesepakatan berdasarkan proporsi pasangan
# model yang memberikan kelas sama.
agreement_values = []

for predictions in binary_predictions:

    same_pairs = 0
    total_pairs = 0

    for i in range(len(predictions)):
        for j in range(i + 1, len(predictions)):

            total_pairs += 1

            if predictions[i] == predictions[j]:
                same_pairs += 1

    agreement = same_pairs / total_pairs

    agreement_values.append(agreement)

agreement_values = np.array(agreement_values)


# =====================================================
# 8. PROSES FUZZY MAMDANI
# =====================================================

fuzzy_results = []

print("\nMenjalankan inferensi Fuzzy...")

for i in range(len(X_test)):

    result = fuzzy_prediction(
        average_probability[i],
        agreement_values[i]
    )

    fuzzy_results.append(result)


# =====================================================
# 9. MEMBUAT TABEL HASIL
# =====================================================

results = pd.DataFrame({
    "Actual": encoder.inverse_transform(y_test),

    "Baseline_Probability": probabilities["Baseline NN"],

    "GA_NN_Probability": probabilities["GA + NN"],

    "PSO_NN_Probability": probabilities["PSO + NN"],

    "GA_PSO_NN_Probability": probabilities["GA + PSO + NN"],

    "Average_Probability": average_probability,

    "Model_Agreement": agreement_values,

    "Fuzzy_Risk_Score": [
        item["risk_score"]
        for item in fuzzy_results
    ],

    "Fuzzy_Risk_Category": [
        item["category"]
        for item in fuzzy_results
    ]
})


# =====================================================
# 10. MENYIMPAN HASIL
# =====================================================

output_path = os.path.join(
    ARTIFACT_DIR,
    "integrated_fuzzy_predictions.csv"
)

results.to_csv(
    output_path,
    index=False
)


# =====================================================
# 11. RINGKASAN KATEGORI RISIKO
# =====================================================

risk_summary = results[
    "Fuzzy_Risk_Category"
].value_counts()

summary = {
    "total_testing_data": int(len(results)),
    "average_probability": float(
        np.mean(average_probability)
    ),
    "average_model_agreement": float(
        np.mean(agreement_values)
    ),
    "average_fuzzy_risk": float(
        results["Fuzzy_Risk_Score"].mean()
    ),
    "risk_distribution": {
        str(key): int(value)
        for key, value in risk_summary.items()
    }
}

with open(
    os.path.join(
        ARTIFACT_DIR,
        "integrated_fuzzy_summary.json"
    ),
    "w"
) as file:

    json.dump(summary, file, indent=4)


# =====================================================
# 12. MENAMPILKAN HASIL
# =====================================================

print("\n" + "=" * 60)
print("INTEGRASI MODEL DAN FUZZY BERHASIL")
print("=" * 60)

print("\nRata-rata probabilitas tidak stabil:",
      f"{np.mean(average_probability):.4f}")

print("Rata-rata kesepakatan model:",
      f"{np.mean(agreement_values):.4f}")

print("Rata-rata skor risiko Fuzzy:",
      f"{results['Fuzzy_Risk_Score'].mean():.2f}")

print("\nDistribusi kategori risiko:")
print(risk_summary)

print("\nContoh lima hasil:")
print(results.head().to_string(index=False))

print("\nHasil disimpan di:")
print(output_path)

print("\nIntegrasi selesai.")
