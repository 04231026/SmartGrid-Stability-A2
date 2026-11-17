
import os
import json
import random
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import f1_score

# ==================================================
# 1. KONFIGURASI
# ==================================================

SEED = 42

POPULATION_SIZE = 10
GENERATIONS = 10
CROSSOVER_RATE = 0.8
MUTATION_RATE = 0.05
ELITE_SIZE = 1
TOURNAMENT_SIZE = 3

DATA_PATH = "data/Data_for_UCI_named.csv"
ARTIFACT_DIR = "artifacts"

os.makedirs(ARTIFACT_DIR, exist_ok=True)

random.seed(SEED)
np.random.seed(SEED)

FEATURES = [
    "tau1", "tau2", "tau3", "tau4",
    "p1", "p2", "p3", "p4",
    "g1", "g2", "g3", "g4"
]

TARGET = "stabf"

# ==================================================
# 2. MEMUAT DAN MEMBAGI DATASET
# ==================================================

print("\n=== GENETIC ALGORITHM: FEATURE SELECTION ===")

df = pd.read_csv(DATA_PATH)

X = df[FEATURES].copy()
y = df[TARGET].copy()

encoder = LabelEncoder()
y = encoder.fit_transform(y)

# 70% training, 15% validation, 15% testing
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

# Scaling hanya berdasarkan data training
scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)

print("Jumlah data training :", len(X_train))
print("Jumlah data validation:", len(X_val))
print("Jumlah data testing   :", len(X_test))
print("Jumlah fitur awal     :", len(FEATURES))

# ==================================================
# 3. FUNGSI FITNESS
# ==================================================

fitness_cache = {}

def calculate_fitness(chromosome):

    # Pastikan minimal satu fitur dipilih
    if np.sum(chromosome) == 0:
        return 0.0

    key = tuple(chromosome)

    if key in fitness_cache:
        return fitness_cache[key]

    selected_indices = np.where(chromosome == 1)[0]

    X_train_selected = X_train_scaled[:, selected_indices]
    X_val_selected = X_val_scaled[:, selected_indices]

    model = MLPClassifier(
        hidden_layer_sizes=(64, 32),
        activation="relu",
        solver="adam",
        alpha=0.001,
        max_iter=100,
        early_stopping=True,
        validation_fraction=0.1,
        n_iter_no_change=10,
        random_state=SEED
    )

    model.fit(X_train_selected, y_train)

    predictions = model.predict(X_val_selected)

    validation_f1 = f1_score(
        y_val,
        predictions,
        average="binary",
        zero_division=0
    )

    # Mengutamakan F1 sekaligus mempertimbangkan jumlah fitur
    feature_ratio = len(selected_indices) / len(FEATURES)

    fitness = (
        0.90 * validation_f1
        + 0.10 * (1 - feature_ratio)
    )

    fitness_cache[key] = fitness

    return fitness


# ==================================================
# 4. INISIALISASI POPULASI
# ==================================================

def create_population():

    population = []

    for _ in range(POPULATION_SIZE):

        chromosome = np.random.randint(
            0, 2, size=len(FEATURES)
        )

        if np.sum(chromosome) == 0:
            chromosome[np.random.randint(0, len(FEATURES))] = 1

        population.append(chromosome)

    return population


# ==================================================
# 5. SELEKSI TOURNAMENT
# ==================================================

def selection(population, fitness_values):

    candidates = random.sample(
        range(len(population)),
        TOURNAMENT_SIZE
    )

    winner = max(
        candidates,
        key=lambda index: fitness_values[index]
    )

    return population[winner].copy()


# ==================================================
# 6. CROSSOVER
# ==================================================

def crossover(parent1, parent2):

    if random.random() < CROSSOVER_RATE:

        point = random.randint(1, len(FEATURES) - 1)

        child1 = np.concatenate([
            parent1[:point],
            parent2[point:]
        ])

        child2 = np.concatenate([
            parent2[:point],
            parent1[point:]
        ])

        return child1, child2

    return parent1.copy(), parent2.copy()


# ==================================================
# 7. MUTASI
# ==================================================

def mutation(chromosome):

    child = chromosome.copy()

    for i in range(len(child)):

        if random.random() < MUTATION_RATE:
            child[i] = 1 - child[i]

    if np.sum(child) == 0:
        child[np.random.randint(0, len(child))] = 1

    return child


# ==================================================
# 8. PROSES EVOLUSI GENETIC ALGORITHM
# ==================================================

population = create_population()

history = []

best_chromosome = None
best_fitness = -1

for generation in range(GENERATIONS):

    fitness_values = [
        calculate_fitness(chromosome)
        for chromosome in population
    ]

    generation_best_index = np.argmax(fitness_values)
    generation_best_fitness = fitness_values[generation_best_index]

    if generation_best_fitness > best_fitness:

        best_fitness = generation_best_fitness

        best_chromosome = population[
            generation_best_index
        ].copy()

    selected_count = int(np.sum(best_chromosome))

    history.append({
        "generation": generation + 1,
        "best_fitness": best_fitness,
        "generation_fitness": generation_best_fitness,
        "selected_features": selected_count
    })

    print(
        f"Generasi {generation + 1:02d} | "
        f"Fitness terbaik: {best_fitness:.5f} | "
        f"Jumlah fitur: {selected_count}"
    )

    # Elitism: mempertahankan individu terbaik
    elite_indices = np.argsort(fitness_values)[-ELITE_SIZE:]

    new_population = [
        population[index].copy()
        for index in elite_indices
    ]

    # Membentuk populasi generasi berikutnya
    while len(new_population) < POPULATION_SIZE:

        parent1 = selection(population, fitness_values)
        parent2 = selection(population, fitness_values)

        child1, child2 = crossover(parent1, parent2)

        child1 = mutation(child1)
        child2 = mutation(child2)

        new_population.append(child1)

        if len(new_population) < POPULATION_SIZE:
            new_population.append(child2)

    population = new_population


# ==================================================
# 9. MENYIMPAN HASIL TERBAIK
# ==================================================

selected_features = [
    FEATURES[i]
    for i in range(len(FEATURES))
    if best_chromosome[i] == 1
]

history_df = pd.DataFrame(history)

history_df.to_csv(
    os.path.join(ARTIFACT_DIR, "ga_fitness_history.csv"),
    index=False
)

pd.DataFrame({
    "feature": selected_features
}).to_csv(
    os.path.join(ARTIFACT_DIR, "ga_selected_features.csv"),
    index=False
)

summary = {
    "best_fitness": float(best_fitness),
    "selected_features": selected_features,
    "number_of_selected_features": len(selected_features),
    "number_of_original_features": len(FEATURES),
    "generations": GENERATIONS,
    "population_size": POPULATION_SIZE,
    "seed": SEED
}

with open(
    os.path.join(ARTIFACT_DIR, "ga_summary.json"),
    "w"
) as file:
    json.dump(summary, file, indent=4)

print("\n=== HASIL GENETIC ALGORITHM ===")
print("Fitur terbaik:", selected_features)
print("Jumlah fitur terpilih:", len(selected_features))
print("Fitness terbaik:", round(best_fitness, 5))

print("\nHasil berhasil disimpan di folder artifacts/")
