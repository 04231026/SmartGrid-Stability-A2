
import streamlit as st
import pandas as pd
import numpy as np
import json
import joblib
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns

from pathlib import Path
from fuzzy_system import fuzzy_prediction


# =====================================================
# 1. KONFIGURASI
# =====================================================

st.set_page_config(
    page_title="Smart Grid Stability - A2",
    page_icon="⚡",
    layout="wide"
)

ROOT = Path(__file__).parent
DATA_PATH = ROOT / "data" / "Data_for_UCI_named.csv"
ARTIFACTS = ROOT / "artifacts"

FEATURES = [
    "tau1", "tau2", "tau3", "tau4",
    "p1", "p2", "p3", "p4",
    "g1", "g2", "g3", "g4"
]

MODELS = {
    "Baseline NN": {
        "model": "baseline_model.keras",
        "scaler": "baseline_scaler.pkl",
        "metrics": "baseline_metrics.csv",
        "features": "all"
    },
    "GA + NN": {
        "model": "ga_nn_model.keras",
        "scaler": "ga_nn_scaler.pkl",
        "metrics": "ga_nn_metrics.csv",
        "features": "ga"
    },
    "PSO + NN": {
        "model": "pso_nn_model.keras",
        "scaler": "pso_nn_scaler.pkl",
        "metrics": "pso_nn_metrics.csv",
        "features": "all"
    },
    "GA + PSO + NN": {
        "model": "ga_pso_nn_model.keras",
        "scaler": "ga_pso_nn_scaler.pkl",
        "metrics": "ga_pso_nn_metrics.csv",
        "features": "ga"
    }
}


# =====================================================
# 2. FUNGSI PEMBACAAN DATA
# =====================================================

@st.cache_data
def load_dataset():
    return pd.read_csv(DATA_PATH)


def load_csv(filename):
    path = ARTIFACTS / filename

    if path.exists():
        return pd.read_csv(path)

    return None


def load_json(filename):
    path = ARTIFACTS / filename

    if path.exists():
        with open(path, "r") as file:
            return json.load(file)

    return None


def find_column(df, aliases):
    if df is None:
        return None

    columns = {
        str(col).lower().replace("-", "_").replace(" ", "_"): col
        for col in df.columns
    }

    for alias in aliases:
        key = alias.lower().replace("-", "_").replace(" ", "_")

        if key in columns:
            return columns[key]

    return None


def get_metric(df, aliases):
    column = find_column(df, aliases)

    if column is None or df.empty:
        return None

    value = pd.to_numeric(
        df[column].iloc[0],
        errors="coerce"
    )

    if pd.isna(value):
        return None

    return float(value)


def show_percent(value):
    if value is None:
        return "N/A"

    return f"{value * 100:.2f}%"


def load_ga_features():
    df = load_csv("ga_selected_features.csv")

    if df is None or df.empty:
        return []

    possible_columns = [
        "feature",
        "selected_feature",
        "features",
        "selected_features"
    ]

    column = find_column(df, possible_columns)

    if column is None:
        column = df.columns[0]

    result = df[column].dropna().astype(str).tolist()

    return [item for item in result if item in FEATURES]


@st.cache_resource
def load_model_and_scaler(model_file, scaler_file):
    model_path = ARTIFACTS / model_file
    scaler_path = ARTIFACTS / scaler_file

    if not model_path.exists():
        raise FileNotFoundError(
            f"Model tidak ditemukan: {model_file}"
        )

    if not scaler_path.exists():
        raise FileNotFoundError(
            f"Scaler tidak ditemukan: {scaler_file}"
        )

    model = tf.keras.models.load_model(
        model_path,
        compile=False
    )

    scaler = joblib.load(scaler_path)

    return model, scaler


def get_model_features(model_name):
    if MODELS[model_name]["features"] == "ga":
        return load_ga_features()

    return FEATURES


def get_history_chart(filename, title):
    history = load_csv(filename)

    if history is None or history.empty:
        st.warning("Data training history belum tersedia.")
        return

    st.subheader(title)

    st.line_chart(history)

    
# =====================================================
# 3. MEMUAT DATASET
# =====================================================

if not DATA_PATH.exists():
    st.error(
        "Dataset tidak ditemukan. Pastikan file berada di "
        "data/Data_for_UCI_named.csv"
    )
    st.stop()

df = load_dataset()


# =====================================================
# 4. SIDEBAR
# =====================================================

st.sidebar.title("⚡ SMART GRID A2")
st.sidebar.caption("Hybrid Artificial Intelligence System")

page = st.sidebar.radio(
    "Pilih Halaman",
    [
        "Project Overview",
        "Dataset Explorer",
        "Exploratory Data Analysis",
        "Baseline Neural Network",
        "Genetic Algorithm",
        "PSO Optimization",
        "Model Comparison",
        "Fuzzy Decision"
    ]
)

st.sidebar.divider()
st.sidebar.write("**Metode AI**")
st.sidebar.write("• Genetic Algorithm")
st.sidebar.write("• Particle Swarm Optimization")
st.sidebar.write("• Neural Network (MLP)")
st.sidebar.write("• Fuzzy Logic Mamdani")

st.sidebar.divider()
st.sidebar.caption("Tugas Besar Kecerdasan Buatan - A2")


# =====================================================
# HALAMAN 1: PROJECT OVERVIEW
# =====================================================

if page == "Project Overview":

    st.title("⚡ Smart Grid Stability Assessment")
    st.subheader("Tugas Besar Kecerdasan Buatan - Kelompok A2")

    st.write(
        "Sistem hybrid Artificial Intelligence untuk "
        "klasifikasi kestabilan jaringan listrik. Sistem "
        "menggabungkan seleksi fitur GA, optimasi parameter "
        "PSO, Neural Network, dan Fuzzy Logic Mamdani."
    )

    st.info(
        "Tujuan sistem adalah meningkatkan performa "
        "klasifikasi kestabilan jaringan listrik dan "
        "memberikan interpretasi tingkat risiko."
    )

    a, b, c, d = st.columns(4)

    a.metric("Jumlah Data", f"{len(df):,}")
    b.metric("Fitur Input", "12")
    c.metric("Konfigurasi Model", "4")
    d.metric("Dataset", "UCI ID 471")

    st.divider()

    st.subheader("Arsitektur Sistem")

    st.code("""
              DATASET UCI
                   |
              PREPROCESSING
                   |
          +--------+--------+
          |                 |
         GA                PSO
     Seleksi Fitur    Optimasi Parameter
          |                 |
          +--------+--------+
                   |
             NEURAL NETWORK
                   |
              PREDIKSI
                   |
             FUZZY MAMDANI
                   |
             TINGKAT RISIKO
                   |
              DASHBOARD
    """)

    st.divider()

    st.subheader("Ringkasan Performa Model")

    rows = []

    for name, config in MODELS.items():

        metrics = load_csv(config["metrics"])

        if metrics is not None:

            rows.append({
                "Model": name,
                "Accuracy": get_metric(
                    metrics, ["Accuracy", "acc"]
                ),
                "F1-Score": get_metric(
                    metrics, ["F1_Score", "F1", "f1_score"]
                )
            })

    if rows:
        result = pd.DataFrame(rows)

        st.dataframe(
            result.style.format({
                "Accuracy": lambda x: show_percent(x),
                "F1-Score": lambda x: show_percent(x)
            }),
            use_container_width=True
        )


# =====================================================
# HALAMAN 2: DATASET EXPLORER
# =====================================================

elif page == "Dataset Explorer":

    st.title("Dataset Explorer")

    a, b, c = st.columns(3)

    a.metric("Jumlah Baris", f"{df.shape[0]:,}")
    b.metric("Jumlah Kolom", df.shape[1])
    c.metric("Missing Values", int(df.isna().sum().sum()))

    st.subheader("Data Preview")
    st.dataframe(df.head(30), use_container_width=True)

    x, y = st.columns(2)

    with x:
        st.subheader("Missing Values")
        st.dataframe(
            df.isna().sum().rename("Jumlah Missing")
        )

    with y:
        st.subheader("Distribusi Target")
        st.dataframe(
            df["stabf"].value_counts().rename_axis(
                "Kelas"
            ).reset_index(name="Jumlah")
        )

    st.subheader("Statistik Deskriptif")
    st.dataframe(df.describe().T, use_container_width=True)

    st.caption(
        "Kolom stab tidak digunakan sebagai fitur input "
        "untuk menghindari data leakage."
    )


# =====================================================
# HALAMAN 3: EDA
# =====================================================

elif page == "Exploratory Data Analysis":

    st.title("Exploratory Data Analysis")

    st.subheader("Distribusi Kelas")

    fig, ax = plt.subplots(figsize=(7, 4))

    sns.countplot(
        data=df,
        x="stabf",
        ax=ax
    )

    ax.set_title("Distribusi Kelas Stabilitas")
    ax.set_xlabel("Kondisi Jaringan")
    ax.set_ylabel("Jumlah Data")

    st.pyplot(fig)
    plt.close(fig)

    st.subheader("Persentase Kelas")

    st.bar_chart(
        df["stabf"].value_counts(normalize=True) * 100
    )

    st.subheader("Correlation Heatmap")

    fig, ax = plt.subplots(figsize=(12, 8))

    sns.heatmap(
        df[FEATURES].corr(),
        cmap="coolwarm",
        center=0,
        ax=ax
    )

    st.pyplot(fig)
    plt.close(fig)

    st.subheader("Distribusi Fitur")

    feature = st.selectbox("Pilih fitur", FEATURES)

    fig, ax = plt.subplots(figsize=(8, 4))

    sns.histplot(
        df[feature],
        kde=True,
        ax=ax
    )

    ax.set_title(f"Distribusi {feature}")

    st.pyplot(fig)
    plt.close(fig)


# =====================================================
# HALAMAN 4: BASELINE NEURAL NETWORK
# =====================================================

elif page == "Baseline Neural Network":

    st.title("Baseline Neural Network")

    st.write(
        "Model awal menggunakan 12 fitur input "
        "tanpa optimasi GA dan PSO."
    )

    metrics = load_csv("baseline_metrics.csv")

    if metrics is not None:

        a, b, c, d = st.columns(4)

        a.metric(
            "Accuracy",
            show_percent(get_metric(metrics, ["Accuracy", "acc"]))
        )

        b.metric(
            "Precision",
            show_percent(get_metric(metrics, ["Precision"]))
        )

        c.metric(
            "Recall",
            show_percent(get_metric(metrics, ["Recall"]))
        )

        d.metric(
            "F1-Score",
            show_percent(get_metric(metrics, ["F1_Score", "F1"]))
        )

        st.metric(
            "ROC-AUC",
            show_percent(get_metric(metrics, ["ROC_AUC", "ROC-AUC", "AUC"]))
        )

    else:
        st.warning("File baseline_metrics.csv tidak ditemukan.")

    get_history_chart(
        "baseline_training_history.csv",
        "Training History Baseline"
    )

    st.subheader("Konfigurasi Model")

    st.write("Input: 12 fitur")
    st.write("Arsitektur: Dense 64 → Dense 32 → Sigmoid")
    st.write("Optimizer: Adam")
    st.write("Loss: Binary Crossentropy")
    st.write("Early Stopping: Aktif")


# =====================================================
# HALAMAN 5: GENETIC ALGORITHM
# =====================================================

elif page == "Genetic Algorithm":

    st.title("Genetic Algorithm")

    st.write(
        "Genetic Algorithm digunakan untuk melakukan "
        "seleksi fitur berdasarkan nilai fitness."
    )

    summary = load_json("ga_summary.json")
    history = load_csv("ga_fitness_history.csv")

    if summary:

        a, b, c = st.columns(3)

        a.metric(
            "Fitur Terpilih",
            summary.get("number_of_selected_features", "N/A")
        )

        b.metric(
            "Fitur Awal",
            summary.get("number_of_original_features", "N/A")
        )

        fitness = summary.get("best_fitness")

        c.metric(
            "Best Fitness",
            show_percent(fitness)
        )

        st.subheader("Fitur Terpilih")

        selected = summary.get("selected_features", [])

        st.write(", ".join(selected))

        if summary.get("number_of_original_features"):
            reduction = (
                1 - summary["number_of_selected_features"]
                / summary["number_of_original_features"]
            ) * 100

            st.metric(
                "Pengurangan Fitur",
                f"{reduction:.2f}%"
            )

    else:
        st.warning("ga_summary.json belum tersedia.")

    if history is not None:

        st.subheader("Konvergensi Genetic Algorithm")

        generation_col = find_column(
            history, ["generation", "generasi"]
        )

        fitness_col = find_column(
            history, ["best_fitness", "fitness", "best"]
        )

        if fitness_col:

            chart_data = history[[fitness_col]].copy()

            if generation_col:
                chart_data.index = history[generation_col]

            st.line_chart(chart_data)

        st.dataframe(history, use_container_width=True)


# =====================================================
# HALAMAN 6: PSO OPTIMIZATION
# =====================================================

elif page == "PSO Optimization":

    st.title("Particle Swarm Optimization")

    pso = load_json("pso_summary.json")
    hybrid = load_json("ga_pso_summary.json")

    if pso:

        st.subheader("PSO + Neural Network")

        params = pso.get("best_parameters", {})

        a, b, c = st.columns(3)

        a.metric("Neuron Layer 1", params.get("neurons_1", "N/A"))
        b.metric("Neuron Layer 2", params.get("neurons_2", "N/A"))
        c.metric("Best Fitness", show_percent(pso.get("best_fitness")))

        st.write("Learning Rate:", params.get("learning_rate", "N/A"))
        st.write("Dropout:", params.get("dropout", "N/A"))

    else:
        st.warning("pso_summary.json tidak ditemukan.")

    history = load_csv("pso_fitness_history.csv")

    if history is not None:

        st.subheader("Konvergensi PSO")

        generation_col = find_column(
            history, ["generation", "generasi"]
        )

        fitness_col = find_column(
            history, ["best_fitness", "fitness", "best"]
        )

        if fitness_col:
            chart = history[[fitness_col]].copy()

            if generation_col:
                chart.index = history[generation_col]

            st.line_chart(chart)

    st.divider()

    if hybrid:

        st.subheader("GA + PSO + Neural Network")

        params = hybrid.get("best_parameters", {})

        a, b, c = st.columns(3)

        a.metric("Neuron Layer 1", params.get("neurons_1", "N/A"))
        b.metric("Neuron Layer 2", params.get("neurons_2", "N/A"))
        c.metric("Best Fitness", show_percent(hybrid.get("best_fitness")))

        st.write("Learning Rate:", params.get("learning_rate", "N/A"))
        st.write("Dropout:", params.get("dropout", "N/A"))

    history = load_csv("ga_pso_fitness_history.csv")

    if history is not None:

        st.subheader("Konvergensi GA + PSO")

        generation_col = find_column(
            history, ["generation", "generasi"]
        )

        fitness_col = find_column(
            history, ["best_fitness", "fitness", "best"]
        )

        if fitness_col:
            chart = history[[fitness_col]].copy()

            if generation_col:
                chart.index = history[generation_col]

            st.line_chart(chart)


# =====================================================
# HALAMAN 7: MODEL COMPARISON
# =====================================================

elif page == "Model Comparison":

    st.title("Model Performance Comparison")

    st.write(
        "Perbandingan hasil pengujian empat konfigurasi "
        "Neural Network."
    )

    rows = []

    for name, config in MODELS.items():

        metrics = load_csv(config["metrics"])

        if metrics is None:
            continue

        rows.append({
            "Model": name,
            "Accuracy": get_metric(
                metrics, ["Accuracy", "acc"]
            ),
            "Precision": get_metric(
                metrics, ["Precision"]
            ),
            "Recall": get_metric(
                metrics, ["Recall"]
            ),
            "F1-Score": get_metric(
                metrics, ["F1_Score", "F1"]
            ),
            "ROC-AUC": get_metric(
                metrics, ["ROC_AUC", "ROC-AUC", "AUC"]
            )
        })

    if rows:

        result = pd.DataFrame(rows)

        st.subheader("Tabel Perbandingan")

        st.dataframe(
            result.style.format({
                "Accuracy": lambda x: show_percent(x),
                "Precision": lambda x: show_percent(x),
                "Recall": lambda x: show_percent(x),
                "F1-Score": lambda x: show_percent(x),
                "ROC-AUC": lambda x: show_percent(x)
            }),
            use_container_width=True
        )

        st.subheader("Grafik Perbandingan")

        chart_metric = st.selectbox(
            "Pilih metrik",
            ["Accuracy", "Precision", "Recall", "F1-Score", "ROC-AUC"]
        )

        chart_data = result[
            ["Model", chart_metric]
        ].dropna()

        if not chart_data.empty:

            st.bar_chart(
                chart_data.set_index("Model")
            )

        else:
            st.warning(
                f"Belum ada nilai {chart_metric} yang valid."
            )

        # Hanya membandingkan model dengan Accuracy valid
        valid = result.dropna(subset=["Accuracy"])

        if not valid.empty:

            best = valid.loc[
                valid["Accuracy"].idxmax()
            ]

            st.success(
                f"Accuracy tertinggi: {best['Model']} "
                f"({best['Accuracy']:.2%})"
            )

        else:
            st.warning(
                "Belum ada data Accuracy yang valid."
            )

    else:
        st.error(
            "Tidak ditemukan file metrik model di folder artifacts."
        )


# =====================================================
# HALAMAN 8: FUZZY DECISION
# =====================================================

elif page == "Fuzzy Decision":

    st.title("Fuzzy Logic Mamdani")

    st.write(
        "Fuzzy Logic digunakan sebagai lapisan keputusan "
        "untuk menginterpretasikan probabilitas ketidakstabilan "
        "dan kesepakatan antar model."
    )

    st.subheader("Simulasi Fuzzy")

    col1, col2 = st.columns(2)

    with col1:
        probability = st.slider(
            "Probabilitas Tidak Stabil",
            0.0, 1.0, 0.5, 0.01
        )

    with col2:
        agreement = st.slider(
            "Kesepakatan Model",
            0.0, 1.0, 0.75, 0.01
        )

    if st.button("Proses Fuzzy"):

        result = fuzzy_prediction(
            probability,
            agreement
        )

        a, b = st.columns(2)

        a.metric(
            "Skor Risiko",
            f"{result['risk_score']:.2f}/100"
        )

        b.metric(
            "Kategori Risiko",
            result["category"]
        )

    st.divider()

    st.subheader("Hasil Integrasi Empat Model")

    integrated = load_csv(
        "integrated_fuzzy_predictions.csv"
    )

    if integrated is not None:

        risk_col = find_column(
            integrated,
            ["Fuzzy_Risk_Score", "risk_score"]
        )

        agreement_col = find_column(
            integrated,
            ["Model_Agreement", "agreement"]
        )

        category_col = find_column(
            integrated,
            ["Fuzzy_Risk_Category", "category"]
        )

        a, b, c = st.columns(3)

        a.metric("Jumlah Data", len(integrated))

        if risk_col:
            a.metric(
                "Rata-rata Risiko",
                f"{integrated[risk_col].mean():.2f}"
            )

        if agreement_col:
            b.metric(
                "Kesepakatan Rata-rata",
                f"{integrated[agreement_col].mean() * 100:.2f}%"
            )

        if category_col:
            c.metric(
                "Kategori Terbanyak",
                str(integrated[category_col].mode().iloc[0])
            )

            st.subheader("Distribusi Kategori Risiko")

            st.bar_chart(
                integrated[category_col].value_counts()
            )

        st.subheader("Data Hasil Integrasi")

        st.dataframe(
            integrated.head(100),
            use_container_width=True
        )

        st.download_button(
            "Download Hasil Integrasi CSV",
            data=integrated.to_csv(index=False).encode("utf-8"),
            file_name="integrated_fuzzy_predictions.csv",
            mime="text/csv"
        )

    else:
        st.warning(
            "File integrated_fuzzy_predictions.csv tidak ditemukan."
        )

    st.divider()

    st.subheader("Prediksi Menggunakan Empat Model")

    st.write(
        "Masukkan nilai parameter jaringan listrik untuk "
        "menjalankan prediksi menggunakan model yang telah dilatih."
    )

    with st.form("prediction_form"):

        input_values = {}

        columns = st.columns(3)

        for i, feature in enumerate(FEATURES):

            with columns[i % 3]:

                input_values[feature] = st.number_input(
                    feature,
                    value=float(df[feature].median()),
                    format="%.5f"
                )

        submitted = st.form_submit_button(
            "Jalankan Prediksi",
            type="primary"
        )

    if submitted:

        try:

            input_df = pd.DataFrame([input_values])

            predictions = {}

            for name, config in MODELS.items():

                features = get_model_features(name)

                if not features:
                    raise ValueError(
                        "Fitur hasil GA tidak ditemukan."
                    )

                model, scaler = load_model_and_scaler(
                    config["model"],
                    config["scaler"]
                )

                scaled = scaler.transform(
                    input_df[features]
                )

                prediction = model.predict(
                    scaled,
                    verbose=0
                )

                predictions[name] = float(
                    np.asarray(prediction).reshape(-1)[0]
                )

            values = np.array(list(predictions.values()))

            average_probability = float(np.mean(values))

            binary = (values >= 0.5).astype(int)

            total_pairs = 0
            same_pairs = 0

            for i in range(len(binary)):
                for j in range(i + 1, len(binary)):

                    total_pairs += 1

                    if binary[i] == binary[j]:
                        same_pairs += 1

            agreement_value = same_pairs / total_pairs

            fuzzy_result = fuzzy_prediction(
                average_probability,
                agreement_value
            )

            st.divider()

            st.subheader("Hasil Prediksi Model")

            prediction_table = []

            for name, value in predictions.items():

                prediction_table.append({
                    "Model": name,
                    "Probabilitas Tidak Stabil": value,
                    "Prediksi": (
                        "Unstable" if value >= 0.5
                        else "Stable"
                    )
                })

            st.dataframe(
                pd.DataFrame(prediction_table).style.format({
                    "Probabilitas Tidak Stabil": "{:.4f}"
                }),
                use_container_width=True
            )

            st.subheader("Hasil Keputusan Fuzzy")

            a, b, c = st.columns(3)

            a.metric(
                "Probabilitas Rata-rata",
                f"{average_probability:.4f}"
            )

            b.metric(
                "Kesepakatan Model",
                f"{agreement_value * 100:.2f}%"
            )

            c.metric(
                "Skor Risiko",
                f"{fuzzy_result['risk_score']:.2f}/100"
            )

            category = fuzzy_result["category"]

            if category == "Risiko Tinggi":
                st.error("RISIKO TINGGI")

            elif category == "Risiko Sedang":
                st.warning("RISIKO SEDANG")

            else:
                st.success("RISIKO RENDAH")

            st.caption(
                "Hasil merupakan simulasi berdasarkan model "
                "yang telah dilatih, bukan pengukuran langsung "
                "kondisi jaringan listrik fisik."
            )

        except Exception as error:

            st.error(
                f"Prediksi gagal dijalankan: {error}"
            )


# =====================================================
# FOOTER
# =====================================================

st.sidebar.divider()

st.sidebar.caption(
    "Smart Grid Stability Assessment | Kelompok A2 | 2026"
)
