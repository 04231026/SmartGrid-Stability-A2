import streamlit as st
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns

st.set_page_config(page_title="Smart Grid Stability - A2", layout="wide")
ROOT = Path(__file__).parent
DATA = ROOT / "data" / "Data_for_UCI_named.csv"
FEATURES = ["tau1","tau2","tau3","tau4","p1","p2","p3","p4","g1","g2","g3","g4"]

st.sidebar.title("Navigasi")
page = st.sidebar.radio("Pilih halaman", ["Project Overview", "Dataset Explorer"])

if page == "Project Overview":
    st.title("Smart Grid Stability Assessment")
    st.subheader("Tugas Besar Kecerdasan Buatan - Kelompok A2")
    st.write("Sistem hybrid AI untuk klasifikasi kestabilan jaringan listrik. GA digunakan untuk seleksi fitur, PSO untuk optimasi hyperparameter, MLP untuk klasifikasi, dan Fuzzy Mamdani untuk interpretasi risiko.")
    st.markdown("**Arsitektur:** Dataset → Preprocessing → GA → PSO → MLP → Fuzzy → Streamlit")
    st.info("Ini adalah dashboard kerangka awal Minggu 12. Halaman optimasi dan prediksi ditambahkan setelah eksperimen dan artefak model tersedia.")
    a,b,c = st.columns(3)
    a.metric("Fitur input", "12")
    b.metric("Target", "stabf")
    c.metric("Dataset", "UCI ID 471")

elif page == "Dataset Explorer":
    st.title("Dataset Explorer")
    if not DATA.exists():
        st.warning("Dataset belum ditemukan. Letakkan Data_for_UCI_named.csv di folder data/.")
        st.code("data/Data_for_UCI_named.csv")
        st.stop()
    df = pd.read_csv(DATA)
    st.write(f"Ukuran dataset: {df.shape[0]:,} baris × {df.shape[1]} kolom")
    st.dataframe(df.head(20), use_container_width=True)
    x,y = st.columns(2)
    with x:
        st.subheader("Missing values")
        st.dataframe(df.isna().sum().rename("Jumlah missing"))
    with y:
        st.subheader("Distribusi target")
        if "stabf" in df:
            st.dataframe(df["stabf"].value_counts().rename_axis("Kelas").reset_index(name="Jumlah"))
            fig, ax = plt.subplots()
            sns.countplot(data=df, x="stabf", ax=ax)
            st.pyplot(fig)
    st.subheader("Statistik deskriptif")
    st.dataframe(df.describe().T, use_container_width=True)
    if all(f in df.columns for f in FEATURES):
        fig, ax = plt.subplots(figsize=(10,6))
        sns.heatmap(df[FEATURES].corr(), cmap="coolwarm", center=0, ax=ax)
        st.pyplot(fig)
    st.caption("Kolom stab tidak digunakan sebagai fitur input untuk menghindari data leakage.")
