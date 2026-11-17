
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Lokasi folder proyek
BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "data" / "Data_for_UCI_named.csv"
OUTPUT_DIR = BASE_DIR / "artifacts"
OUTPUT_DIR.mkdir(exist_ok=True)

print("=" * 55)
print("EDA - SMART GRID STABILITY A2")
print("=" * 55)

# 1. Membaca dataset
if not DATA_PATH.exists():
    raise FileNotFoundError(
        f"Dataset tidak ditemukan: {DATA_PATH}\n"
        "Pastikan nama file dan ekstensi CSV sudah benar."
    )

df = pd.read_csv(DATA_PATH)

# 2. Informasi dasar
print("\n1. INFORMASI DATASET")
print("Ukuran dataset:", df.shape)
print("\nNama kolom:")
print(df.columns.tolist())

print("\n5 Data pertama:")
print(df.head())

# 3. Pemeriksaan data kosong
print("\n2. PEMERIKSAAN DATA KOSONG")
print(df.isnull().sum())

# 4. Pemeriksaan data duplikat
print("\n3. PEMERIKSAAN DATA DUPLIKAT")
print("Jumlah duplikat:", df.duplicated().sum())

# 5. Statistik deskriptif
print("\n4. STATISTIK DESKRIPTIF")
print(df.describe())

# 6. Distribusi target
print("\n5. DISTRIBUSI TARGET STABF")
if "stabf" not in df.columns:
    raise ValueError("Kolom target 'stabf' tidak ditemukan.")

print(df["stabf"].value_counts())
print("\nPersentase target:")
print(df["stabf"].value_counts(normalize=True) * 100)

# 7. Simpan ringkasan
with open(OUTPUT_DIR / "eda_summary.txt", "w", encoding="utf-8") as f:
    f.write("RINGKASAN EDA SMART GRID STABILITY\n")
    f.write(f"Ukuran dataset: {df.shape}\n\n")
    f.write("Data kosong:\n")
    f.write(df.isnull().sum().to_string())
    f.write("\n\nJumlah duplikat: ")
    f.write(str(df.duplicated().sum()))
    f.write("\n\nDistribusi target:\n")
    f.write(df["stabf"].value_counts().to_string())

# 8. Grafik distribusi target
plt.figure(figsize=(7, 5))
sns.countplot(data=df, x="stabf")
plt.title("Distribusi Kelas Smart Grid Stability")
plt.xlabel("Kondisi Grid")
plt.ylabel("Jumlah Data")
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "target_distribution.png", dpi=300)
plt.close()

# 9. Grafik korelasi
numeric_df = df.select_dtypes(include="number")

plt.figure(figsize=(12, 9))
sns.heatmap(
    numeric_df.corr(),
    annot=True,
    cmap="coolwarm",
    fmt=".2f"
)
plt.title("Korelasi Antarvariabel")
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "correlation_heatmap.png", dpi=300)
plt.close()

print("\n" + "=" * 55)
print("EDA SELESAI!")
print("Hasil disimpan di folder artifacts.")
print("=" * 55)
