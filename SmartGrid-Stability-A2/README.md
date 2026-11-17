# Smart Grid Stability Assessment – Kelompok A2

Hybrid AI: Genetic Algorithm (seleksi fitur), Particle Swarm Optimization (optimasi hyperparameter), MLP (klasifikasi), dan Fuzzy Mamdani (interpretasi risiko).

## Struktur
- `app.py`: dashboard awal Streamlit (Overview dan Dataset Explorer)
- `fuzzy_system.py`: modul Fuzzy Mamdani
- `requirements.txt`: library Python
- `data/`: simpan `Data_for_UCI_named.csv`
- `artifacts/`: artefak model setelah training
- `notebooks/`: notebook EDA dan eksperimen

## Menjalankan
```bash
pip install -r requirements.txt
streamlit run app.py
```

Dashboard awal belum menjalankan prediksi model karena artefak hasil training harus dibuat terlebih dahulu.

Dataset: Electrical Grid Stability Simulated Data, UCI Machine Learning Repository, ID 471.
Target klasifikasi: `stabf`. Kolom `stab` tidak boleh dipakai sebagai fitur karena menyebabkan data leakage.
Catatan: metadata UCI menyebut `p1` redundan terhadap `p2`, `p3`, dan `p4`; jelaskan perlakuan fitur ini dalam metodologi.
