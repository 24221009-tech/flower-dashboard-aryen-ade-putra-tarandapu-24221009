# Flower Federated Learning Quickstart (MLX with PyTorch Fallback)

Proyek ini adalah implementasi standar **Flower Federated Learning** berbasis template `@flwrlabs/quickstart-mlx` dengan **1 Server dan 3 SuperNode Clients**, dilengkapi sistem **fallback otomatis ke PyTorch** agar dapat langsung dijalankan di Windows, Linux, maupun di **Streamlit Community Cloud**.

---

## 🌟 Fitur Utama
- **Arsitektur Standar Flower 1.x**: Menggunakan `ClientApp`, `ServerApp`, dan konfigurasi `pyproject.toml`.
- **Topologi**: 1 Server (SuperLink) dan 3 Clients (`num-supernodes = 3`, `fraction-fit = 1.0`, `min-available-clients = 3`).
- **Universal Cross-Platform**:
  - Jika dijalankan di macOS Apple Silicon (ARM64) dengan paket `mlx`: Otomatis memakai Apple MLX.
  - Jika dijalankan di Windows atau Linux/Streamlit Cloud: Otomatis fallback ke PyTorch tanpa error.
- **Interactive Streamlit Dashboard**: Web UI real-time visual simulasi federated learning dengan diagram alur 4 fase (Broadcast -> Local Training -> Upload Weights -> FedAvg Aggregation), grafik metrik loss & akurasi, dan log interaktif.

---

## 📁 Struktur File
```text
aryen_federated_learning_quickstart-mlx/
├── mlxexample/
│   ├── __init__.py
│   ├── client_app.py   # Implementasi Flower ClientApp (3 Client)
│   ├── server_app.py   # Implementasi Flower ServerApp (FedAvg Strategy)
│   └── task.py         # Model MLP, training loop, dan pembagian dataset partisi
├── pyproject.toml      # Konfigurasi Flower App Bundle & Simulation (3 SuperNodes)
├── streamlit_app.py    # Visual Web Dashboard simulasi real-time 1 Server & 3 Client
├── requirements.txt    # Kebutuhan library untuk Windows & Streamlit Cloud
└── README.md
```

---

## 🚀 Cara Menjalankan

### 1. Menjalankan Dashboard Simulasi Visual (Streamlit)
```bash
streamlit run streamlit_app.py
```
Buka browser pada alamat [http://localhost:8501](http://localhost:8501).

### 2. Menjalankan Simulasi Flower CLI Langsung
```bash
flwr run .
```

---

## ☁️ Deploy ke Streamlit Community Cloud
1. Push repositori ini ke GitHub.
2. Buka [share.streamlit.io](https://share.streamlit.io).
3. Pilih repositori, atur `Main file path` ke `streamlit_app.py`.
4. Streamlit Cloud akan membaca `requirements.txt` dan aplikasi langsung aktif!
