# 🤟 ASL Alphabet Recognition

![Python](https://img.shields.io/badge/Python-3.11-blue?logo=python&logoColor=white)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.21-orange?logo=tensorflow&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-deployed-red?logo=streamlit&logoColor=white)
![CRISP-DM](https://img.shields.io/badge/Framework-CRISP--DM-purple)

Klasifikasi alfabet bahasa isyarat Amerika (ASL) menggunakan deep learning. Membandingkan CNN baseline dengan MobileNetV2 transfer learning pada 29 kelas (A–Z, del, nothing, space).

> 🎓 Final Project — Intelligo ID AI Bootcamp | Syifa Rahma Sabita

---

## 📊 Hasil

| Model | Val Accuracy | Val Loss | Epochs |
|---|---|---|---|
| CNN Baseline | 88.10% | 0.3099 | 23 |
| **MobileNetV2** | **99.83%** | **0.0233** | **10** |

---

## 🗂️ Struktur Project

    asl-alphabet-recognition/
    ├── app/
    │   ├── app.py                  # Streamlit app
    │   └── utils.py                # Prediksi dan utilitas model
    ├── data/                       # Dataset (gitignored) — download terpisah
    ├── models/
    │   ├── cnn_baseline.keras      # Pretrained model arsitektur CNN
    │   └── mobilenet_asl.keras     # MobileNetV2 transfer learning model
    ├── notebooks/
    │   └── asl_pipeline.ipynb      # Full pipeline CRISP-DM
    ├── reports/
    │   └── figures/                # Plot EDA & evaluasi
    ├── requirements.txt
    └── README.md

---
## 📦 Dataset

Models: [CNN dan MobileNetV2 Model](https://drive.google.com/drive/folders/1LgvaBMVBrraTw3kqNNrDX7wui2AMBIc-?usp=sharing)
Dataset: [ASL Alphabet Dataset](https://drive.google.com/drive/folders/1qtRqctFgSu6S-ngJvEQrbUPoib2o_Ir8?usp=sharing)  
- 29 kelas (A–Z, del, nothing, space)  
- ~2.900 gambar (100 gambar/kelas)  
- Ukuran gambar: 200×200px JPG

---

## 🚀 Cara Menjalankan

### 1. Clone repo
```bash
git clone https://github.com/syifarahma16/asl-alphabet-recognition.git
cd asl-alphabet-recognition
```

### 2. Setup environment
```bash
python -m venv venv
.\venv\Scripts\Activate
pip install -r requirements.txt
```

### 3. Download model
Download `mobilenet_asl.keras` dari Google Drive dan taruh di folder `models/`.

### 4. Jalankan Streamlit
```bash
streamlit run app/app.py
```

---

## 🛠️ Tech Stack

| | Tools |
|---|---|
| Language | Python 3.11 |
| Deep Learning | TensorFlow 2.21 / Keras |
| Deployment | Streamlit |
| Training | Google Colab (GPU T4) |
| Version Control | Git + GitHub |

---

## ⚠️ Limitasi

Dataset menggunakan kondisi terkontrol (background polos, 200×200px). Model mungkin kurang akurat untuk foto real-world dengan background ramai atau pencahayaan berbeda.

---

##  Presentasi

- 📊 [Slide Presentasi (Google Drive)](https://docs.google.com/presentation/d/1uefXj9hDAXnBRrsQn6KLaFOe87eQ1zDhBSdZ75S9zqI/edit?usp=sharing)

---

## 👤 Author

**Syifa Rahma Sabita**  
Intelligo ID AI Bootcamp  
[GitHub](https://github.com/syifarahma16)