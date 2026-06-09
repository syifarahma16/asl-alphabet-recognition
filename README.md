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
    ├── config/
    │   └── config.yaml             # Parameter dan jalur konfigurasi
    ├── data/                       # Dataset (umumnya gitignored)
    ├── models/
    │   ├── cnn_baseline.keras      # Pretrained model arsitektur CNN
    │   └── mobilenet_asl.keras     # MobileNetV2 transfer learning model
    ├── notebooks/
    │   └── asl_pipeline.ipynb      # Full pipeline CRISP-DM
    ├── reports/
    │   └── figures/                # Plot EDA & evaluasi
    ├── src/
    │   ├── augmentation.py         # Augmentasi data
    │   ├── data_loader.py          # Loading dataset
    │   ├── model_cnn.py            # Definisi model CNN
    │   ├── model_mobilenet.py      # Definisi model MobileNetV2
    │   └── evaluate.py             # Evaluasi performa model
    ├── requirements.txt
    └── README.md

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

## 👤 Author

**Syifa Rahma Sabita**  
Intelligo ID AI Bootcamp  
[GitHub](https://github.com/syifarahma16)