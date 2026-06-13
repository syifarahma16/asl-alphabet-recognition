import numpy as np
from PIL import Image
import tensorflow as tf
import os
import gdown

# ── Konfigurasi path model 
# MODEL_DIR dihitung relatif dari lokasi file utils.py ini
# jadi kalau utils.py ada di app/, maka models/ ada di root project
MODEL_DIR = os.path.join(os.path.dirname(__file__), '..', 'models')
MODEL_PATH = os.path.join(MODEL_DIR, 'mobilenet_asl.keras')

# File ID dari link Google Drive model MobileNetV2
# https://drive.google.com/file/d/<FILE_ID>/view
GDRIVE_FILE_ID = "12tnwjCQa9IaGMA5lBL5bW74xlGJ1qkun"

# 29 kelas ASL: A-Z + del, nothing, space
CLASS_NAMES = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J',
               'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T',
               'U', 'V', 'W', 'X', 'Y', 'Z', 'del', 'nothing', 'space']

# Ukuran input model — harus sama dengan waktu training
IMG_SIZE = (200, 200)


def download_model():
    """
    Download model dari Google Drive kalau belum ada di lokal.
    Dipanggil otomatis saat pertama kali load_model() dijalankan.
    Ini penting untuk Streamlit Cloud karena model tidak di-push ke GitHub.
    """
    os.makedirs(MODEL_DIR, exist_ok=True)  # buat folder models/ kalau belum ada

    if not os.path.exists(MODEL_PATH):
        # Model belum ada — download dari Google Drive
        print("⬇️ Downloading model dari Google Drive...")
        url = f"https://drive.google.com/uc?id={GDRIVE_FILE_ID}"
        gdown.download(url, MODEL_PATH, quiet=False)
        print("✅ Model berhasil didownload!")
    else:
        # Model sudah ada — skip download
        print("✅ Model sudah ada, skip download.")


# Variable global untuk menyimpan model yang sudah di-load
# Tujuannya supaya model hanya di-load sekali, tidak setiap kali predict dipanggil
_model = None

def load_model():
    """
    Load model Keras dari file .keras.
    Menggunakan pattern singleton — model hanya di-load sekali
    dan disimpan di _model untuk reuse di prediksi berikutnya.
    """
    global _model
    if _model is None:
        download_model()  # pastikan file model sudah tersedia dulu
        _model = tf.keras.models.load_model(MODEL_PATH)
        print("✅ Model loaded!")
    return _model


def preprocess_image(image):
    """
    Preprocessing gambar sebelum dimasukkan ke model:
    1. Konversi ke RGB (jaga-jaga kalau input RGBA atau grayscale)
    2. Resize ke 200x200px sesuai input model
    3. Normalisasi pixel values dari [0, 255] ke [0, 1]
    4. Tambah dimensi batch → shape jadi (1, 200, 200, 3)
    """
    if image.mode != 'RGB':
        image = image.convert('RGB')

    image = image.resize(IMG_SIZE)                  # resize ke 200x200
    img_array = np.array(image) / 255.0             # normalisasi
    img_array = np.expand_dims(img_array, axis=0)   # tambah dimensi batch
    return img_array


def predict(image):
    """
    Prediksi kelas ASL dari gambar input.
    
    Returns:
        predicted_class (str) : nama kelas dengan probabilitas tertinggi
        confidence (float)    : confidence score (0.0 - 1.0)
        top5 (list of tuple)  : 5 kelas teratas beserta probabilitasnya
    """
    model = load_model()
    img_array = preprocess_image(image)

    # Jalankan inferensi — verbose=0 supaya tidak print progress bar
    predictions = model.predict(img_array, verbose=0)
    pred_probs = predictions[0]  # ambil hasil prediksi untuk 1 gambar

    # Ambil kelas dengan probabilitas tertinggi
    pred_idx = np.argmax(pred_probs)
    predicted_class = CLASS_NAMES[pred_idx]
    confidence = float(pred_probs[pred_idx])

    # Ambil top 5 kelas berdasarkan probabilitas tertinggi
    top5_idx = np.argsort(pred_probs)[::-1][:5]
    top5 = [(CLASS_NAMES[i], float(pred_probs[i])) for i in top5_idx]

    return predicted_class, confidence, top5