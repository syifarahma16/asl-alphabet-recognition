import numpy as np
from PIL import Image
import keras
import os
import gdown

MODEL_DIR = os.path.join(os.path.dirname(__file__), '..', 'models')
MODEL_PATH = os.path.join(MODEL_DIR, 'mobilenet_asl.keras')
GDRIVE_FILE_ID = "12tnwjCQa9IaGMA5lBL5bW74xlGJ1qkun"

CLASS_NAMES = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J',
               'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T',
               'U', 'V', 'W', 'X', 'Y', 'Z', 'del', 'nothing', 'space']

IMG_SIZE = (200, 200)

def download_model():
    os.makedirs(MODEL_DIR, exist_ok=True)
    if not os.path.exists(MODEL_PATH):
        print("⬇️ Downloading model dari Google Drive...")
        url = f"https://drive.google.com/uc?id={GDRIVE_FILE_ID}"
        gdown.download(url, MODEL_PATH, quiet=False)
        print("✅ Model berhasil didownload!")

_model = None

def load_model():
    global _model
    if _model is None:
        download_model()
        _model = keras.models.load_model(MODEL_PATH)
        print("✅ Model loaded!")
    return _model

def preprocess_image(image):
    if image.mode != 'RGB':
        image = image.convert('RGB')
    image = image.resize(IMG_SIZE)
    img_array = np.array(image) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

def predict(image):
    model = load_model()
    img_array = preprocess_image(image)
    predictions = model.predict(img_array, verbose=0)
    pred_probs = predictions[0]
    pred_idx = np.argmax(pred_probs)
    predicted_class = CLASS_NAMES[pred_idx]
    confidence = float(pred_probs[pred_idx])
    top5_idx = np.argsort(pred_probs)[::-1][:5]
    top5 = [(CLASS_NAMES[i], float(pred_probs[i])) for i in top5_idx]
    return predicted_class, confidence, top5