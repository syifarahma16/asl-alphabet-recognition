import numpy as np
from PIL import Image
import tensorflow as tf
import os

MODEL_DIR = os.path.join(os.path.dirname(__file__), '..', 'models')
MODEL_PATH = os.path.join(MODEL_DIR, 'mobilenet_asl.keras')

CLASS_NAMES = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J',
               'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T',
               'U', 'V', 'W', 'X', 'Y', 'Z', 'del', 'nothing', 'space']

IMG_SIZE = (200, 200)

# Load model sekali saja saat module di-import
_model = None

def load_model():
    global _model
    if _model is None:  # hanya load kalau belum ada di memory
        _model = tf.keras.models.load_model(MODEL_PATH)
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