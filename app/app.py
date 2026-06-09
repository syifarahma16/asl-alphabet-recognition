import streamlit as st
import sys
import os
from PIL import Image

# Supaya bisa import dari folder app/
sys.path.append(os.path.dirname(__file__))
from utils import predict, CLASS_NAMES

# ============================================================
# PAGE CONFIG
# ============================================================
st.set_page_config(
    page_title="ASL Alphabet Recognition",
    page_icon="🤟",
    layout="centered"
)

# ============================================================
# HEADER
# ============================================================
st.title("🤟 ASL Alphabet Recognition")
st.markdown("**Klasifikasi Alfabet Bahasa Isyarat Amerika (ASL)**")
st.markdown("Upload gambar tangan atau gunakan kamera untuk mengenali huruf ASL (A–Z, del, nothing, space).")
st.divider()

# ============================================================
# INPUT: Upload atau Kamera
# ============================================================
st.subheader("📷 Input Gambar")

input_method = st.radio(
    "Pilih metode input:",
    ["Upload Gambar", "Kamera"],
    horizontal=True
)

image = None

if input_method == "Upload Gambar":
    uploaded_file = st.file_uploader(
        "Upload gambar tangan ASL",
        type=['jpg', 'jpeg', 'png'],
        help="Format yang didukung: JPG, JPEG, PNG"
    )
    if uploaded_file:
        image = Image.open(uploaded_file)

else:
    camera_input = st.camera_input("Ambil foto tangan kamu")
    if camera_input:
        image = Image.open(camera_input)

# ============================================================
# PREDIKSI
# ============================================================
if image is not None:
    st.divider()
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("🖼️ Gambar Input")
        st.image(image, use_container_width=True)
    
    with col2:
        st.subheader("🎯 Hasil Prediksi")
        
        with st.spinner("Memproses gambar..."):
            predicted_class, confidence, top5 = predict(image)
        
        # Tampilkan hasil utama
        if confidence >= 0.7:
            st.success(f"### {predicted_class}")
        elif confidence >= 0.4:
            st.warning(f"### {predicted_class}")
        else:
            st.error(f"### {predicted_class}")
        
        # Confidence bar
        st.metric(
            label="Confidence",
            value=f"{confidence*100:.1f}%"
        )
        
        # Warna berdasarkan confidence
        if confidence >= 0.7:
            st.progress(confidence, text="Tinggi ✅")
        elif confidence >= 0.4:
            st.progress(confidence, text="Sedang ⚠️")
        else:
            st.progress(confidence, text="Rendah ❌")
    
    st.divider()
    
    # Top 5 prediksi
    st.subheader("📊 Top 5 Prediksi")
    for i, (cls, prob) in enumerate(top5):
        col_rank, col_bar = st.columns([1, 4])
        with col_rank:
            if i == 0:
                st.markdown(f"**🥇 {cls}**")
            elif i == 1:
                st.markdown(f"**🥈 {cls}**")
            elif i == 2:
                st.markdown(f"**🥉 {cls}**")
            else:
                st.markdown(f"**{i+1}. {cls}**")
        with col_bar:
            st.progress(float(prob), text=f"{prob*100:.1f}%")

# ============================================================
# FOOTER
# ============================================================
st.divider()
st.markdown("""
<div style='text-align: center; color: gray; font-size: 12px;'>
    ASL Alphabet Recognition | Intelligo ID AI Bootcamp<br>
    Model: MobileNetV2 Transfer Learning | 29 Kelas
</div>
""", unsafe_allow_html=True)