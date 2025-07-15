import streamlit as st
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image

# Konfigurasi halaman
st.set_page_config(page_title="NaturReal", layout="wide")

# CSS custom font judul
st.markdown(
    """
    <link href="https://fonts.googleapis.com/css2?family=Playfair+Display&family=Poppins&display=swap" rel="stylesheet">
    <style>
        .custom-title {
            font-family: 'Playfair Display', serif;
            font-size: 35px;
            font-weight: semibold;
            padding-top: 4px;
        }
    </style>
    """,
    unsafe_allow_html=True
)

# Logo dan judul
col1, col2 = st.columns([1, 10])
with col1:
    st.image("image/logo.png", width=60)
with col2:
    st.markdown("<div class='custom-title'>NaturReal</div>", unsafe_allow_html=True)

# Garis pemisah
st.markdown("---")

# Banner
st.image("image/banner.jpg", use_container_width=True)

# Load model
model = load_model('model/fine_tuned_model.keras')

# Mapping label
class_names = ['Naturalisme', 'Realisme']

# Jarak
st.markdown("<br>", unsafe_allow_html=True)

# Upload gambar
uploaded_file = st.file_uploader("Upload gambar lukisan", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Gambar di tengah
    col_left, col_center, col_right = st.columns([1, 1, 1])
    with col_center:
        img = Image.open(uploaded_file)
        st.image(img, caption="Gambar yang diupload", use_container_width=True)

        # Jarak
        st.markdown("<br>", unsafe_allow_html=True)

        # Tombol di tengah
        col_left, col_center, col_right = st.columns([1, 2, 1])
        with col_center:
            predict = st.button("Lihat Hasil Prediksi", use_container_width=True)

    # Hasil prediksi
    if predict:
        with st.spinner("Memproses..."):
                # Preprocessing
                img = img.resize((224, 224))
                img_array = image.img_to_array(img)
                img_array = np.expand_dims(img_array, axis=0) / 255.0

                # Prediksi
                prediction = model.predict(img_array)
                prob = float(prediction[0])

                # Penentuan label & confidence
                if prob > 0.5:
                    predicted_label = 1
                    confidence = prob
                else:
                    predicted_label = 0
                    confidence = 1 - prob

                predicted_class = class_names[predicted_label]

                # Tampilkan hasil
                st.markdown("<br>", unsafe_allow_html=True)
                st.subheader("Hasil Prediksi:")
                st.info(f"Lukisan ini diprediksi sebagai **{predicted_class}** dengan tingkat keyakinan sebesar **{confidence*100:.2f}%**")