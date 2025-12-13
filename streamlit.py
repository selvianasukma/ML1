import streamlit as st
import numpy as np
import pandas as pd
import pickle

st.set_page_config(page_title="Prediksi Produksi Padi", layout="wide")

st.title("🌾 Prediksi Produksi Padi")
st.write("Aplikasi ini **otomatis menggunakan model yang sudah dilatih** dan disimpan dari notebook *produksi_padi.ipynb*.")

# =============================
# Load Model Pickle
# =============================
@st.cache_resource
def load_model():
    with open("model_produksi_padi.pkl", "rb") as file:
        data = pickle.load(file)
    return data

try:
    model_data = load_model()

    # Jika pickle berisi dict (recommended)
    if isinstance(model_data, dict):
        model = model_data.get("model")
        features = model_data.get("features")
    else:
        # Jika pickle hanya berisi model
        model = model_data
        features = ["luas_panen", "tadah_hujan", "irigasi"]

    st.success("Model berhasil dimuat dari pickle")
except Exception as e:
    st.error("Gagal memuat model_produksi_padi.pkl")
    st.stop()

# =============================
# Input Prediksi
# =============================
st.header("Input Data")
st.write("Masukkan nilai sesuai **jumlah fitur saat model dilatih**.")

# Ambil jumlah fitur yang diharapkan model
if hasattr(model, "n_features_in_"):
    expected_features = model.n_features_in_
else:
    expected_features = len(features)

st.info(f"Model ini mengharapkan **{expected_features} fitur** sebagai input.")

input_values = []
for i in range(expected_features):
    label = features[i] if i < len(features) else f"fitur_{i+1}"
    value = st.number_input(label.replace('_', ' ').title(), min_value=0.0)
    input_values.append(value)

# =============================
# Prediksi

# =============================
if st.button("Prediksi Produksi"):
    input_array = np.array([input_values])
    prediction = model.predict(input_array)

    st.subheader("Hasil Prediksi")
    st.success(f"Perkiraan Produksi Padi: **{prediction[0]:,.2f}**")

# =============================
# Informasi Model
# =============================
st.header("Informasi Model")

# Aman dari error panjang tidak sama
if hasattr(model, "coef_"):
    coef = model.coef_

    # Jika koefisien 2D (misalnya multi-output), ambil baris pertama
    if len(coef.shape) > 1:
        coef = coef[0]

    # Samakan panjang fitur dan koefisien
    min_len = min(len(features), len(coef))

    coef_df = pd.DataFrame({
        "Fitur": features[:min_len],
        "Koefisien": coef[:min_len]
    })

    st.table(coef_df)
    st.write("**Intercept:**", model.intercept_)
else:
    st.info("Informasi koefisien tidak tersedia untuk model ini")

st.caption("Model dilatih di Jupyter Notebook dan hanya di-load di Streamlit (tanpa training ulang).")
