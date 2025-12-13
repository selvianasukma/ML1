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
st.write("Masukkan nilai sesuai variabel yang digunakan saat training model.")

input_values = []
for feature in features:
    value = st.number_input(f"{feature.replace('_', ' ').title()}", min_value=0.0)
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

if hasattr(model, "coef_"):
    coef_df = pd.DataFrame({
        "Fitur": features,
        "Koefisien": model.coef_
    })
    st.table(coef_df)
    st.write("**Intercept:**", model.intercept_)
else:
    st.info("Informasi koefisien tidak tersedia untuk model ini")

st.caption("Model dilatih di Jupyter Notebook dan hanya di-load di Streamlit (tanpa training ulang).")