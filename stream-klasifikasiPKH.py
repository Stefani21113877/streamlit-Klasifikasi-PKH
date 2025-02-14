import streamlit as st
import pandas as pd
import numpy as np
import pickle
from datetime import datetime
import os

# Load model dan ordinal encoder
model = pickle.load(open('modelrf1.pkl', 'rb'))
ordinal_encoder = pickle.load(open('ordinal_encoder.pkl', 'rb'))

# Nama file Excel tempat data akan disimpan
excel_file = 'data_penerima_bantuan.xlsx'

# Pastikan file Excel ada dan memiliki header yang benar
def create_excel_file():
    if not os.path.exists(excel_file):
        df = pd.DataFrame(columns=["Nama", "Jumlah Anggota Keluarga", "Jumlah Tanggungan Anak", 
                                   "Pekerjaan", "Usia", "Penghasilan", "Prediksi", "Tanggal"])
        df.to_excel(excel_file, index=False)

create_excel_file()

# Tampilan utama aplikasi
st.title("Klasifikasi Kelayakan Masyarakat untuk Bantuan PKH")
st.markdown("---")

# Form input
with st.form("prediction_form"):
    nama = st.text_input("Nama Lengkap", placeholder="Masukkan Nama")
    jumlah_anggota = st.number_input("Jumlah Anggota Keluarga", min_value=1, step=1)
    jumlah_tanggungan = st.number_input("Jumlah Tanggungan Anak", min_value=0, step=1)
    pekerjaan = st.selectbox("Pekerjaan", ["Buruh", "Petani", "Penjahit", "PNS", "Tidak Bekerja"], index=None)
    usia = st.number_input("Usia", min_value=0, step=1)
    penghasilan = st.number_input("Penghasilan", min_value=0, step=1000)
    submit_button = st.form_submit_button("Submit")

# Proses prediksi
if submit_button:
    if nama and pekerjaan:
        pekerjaan_encoded = ordinal_encoder.transform([[pekerjaan]])[0][0]
        features = np.array([jumlah_anggota, jumlah_tanggungan, penghasilan, usia, pekerjaan_encoded]).reshape(1, -1)
        prediction = model.predict(features)
        output_text = "Sangat layak diberi bantuan" if prediction[0] == 1 else "Tidak layak diberi bantuan"
        
        # Simpan data ke dalam Excel
        new_data = pd.DataFrame([{
            "Nama": nama,
            "Jumlah Anggota Keluarga": jumlah_anggota,
            "Jumlah Tanggungan Anak": jumlah_tanggungan,
            "Pekerjaan": pekerjaan,
            "Usia": usia,
            "Penghasilan": penghasilan,
            "Prediksi": output_text,
            "Tanggal": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }])
        
        df = pd.read_excel(excel_file)
        df = pd.concat([df, new_data], ignore_index=True)
        df.to_excel(excel_file, index=False)
        
        # Tampilkan hasil prediksi
        st.success(f"Hasil Prediksi: {output_text}")
    else:
        st.warning("Harap isi semua field yang diperlukan.")

# Tombol untuk melihat data penerima bantuan
st.markdown("---")
st.subheader("Data Penerima Bantuan PKH")
if st.button("Lihat Data Penerima Bantuan"):
    df = pd.read_excel(excel_file)
    st.dataframe(df)

# Tombol untuk mengunduh data sebagai Excel
with open(excel_file, "rb") as f:
    st.download_button(label="Download Data dalam Excel", data=f, file_name="data_penerima_bantuan.xlsx", mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
