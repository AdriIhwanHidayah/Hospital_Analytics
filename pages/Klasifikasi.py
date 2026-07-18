import streamlit as st
import pandas as pd
import plotly.express as px

from utils import load_data
from models.classification import train_classifier

st.set_page_config(
    page_title="Klasifikasi Rekomendasi Obat",
    layout="wide"
)

st.title("🧠 Analisis Klasifikasi Prediksi Obat Pasien (Random Forest)")

st.markdown("""
Halaman ini menampilkan hasil prediksi **Rekomendasi Obat (Medication)** pasien menggunakan
algoritma **Random Forest** berdasarkan karakteristik klinis dan administrasi pasien.
""")
df = load_data()

# Mengambil feature_names dari backend
model, feature_names, y_test, pred, accuracy, cm, report = train_classifier(df)

# =======================================
# KPI
# =======================================
c1, c2, c3 = st.columns(3)
c1.metric("Jumlah Data", len(df))
c2.metric("Jumlah Jenis Obat (Kelas)", df["Medication"].nunique())
c3.metric("Accuracy", f"{accuracy:.2%}")

st.divider()

# =======================================
# Confusion Matrix
# =======================================
st.subheader("Confusion Matrix")
cm_df = pd.DataFrame(cm)
fig = px.imshow(cm_df, text_auto=True, color_continuous_scale="Blues")
st.plotly_chart(fig, width="stretch")

# =======================================
# Feature Importance
# =======================================
st.subheader("Feature Importance")
importance = pd.DataFrame({
    "Feature": feature_names,
    "Importance": model.feature_importances_
})
importance = importance.sort_values("Importance", ascending=False)

fig = px.bar(importance, x="Feature", y="Importance", color="Importance", text_auto=".2f")
st.plotly_chart(fig, width="stretch")

# =======================================
# Classification Report
# =======================================
st.subheader("Classification Report")
report_df = pd.DataFrame(report).transpose()
st.dataframe(report_df, width="stretch")

# =======================================
# Prediksi Pasien
# =======================================
st.subheader("Simulasi Rekomendasi Obat Pasien")
col1, col2 = st.columns(2)

age = col1.number_input("Usia", 1, 100, 30)
billing = col1.number_input("Jumlah Tagihan", 0.0, value=10000.0)
stay = col2.number_input("Lama Rawat", 1, 30, 5)
insurance = col2.number_input("Asuransi Terenkripsi", 0, 10, 1)

kondisi_medis_dict = {
    "Arthritis": 0,
    "Asthma": 1,
    "Cancer": 2,
    "Diabetes": 3,
    "Hypertension": 4,
    "Obesity": 5
}
selected_condition = col2.selectbox("Kondisi Medis Pasien", options=list(kondisi_medis_dict.keys()))
condition_encoded = kondisi_medis_dict[selected_condition]

if st.button("Rekomendasikan Obat"):
    # Kirim langsung dalam bentuk Array List tanpa menyertakan nama kolom dataframe
    input_pasien = [[
        age,
        billing,
        stay,
        insurance,
        condition_encoded
    ]]

    hasil = model.predict(input_pasien)
    st.success(f"Rekomendasi Obat Berdasarkan Model : **{hasil[0]}**")