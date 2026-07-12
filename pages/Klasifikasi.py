import streamlit as st
import pandas as pd
import plotly.express as px

from utils import load_data
from models.classification import train_classifier

st.set_page_config(
    page_title="Klasifikasi",
    layout="wide"
)

st.title("🧠 Analisis Klasifikasi Random Forest")

st.markdown("""
Halaman ini menampilkan hasil klasifikasi kondisi medis pasien menggunakan
algoritma **Random Forest** berdasarkan karakteristik pasien seperti umur,
lama rawat, biaya perawatan, dan jenis asuransi.
""")
df = load_data()

model, X_test, y_test, pred, accuracy, cm, report = train_classifier(df)

# =======================================
# KPI
# =======================================

c1,c2,c3 = st.columns(3)

c1.metric(
    "Jumlah Data",
    len(df)
)

c2.metric(
    "Jumlah Kelas",
    df["Medical Condition"].nunique()
)

c3.metric(
    "Accuracy",
    f"{accuracy:.2%}"
)

st.divider()

# =======================================
# Confusion Matrix
# =======================================

st.subheader("Confusion Matrix")

cm_df = pd.DataFrame(cm)

fig = px.imshow(
    cm_df,
    text_auto=True,
    color_continuous_scale="Blues"
)

st.plotly_chart(
    fig,
    width="stretch"
)

# =======================================
# Feature Importance
# =======================================

st.subheader("Feature Importance")

importance = pd.DataFrame({
    "Feature":X_test.columns,
    "Importance":model.feature_importances_
})

importance = importance.sort_values(
    "Importance",
    ascending=False
)

fig = px.bar(
    importance,
    x="Feature",
    y="Importance",
    color="Importance",
    text_auto=".2f"
)

st.plotly_chart(
    fig,
    width="stretch"
)

# =======================================
# Classification Report
# =======================================

st.subheader("Classification Report")

report_df = pd.DataFrame(report).transpose()

st.dataframe(
    report_df,
    width="stretch"
)

# =======================================
# Prediksi
# =======================================

st.subheader("Prediksi Pasien")

col1,col2 = st.columns(2)

age = col1.number_input(
    "Usia",
    1,
    100,
    30
)

billing = col1.number_input(
    "Jumlah Tagihan",
    0.0,
    value=10000.0
)

stay = col2.number_input(
    "Lama Rawat",
    1,
    30,
    5
)

insurance = col2.number_input(
    "Asuransi Terenkripsi",
    0,
    10,
    1
)

if st.button("Prediksi"):

    hasil = model.predict(
        [[
            age,
            billing,
            stay,
            insurance
        ]]
    )

    st.success(
        f"Prediksi Medical Condition : **{hasil[0]}**"
    )