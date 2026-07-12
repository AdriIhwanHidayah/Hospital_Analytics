import streamlit as st
import plotly.express as px
import pandas as pd
from utils import load_data

st.set_page_config(
    page_title="Regresi Linear",
    layout="wide"
)

st.title("Analisis Regresi Linear")

st.markdown("""
Halaman ini menampilkan hasil analisis menggunakan algoritma **Regresi Linear**
untuk memprediksi estimasi biaya perawatan pasien berdasarkan karakteristik pasien.
""")

df = load_data()

# ==========================
# Sidebar Filter
# ==========================

st.sidebar.header("Filter Data")

condition = st.sidebar.multiselect(
    "Medical Condition",
    options=df["Medical Condition"].unique(),
    default=df["Medical Condition"].unique()
)

gender = st.sidebar.multiselect(
    "Gender",
    options=df["Gender"].unique(),
    default=df["Gender"].unique()
)

df = df[
    (df["Medical Condition"].isin(condition)) &
    (df["Gender"].isin(gender))
]

# ==========================
# KPI
# ==========================

c1,c2,c3,c4 = st.columns(4)

c1.metric(
    "Jumlah Pasien",
    f"{len(df):,}"
)

c2.metric(
    "Rata-rata Tagihan",
    f"Rp {df['Billing Amount'].mean():,.0f}"
)

c3.metric(
    "Rata-rata Estimasi",
    f"Rp {df['Estimated_Billing'].mean():,.0f}"
)

c4.metric(
    "Rata-rata Lama Rawat",
    round(df["Stay_Duration"].mean(),2)
)

st.divider()

# ==========================
# Scatter Aktual vs Prediksi
# ==========================

st.subheader("Penagihan Aktual vs Prediksi")

fig = px.scatter(
    df,
    x="Billing Amount",
    y="Estimated_Billing",
    color="Medical Condition",
    opacity=0.7,
    title="Jumlah Tagihan vs Perkiraan Tagihan"
)

st.plotly_chart(
    fig,
    width="stretch"
)

# ==========================
# Scatter Stay Duration
# ==========================

st.subheader("Hubungan Lama Rawat dengan Estimasi Tagihan")

fig = px.scatter(
    df,
    x="Stay_Duration",
    y="Estimated_Billing",
    color="Medical Condition",
    trendline="ols"
)

st.plotly_chart(
    fig,
    width="stretch"
)

# ==========================
# Scatter Age
# ==========================

st.subheader("Hubungan Umur dengan Estimasi Tagihan")

fig = px.scatter(
    df,
    x="Age",
    y="Estimated_Billing",
    color="Gender",
    trendline="ols"
)

st.plotly_chart(
    fig,
    width="stretch"
)

# ==========================
# Average Billing
# ==========================

st.subheader("Rata-rata Estimasi Tagihan")

billing = (
    df.groupby("Medical Condition")["Estimated_Billing"]
    .mean()
    .reset_index()
)

fig = px.bar(
    billing,
    x="Medical Condition",
    y="Estimated_Billing",
    color="Estimated_Billing",
    text_auto=".2s"
)

st.plotly_chart(
    fig,
    width="stretch"
)

# ==========================
# Tabel
# ==========================

st.divider()

st.subheader("Dataset Regresi")

st.dataframe(
    df[
        [
            "Age",
            "Stay_Duration",
            "Medical Condition",
            "Billing Amount",
            "Estimated_Billing"
        ]
    ],
    width="stretch"
)