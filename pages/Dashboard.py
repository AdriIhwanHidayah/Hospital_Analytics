import streamlit as st
import plotly.express as px
from utils import load_data

# ==================================================
# CONFIG
# ==================================================

st.set_page_config(
    page_title="Hospital Analytics Dashboard",
    page_icon="🏥",
    layout="wide"
)

# ==================================================
# LOAD DATA
# ==================================================

df = load_data()



# ==================================================
# HEADER
# ==================================================

st.title("Dashboard Analisis Data Rumah Sakit")

st.markdown("""
Dashboard ini menyajikan ringkasan informasi mengenai data pasien,
biaya perawatan, kondisi medis, serta visualisasi analisis untuk membantu
proses pengambilan keputusan.
""")

st.success("Dataset berhasil dimuat.")

# ==================================================
# KPI
# ==================================================

st.subheader("📌 Ringkasan Data")

col1, col2, col3 = st.columns(3)
col4, col5, col6 = st.columns(3)

col1.metric(
    "Total Pasien",
    f"{len(df):,}"
)

col2.metric(
    "Total Biaya Perawatan",
    f"Rp {df['Billing Amount'].sum():,.0f}"
)

col3.metric(
    "Rata-rata Biaya",
    f"Rp {df['Billing Amount'].mean():,.0f}"
)

col4.metric(
    "Rata-rata Lama Rawat",
    f"{df['Stay_Duration'].mean():.1f} Hari"
)

col5.metric(
    "Jumlah Rumah Sakit",
    df["Hospital"].nunique()
)

col6.metric(
    "Jumlah Dokter",
    df["Doctor"].nunique()
)

# ==================================================
# ROW 1
# ==================================================

left, right = st.columns(2)

with left:

    condition = (
        df["Medical Condition"]
        .value_counts()
        .reset_index()
    )

    condition.columns = [
    "Medical Condition",
    "Jumlah Pasien"
]

    fig = px.bar(
        condition,
        x="Medical Condition",
        y="Jumlah Pasien",
        color="Jumlah Pasien",
        text_auto=True,
        title="Distribusi Kondisi Medis Pasien"
    )

    st.plotly_chart(fig, width="stretch")

with right:

    fig = px.pie(
        df,
        names="Gender",
        hole=0.5,
        title="Jenis Kelamin Terdistribusi"
    )

    st.plotly_chart(fig, width="stretch")

# ==================================================
# ROW 2
# ==================================================

left, right = st.columns(2)

with left:

    hospital = (
        df.groupby("Hospital")["Billing Amount"]
        .sum()
        .sort_values(ascending=False)
        .head(10)
        .reset_index()
    )

    fig = px.bar(
        hospital,
        x="Billing Amount",
        y="Hospital",
        orientation="h",
        color="Billing Amount",
        title="Top 10 Rumah Sakit berdasarkan Penagihan"
    )

    st.plotly_chart(fig, width="stretch")

with right:

    fig = px.histogram(
        df,
        x="Age",
        nbins=20,
        title="Distribusi Umur Pasien"
    )

    st.plotly_chart(fig, width="stretch")

# ==================================================
# ROW 3
# ==================================================

left, right = st.columns(2)

with left:

    billing = (
        df.groupby("Medical Condition")["Billing Amount"]
        .mean()
        .reset_index()
    )

    fig = px.bar(
        billing,
        x="Medical Condition",
        y="Billing Amount",
        color="Billing Amount",
        text_auto=".2s",
        title="Rata-rata Penagihan berdasarkan Medical Condition"
    )

    st.plotly_chart(fig, width="stretch")

with right:

    stay = (
        df.groupby("Medical Condition")["Stay_Duration"]
        .mean()
        .reset_index()
    )

    fig = px.bar(
        stay,
        x="Medical Condition",
        y="Stay_Duration",
        color="Stay_Duration",
        text_auto=".2f",
        title="Rata-rata Lama Rawat berdasarkan Medical Condition"
    )

    st.plotly_chart(fig, width="stretch")

# ==================================================
# ROW 4
# ==================================================

st.subheader("Hubungan Umur dengan Billing Amount")

sample_df = df.sample(
    min(1000, len(df)),
    random_state=42
)

fig = px.scatter(
    sample_df,
    x="Age",
    y="Billing Amount",
    color="Medical Condition",
    hover_data=[
        "Gender",
        "Hospital",
        "Doctor"
    ],
    title="Usia vs Jumlah Tagihan"
)

st.plotly_chart(fig, width="stretch")

# ==================================================
# PREVIEW DATASET
# ==================================================

st.divider()

st.subheader("Preview Dataset")

st.dataframe(
    df.head(20),
    width="stretch",
    hide_index=True
)