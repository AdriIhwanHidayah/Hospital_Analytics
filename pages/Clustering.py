import streamlit as st
import plotly.express as px

from utils import load_data
from models.clustering import run_kmeans

st.set_page_config(
    page_title="K-Means Clustering",
    layout="wide"
)

st.title("K-Means Clustering")

df = load_data()

df = run_kmeans(df)

# ==========================
# KPI
# ==========================

c1,c2,c3,c4 = st.columns(4)

c1.metric(
    "Jumlah Pasien",
    len(df)
)

c2.metric(
    "Jumlah Cluster",
    df["Cluster"].nunique()
)

c3.metric(
    "Cluster Terbesar",
    df["Cluster"].value_counts().idxmax()
)

c4.metric(
    "Rata-rata Penagihan",
    f"Rp {df['Billing Amount'].mean():,.0f}"
)

st.divider()

# ==========================
# Scatter Plot
# ==========================

st.subheader("Visualisasi Hasil Clustering")

fig = px.scatter(
    df.sample(min(3000, len(df)), random_state=42),
    x="Age",
    y="Billing Amount",
    color=df.sample(min(3000, len(df)), random_state=42)["Cluster"].astype(str),
    title="Age vs Billing Amount"
)

st.plotly_chart(
    fig,
    width="stretch"
)

# ==========================
# Pie Chart
# ==========================

cluster = (
    df["Cluster"]
    .value_counts()
    .reset_index()
)

cluster.columns=[
    "Cluster",
    "Jumlah"
]

fig = px.pie(
    cluster,
    names="Cluster",
    values="Jumlah",
    hole=.5,
    title="Distribusi Cluster"
)

st.plotly_chart(
    fig,
    width="stretch"
)

# ==========================
# Boxplot
# ==========================

fig = px.box(
    df,
    x="Cluster",
    y="Billing Amount",
    color="Cluster",
    title="Jumlah Tagihan per Klaster"
)

st.plotly_chart(
    fig,
    width="stretch"
)

# ==========================
# Ringkasan Cluster
# ==========================

st.subheader("Karakteristik Tiap Cluster")

summary = (
    df.groupby("Cluster")[["Age","Billing Amount"]]
    .mean()
)

st.dataframe(
    summary,
    width="stretch"
)

# ==========================
# Dataset
# ==========================

st.subheader("Hasil Clustering")

st.dataframe(
    df[
        [
            "Age",
            "Billing Amount",
            "Medical Condition",
            "Cluster"
        ]
    ],
    width="stretch"
)   