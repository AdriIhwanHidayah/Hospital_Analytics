import streamlit as st
import plotly.express as px
import pandas as pd
from sklearn.linear_model import LinearRegression
from utils import load_data


def format_currency_compact(value):
    if pd.isna(value):
        return "Rp 0"

    value = float(value)
    abs_value = abs(value)

    if abs_value >= 1e15:
        return f"Rp {value / 1e15:,.2f} kuadriliun"
    if abs_value >= 1e12:
        return f"Rp {value / 1e12:,.2f} triliun"
    if abs_value >= 1e9:
        return f"Rp {value / 1e9:,.2f} miliar"
    if abs_value >= 1e6:
        return f"Rp {value / 1e6:,.2f} juta"
    if abs_value >= 1e3:
        return f"Rp {value / 1e3:,.2f} ribu"

    return f"Rp {value:,.0f}"


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


def build_regression_model(df_reg):
    feature_df = df_reg[["Age", "Stay_Duration", "Medical Condition"]].copy()
    feature_df["Age"] = pd.to_numeric(feature_df["Age"], errors="coerce")
    feature_df["Stay_Duration"] = pd.to_numeric(feature_df["Stay_Duration"], errors="coerce")
    feature_df = feature_df.dropna(subset=["Age", "Stay_Duration", "Medical Condition"])

    target = df_reg.loc[feature_df.index, "Estimated_Billing"]
    target = target.fillna(df_reg.loc[feature_df.index, "Billing Amount"])

    valid_idx = target.notna()
    feature_df = feature_df.loc[valid_idx]
    target = target.loc[valid_idx]

    X = pd.get_dummies(
        feature_df[["Age", "Stay_Duration", "Medical Condition"]],
        columns=["Medical Condition"]
    )
    y = target

    model = LinearRegression()
    model.fit(X, y)

    return model, X.columns


def predict_estimated_billing(model, feature_columns, age, stay_duration, medical_condition):
    input_df = pd.DataFrame(
        [{
            "Age": age,
            "Stay_Duration": stay_duration,
            "Medical Condition": medical_condition
        }]
    )

    encoded_input = pd.get_dummies(
        input_df,
        columns=["Medical Condition"]
    )
    encoded_input = encoded_input.reindex(columns=feature_columns, fill_value=0)
    return float(model.predict(encoded_input)[0])


model, feature_columns = build_regression_model(df)

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

fig.update_xaxes(
    tickprefix="Rp ",
    tickformat=",.0f"
)
fig.update_yaxes(
    tickprefix="Rp ",
    tickformat=",.0f"
)
fig.update_traces(
    hovertemplate=(
        "Billing Amount: %{x:,.0f}<br>"
        "Estimasi Billing: %{y:,.0f}<extra></extra>"
    )
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

fig.update_yaxes(
    tickprefix="Rp ",
    tickformat=",.0f"
)
fig.update_traces(
    hovertemplate=(
        "Lama Rawat: %{x}<br>"
        "Estimasi Billing: %{y:,.0f}<extra></extra>"
    )
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

fig.update_yaxes(
    tickprefix="Rp ",
    tickformat=",.0f"
)
fig.update_traces(
    hovertemplate=(
        "Umur: %{x}<br>"
        "Estimasi Billing: %{y:,.0f}<extra></extra>"
    )
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
    text=[format_currency_compact(value) for value in billing["Estimated_Billing"]]
)

fig.update_yaxes(
    tickprefix="Rp ",
    tickformat=",.0f"
)
fig.update_traces(
    textposition="outside",
    hovertemplate=(
        "Kondisi: %{x}<br>"
        "Rata-rata Estimasi Billing: %{y:,.0f}<extra></extra>"
    )
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


st.divider()

# ==========================
# Form Prediksi Billing
# ==========================

st.subheader("Prediksi Estimasi Billing")
st.caption("Masukkan umur, lama rawat, dan kondisi medis untuk memprediksi estimasi tagihan.")

with st.form("regresi_prediction_form"):
    col1, col2, col3 = st.columns(3)

    age = col1.number_input(
        "Age",
        min_value=0,
        max_value=120,
        value=35,
        step=1
    )

    stay_duration = col2.number_input(
        "Stay Duration",
        min_value=1,
        max_value=365,
        value=7,
        step=1
    )

    medical_condition = col3.selectbox(
        "Medical Condition",
        options=sorted(df["Medical Condition"].dropna().unique())
    )

    submitted = st.form_submit_button("Prediksi Estimasi Billing")

    if submitted:
        prediction = predict_estimated_billing(
            model,
            feature_columns,
            age,
            stay_duration,
            medical_condition
        )
        st.success(f"Estimasi Billing: Rp {prediction:,.0f}")
