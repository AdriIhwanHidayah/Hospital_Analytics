import streamlit as st

st.set_page_config(
    page_title="Hospital Analytics",
    page_icon="🏥",
    layout="wide"
)

st.title("🏥 Hospital Analytics")

st.markdown("""
## Selamat Datang

Aplikasi ini digunakan untuk menganalisis data pasien menggunakan beberapa metode Machine Learning.

### Menu

📊 Dashboard

📈 Regresi Linear

🎯 K-Means Clustering

🧠 Klasifikasi

Silakan pilih menu di sidebar.
""")