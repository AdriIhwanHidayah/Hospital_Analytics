import streamlit as st

st.set_page_config(
    page_title="Hospital Analytics",
    page_icon="🏥",
    layout="wide"
)

st.title("Hospital Analytics")

st.markdown("""
## Selamat Datang

Aplikasi Hospital Analytics merupakan dashboard analitik yang dirancang untuk membantu proses eksplorasi dan analisis data pasien rumah sakit. Dashboard ini menyajikan informasi dalam bentuk visualisasi interaktif serta hasil penerapan beberapa metode Machine Learning untuk mendukung pengambilan keputusan berbasis data.

### Menu Utama

**Dashboard**  
Menampilkan ringkasan data, indikator utama (KPI), dan visualisasi statistik pasien.

**Regresi Linear**  
Menyajikan hasil prediksi estimasi biaya perawatan pasien berdasarkan karakteristik pasien.

**K-Means Clustering**  
Mengelompokkan pasien ke dalam beberapa cluster berdasarkan kemiripan karakteristik usia dan biaya perawatan.

**Klasifikasi**  
Menampilkan hasil klasifikasi kondisi medis pasien menggunakan algoritma Random Forest beserta evaluasi model.

Silakan pilih salah satu menu pada sidebar untuk memulai proses analisis data.
""")