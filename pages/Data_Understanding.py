import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from ucimlrepo import fetch_ucirepo

# Import custom bootstrap and navbar for styling
from component.bootstrap import bootstrap
from component.nav import navbar

# Menampilkan Navbar
navbar()

# Latar Belakang dan Tujuan
def html():
    bootstrap()
    st.header('Latar Belakang dan Tujuan')
    
    st.subheader('Latar Belakang')
    st.write("""
    Obesitas adalah masalah kesehatan global yang dapat menyebabkan berbagai penyakit serius, seperti diabetes, hipertensi, dan penyakit jantung. 
    Penelitian ini bertujuan untuk memahami faktor-faktor yang memengaruhi tingkat obesitas dengan menggunakan dataset yang mencakup berbagai fitur terkait kebiasaan makan, aktivitas fisik, dan faktor gaya hidup lainnya.
    Dataset ini membantu dalam membangun sistem prediksi tingkat obesitas berdasarkan input dari individu.
    """)

    st.subheader('Tujuan')
    st.write("""
    Tujuan dari penelitian ini adalah untuk mengidentifikasi tingkat obesitas berdasarkan kebiasaan makan dan kondisi fisik. 
    Penelitian ini bertujuan untuk membangun sebuah sistem yang dapat membantu memantau tingkat obesitas seseorang dan memberikan rekomendasi untuk memperbaiki gaya hidup terkait obesitas.
    """)

    st.image('asset/obesity_image.jpg', caption='Ilustrasi Tingkat Obesitas', width=500)

# Deskripsi Dataset dan Fitur
def desk_fitur():
    bootstrap()
    st.header('Deskripsi Dataset dan Fitur')
    st.write("""
    Dataset ini berisi informasi tentang individu yang berkaitan dengan gaya hidup mereka dan status obesitas mereka. 
    Berikut adalah fitur-fitur yang terkandung dalam dataset ini:
    """)

    fitur = """
    <div class="list-group">
        <ol class="list-group list-group-numbered">
            <li class="list-group-item"><strong>Gender</strong>: Jenis kelamin (Pria/Wanita).</li>
            <li class="list-group-item"><strong>Age</strong>: Usia dalam tahun.</li>
            <li class="list-group-item"><strong>Height</strong>: Tinggi badan dalam meter.</li>
            <li class="list-group-item"><strong>Weight</strong>: Berat badan dalam kilogram.</li>
            <li class="list-group-item"><strong>family_history_with_overweight</strong>: Apakah ada riwayat obesitas dalam keluarga (Ya/Tidak).</li>
            <li class="list-group-item"><strong>FAVC</strong>: Frekuensi konsumsi makanan berkalori tinggi (Kadang-kadang/Sering/Tidak pernah).</li>
            <li class="list-group-item"><strong>FCVC</strong>: Frekuensi konsumsi sayuran (Kadang-kadang/Sering/Tidak pernah).</li>
            <li class="list-group-item"><strong>NCP</strong>: Jumlah makan utama per hari.</li>
            <li class="list-group-item"><strong>CAEC</strong>: Frekuensi makan di antara waktu makan (Kadang-kadang/Sering/Tidak pernah).</li>
            <li class="list-group-item"><strong>SMOKE</strong>: Kebiasaan merokok (Ya/Tidak).</li>
            <li class="list-group-item"><strong>CH2O</strong>: Konsumsi air dalam liter per hari.</li>
            <li class="list-group-item"><strong>SCC</strong>: Pemantauan konsumsi kalori (Ya/Tidak).</li>
            <li class="list-group-item"><strong>FAF</strong>: Frekuensi aktivitas fisik per minggu.</li>
            <li class="list-group-item"><strong>TUE</strong>: Waktu penggunaan perangkat teknologi per hari dalam jam.</li>
            <li class="list-group-item"><strong>CALC</strong>: Frekuensi konsumsi alkohol (Kadang-kadang/Sering/Tidak pernah).</li>
            <li class="list-group-item"><strong>MTRANS</strong>: Mode transportasi yang digunakan (Jalan kaki, Sepeda, Mobil).</li>
        </ol>
    </div>
    """
    st.markdown(fitur, unsafe_allow_html=True)

# Deskripsi Kelas
def desk_kelas():
    bootstrap()
    st.header('Deskripsi Kelas')
    st.write("""
    Kolom target dalam dataset ini adalah <strong>NObeyesdad</strong>, yang mengklasifikasikan individu ke dalam salah satu kategori obesitas berikut:
    """)

    kelas = """
    <div class="list-group">
        <ol class="list-group list-group-numbered">
            <li class="list-group-item">Normal</li>
            <li class="list-group-item">Kelebihan berat badan I</li>
            <li class="list-group-item">Kelebihan berat badan II</li>
            <li class="list-group-item">Obesitas I</li>
            <li class="list-group-item">Obesitas II</li>
            <li class="list-group-item">Obesitas III</li>
        </ol>
    </div>
    """
    st.markdown(kelas, unsafe_allow_html=True)

# Memeriksa Missing Values dan Kualitas Data
def missval():
    st.subheader('Kualitas Data: Missing Values')
    st.write("""
    Pada tahap ini, kita akan memeriksa apakah dataset mengandung missing values atau tidak. 
    Dataset yang baik harus memiliki data lengkap tanpa ada nilai yang hilang untuk setiap fitur, 
    kecuali jika ada fitur yang secara eksplisit memang tidak terisi dalam beberapa entri.
    """)

def penjelasan_missval():
    st.write("""
    Berdasarkan analisis, dataset ini tidak mengandung missing values. Artinya, dataset sudah siap untuk diproses dan dianalisis tanpa memerlukan pengolahan data lebih lanjut terkait missing values.
    """)

# Menampilkan Data dan Visualisasi Distribusi
def display():
    st.markdown('<h1 align="center">DATA UNDERSTANDING</h1>', unsafe_allow_html=True)
    html()

    # Fetch dataset from UCI repository
    st.subheader('Dataset Obesitas')
    estimation_of_obesity_levels = fetch_ucirepo(id=544)
    data = estimation_of_obesity_levels.data
    X = data.features
    y = data.targets

    # Convert to DataFrame
    X_df = pd.DataFrame(X, columns=estimation_of_obesity_levels.features)
    y_df = pd.DataFrame(y, columns=estimation_of_obesity_levels.targets)

    # Gabungkan fitur dan target
    df = pd.concat([X_df, y_df], axis=1)
    st.write(df)

    # Deskripsi Data
    st.subheader('Deskripsi Fitur Data')
    dtypes = pd.DataFrame(df.dtypes, columns=["Tipe Data"])
    st.dataframe(dtypes)

    # Penjelasan Fitur
    desk_fitur()

    # Penjelasan Kelas
    desk_kelas()

    # Cek Missing Values
    missval()
    df_missing = df.isnull().sum()
    st.write(df_missing)
    penjelasan_missval()

    # Visualisasi Distribusi Data
    st.subheader('Distribusi Data')

    # Fitur untuk visualisasi
    features = list(X_df.columns)

    # Tentukan jumlah baris subplot
    num_features = len(features)
    num_rows = (num_features + 1) // 2  # Pembulatan ke atas

    fig, axs = plt.subplots(num_rows, 2, figsize=(12, num_rows * 4))
    for i, feature in enumerate(features):
        row = i // 2
        col = i % 2
        axs[row, col].hist(df[feature], bins=20, color='blue', alpha=0.7)
        axs[row, col].set_title(f'Distribusi {feature}')
        axs[row, col].set_xlabel(feature)
        axs[row, col].set_ylabel('Frekuensi')

    if num_features % 2 == 1:
        fig.delaxes(axs[num_rows - 1, 1])

    plt.tight_layout()
    st.pyplot(fig)

    # Visualisasi Kelas
    st.subheader('Distribusi Kelas (Target)')
    class_counts = df['NObeyesdad'].value_counts()
    st.bar_chart(class_counts)

# Menjalankan Tampilan Utama
display()
