# Laporan Travel Insurance Predictive Analysis - Ardena Afif

## Domain Proyek

Proyek ini bertujuan untuk memprediksi pelanggan yang kemungkinan akan membeli asuransi perjalanan.

Latar belakang proyek ini timbul karena kebutuhan untuk mengidentifikasi calon pelanggan yang memiliki potensi untuk membeli asuransi perjalanan. Asuransi perjalanan memberikan perlindungan kepada individu yang bepergian, melindungi mereka dari berbagai risiko seperti sakit, keterlambatan pesawat, atau kejadian tak terduga lainnya yang mungkin terjadi selama perjalanan mereka, terutama yang berkaitan dengan kondisi rumah mereka yang ditinggalkan. [[1]](https://kc.umn.ac.id/13580/)

Hasil dari proyek ini akan berupa model machine learning yang dapat berperan sebagai alat pendukung dalam pengambilan keputusan bagi perusahaan asuransi perjalanan. Hal ini menjadi relevan mengingat tren bisnis di bidang asuransi perjalanan diprediksi akan meningkat kembali setelah hampir mengalami kepunahan selama masa pandemi. Dengan pemulihan dalam industri penerbangan, asuransi perjalanan menjadi produk yang menarik bagi para pelancong, terutama mengingat risiko pandemi yang masih memerlukan waktu untuk kembali ke kondisi normal.

## Business Understanding

Perjalanan yang mencakup kunjungan ke lokasi tertentu dengan maksud tertentu telah menjadi kegiatan umum dan semakin dapat diakses oleh masyarakat luas dalam beberapa dekade terakhir. Penerbangan atau perjalanan udara kini menjadi bagian biasa dari mobilitas, memberikan kemudahan bagi banyak orang. Namun, perjalanan juga melibatkan risiko yang dapat mengganggu kenyamanan selama perjalanan dan setelah kembali ke tempat asal. Risiko-risiko ini, seperti kemungkinan sakit, kehilangan paspor, keterlambatan pesawat, atau masalah yang timbul di rumah yang ditinggalkan, menciptakan peluang bagi perusahaan asuransi perjalanan untuk menyediakan layanan perlindungan.

Perusahaan asuransi perjalanan perlu memiliki strategi pemasaran yang tepat untuk menawarkan jasanya kepada pelanggan potensial, sehingga upaya promosi dan pemasaran dapat dilakukan secara lebih efisien. Dalam rangka memenuhi kebutuhan tersebut, proyek ini bertujuan untuk mengembangkan model machine learning yang dapat mengklasifikasikan pelanggan dalam konteks pembelian asuransi perjalanan.

### Problem Statements

Berdasarkan konteks yang telah dijelaskan, dapat dirumuskan permasalahan utama sebagai berikut:

1. Bagaimana melakukan pra-pemrosesan data asuransi perjalanan agar menghasilkan dataset yang optimal bagi model machine learning dalam memprediksi keputusan pelanggan saat pembelian asuransi perjalanan?
2. Bagaimana merancang dan melatih model machine learning yang dapat efektif memprediksi keputusan pelanggan dalam pembelian asuransi perjalanan, dengan mempertimbangkan akurasi model minimal sebesar 80%?

### Goals

Adapun tujuan dari proyek ini adalah:

1. Melakukan pra-pemrosesan data asuransi perjalanan untuk menghasilkan dataset yang sesuai standar bagi model machine learning.
2. Membangun model machine learning yang dapat memprediksi keputusan pelanggan dengan tingkat akurasi setidaknya 85%.

### Solution statements
Agar tujuan di atas dapat tercapai, langkah-langkah solusi yang diusulkan adalah sebagai berikut:

1. Pra-pemrosesan Data:

    - Melakukan pemilihan kolom fitur berdasarkan korelasi dengan kolom target. 
    - Memisahkan dataset menjadi data latih (80%) dan data uji (20%). 
    - Melakukan normalisasi data untuk memastikan skala data yang seragam.

2. Pembuatan Model:
   - Menggunakan algoritma KNN (K-Nearest Neighbors) untuk menghasilkan model baseline. 
   - Mengimplementasikan algoritma Gradient Boosting untuk model baseline sebagai perbandingan. 
   - Menyusun strategi pengembangan model dengan melakukan tuning pada hyperparameter guna meningkatkan akurasi.

Solusi di atas diarahkan untuk memastikan model dapat memenuhi tujuan akurasi yang telah ditetapkan.

## Data Understanding
Data yang digunakan dalam proyek ini diperoleh dari platform Kaggle dan diunggah oleh pengguna bernama TejasTheBard dengan judul [Travel Insurance Prediction Data](https://www.kaggle.com/tejashvi14/travel-insurance-prediction-data).

Berdasarkan informasi metadata, dataset ini berasal dari basis data perusahaan perjalanan di India. Dataset TravelInsurancePrediction.csv yang telah diunduh terdiri dari 10 kolom dengan deskripsi sebagai berikut:

`Unnamed 0` : Indeks atau nomor baris.

`Age` : Umur pelanggan.

`Employment Type` : Sektor pelanggan bekerja (Pemerintah (Government Sector) atau Swasta (Private Sector/Self Employed').

`GraduateOrNot` : Status lulusan perguruan tinggi.

`AnnualIncome` : Pendapatan tahunan (Rupee).

`FamilyMembers` : Jumlah anggota keluarga.

`ChronicDiseases` : Status ada tidaknya penyakit kronis pelanggan (asma, diabetes, darah tinggi, dll).

`FrequentFlyer` : Status jika sering bepergian berdasarkan riwayat 2 tahun terakhir.

`EverTravelledAbroad`: Status bepergian ke luar negeri.

`TravelInsurance` : Status pelanggan membeli paket asuransi.


## Data Preparation

- ### Explanatory Data Analysis
  Visualisasi Korelasi antar varibel

  ![korelasi-attributes](../main/images/korelasi-attributes.png "korelasi-attributes")

  Menganalisis data menggunakan visualisasi

   ![eda](../main/images/eda.png "eda")

  Hasil dari Explanatory Data Analysis:
  1. Pada distribusi umur dalam penggunaan TravelInsurance, bahwa umur dibawah 33 tahun cenderung tidak menggunakan Asuransi saat bepergian 
  2. Pelanggan yang telah melakukan perjalan lebih dari 2 kali, cenderung menggunakan Asuransi 
  3. Pelanggan yang sering bepergian keluar negeri juga menggunakan Asuransi Travel

  - ### Resampling dataset
    **Resampling dataset** untuk mencapai keseimbangan jumlah data. Resampling diperlukan untuk mengatasi bias dalam prediksi akibat ketidakseimbangan kuantitas data. 
      
    ```
    # Membagi feature dan label
    label = 'TravelInsurance'

    features = ['Age', 'GraduateOrNot', 'AnnualIncome', 'FamilyMembers',
    'ChronicDiseases', 'FrequentFlyer', 'EverTravelledAbroad',
    'GovernmentSector']
    ```
    `label = 'TravelInsurance'`: Menyimpan kolom 'TravelInsurance' sebagai variabel target yang ingin diprediksi. Dalam proyek ini, sepertinya kita sedang mencoba memprediksi apakah pelanggan akan membeli asuransi perjalanan atau tidak. Oleh karena itu, 'TravelInsurance' adalah label atau target yang ingin diprediksi.

    ```
    features = ['Age', 'GraduateOrNot', 'AnnualIncome', 'FamilyMembers',
    'ChronicDiseases', 'FrequentFlyer', 'EverTravelledAbroad',
    'GovernmentSector']
    ```
    Menyimpan kolom-kolom yang akan digunakan sebagai fitur-fitur dalam model. Fitur-fitur ini adalah atribut-atribut yang diharapkan memiliki pengaruh terhadap keputusan pembelian asuransi perjalanan. Dalam hal ini, fiturnya mencakup informasi seperti usia pelanggan, status lulusan perguruan tinggi, pendapatan tahunan, jumlah anggota keluarga, status penyakit kronis, kebiasaan sering bepergian, riwayat bepergian ke luar negeri, dan sektor tempat pelanggan bekerja (pemerintah atau swasta).

    **Data sebelum di _Resampling_**
  
    ![before-resampling](../main/images/before-resampling.png "eda")
  
    **Data setelah di _Resampling_**
  
    ![after-resampling](../main/images/after-resampling.png "after-resampling")

- ### Membagi dataset
    ```
  X_train, X_test, y_train, y_test = train_test_split(df_balanced[features], 
                                                    df_balanced[label], 
                                                    test_size=0.2, 
                                                    random_state=21, 
                                                    shuffle=True, 
                                                    stratify=df_balanced[label])
  ```
  **Membagi dataset** menjadi dua bagian, yaitu data latih sebesar 80% dan data uji sebesar 20%. Pembagian dataset penting untuk menguji performa model terlatih pada data baru. Dalam kasus dataset ini, rasio 80:20 dianggap optimal karena jumlah datanya masih dalam skala ribuan (1987 baris).


- ### Normalisasi Data
    ```
    scaler = MinMaxScaler()

    X_train = scaler.fit_transform(X_train)
    X_test = scaler.fit_transform(X_test)
  ```
  **Normalisasi** dengan mengubah skala data sehingga memiliki distribusi yang relatif seragam atau mendekati distribusi normal. Langkah standarisasi bertujuan untuk membuat fitur numerik memiliki skala yang seragam, memudahkan proses pelatihan model.


## Modeling

Sebagaimana disebutkan dalam pernyataan solusi, proyek ini menggunakan dua model machine learning untuk menangani permasalahan yang dihadapi, yaitu KNN dan Gradient Boosting.

- **KNN**: Model KNN pada proyek ini akan menggunakan modul sklearn. Model ini akan dilatih dengan menggunakan data yang telah melalui proses pra-pemrosesan. Selanjutnya, pengembangan model KNN akan dilakukan dengan menerapkan GridSearchCV untuk mencari kombinasi hyperparameter terbaik.
   
    Hasil dari pelatihan dan pengujian model sebagai berikut:

    ![knn-model](../main/images/knn-model.png "knn-model")


- **Gradient Boosting**: Model Gradient Boosting juga akan menggunakan modul sklearn dengan GradientBoostingClassifier, dan akan dilatih dengan data yang telah melewati tahap pra-pemrosesan. Seperti model KNN, proses pengembangan model Gradient Boosting akan memanfaatkan GridSearchCV untuk mengidentifikasi kombinasi hyperparameter optimal.
  
    Hasil dari pelatihan dan pengujian model sebagai berikut:

    ![gb-model](../main/images/gb-model.png "gb-model")

## Evaluation

- ### Confusion Matrix
**Confusion Matrix** merupakan suatu tabel yang berisi empat notasi yaitu tp, tn, fp, fn. Notasi tp (true positive) dan tn (true negative) mencerminkan jumlah nilai positif dan negatif yang berhasil diprediksi dengan benar oleh model. Di sisi lain, notasi fp (false positive) dan fn (false negative) mengindikasikan jumlah nilai positif dan negatif yang diprediksi secara keliru oleh model. Meskipun matriks ini memberikan pemahaman yang relatif sederhana, namun kekurangannya terletak pada kurangnya informativitas untuk mengukur hasil secara mendalam, sehingga memerlukan analisis lebih lanjut.

![confusion-matrix](../main/images/confusion-matrix.png "confusion-matrix")

- ### Evaluation Matrix

Metrik evaluasi model adalah ukuran yang digunakan untuk menilai kinerja suatu model machine learning.

1. **_Accuracy (Akurasi)_**

   **Accuracy** mengukur sejauh mana model dapat memprediksi dengan benar keseluruhan kelas, baik positif maupun negatif.
2. **_Precision_**

   **Presisi** mengukur sejauh mana prediksi positif model benar, dengan fokus pada seberapa akurat model dalam mengidentifikasi kelas positif.
3. **_Recall_**

   **Recall** mengukur sejauh mana model dapat menemukan atau mengenali instance dari kelas positif. Ini menyoroti kemampuan model untuk mendeteksi keseluruhan kelas positif.
4. **_F1-Score_**

    **F1-score** adalah rata-rata harmonik dari precision dan recall. Metrik ini memberikan keseimbangan antara presisi dan recall, dan bermanfaat ketika ada ketidakseimbangan antara kelas positif dan negatif.

Hasil dari Evaluation Matrix :

![metrics-matrix](../main/images/metrics-matrix.png "metrics-matrix")


