# Laporan Proyek Machine Learning (Prediksi Kanker Payudara) - Fahruddin A. Lebe

## Domain Proyek

Kanker payudara merupakan salah satu jenis kanker yang paling umum dan menjadi penyebab kematian kedua akibat kanker pada wanita di seluruh dunia. Deteksi dini adalah kunci untuk meningkatkan tingkat kelangsungan hidup pasien. Namun, proses diagnosis manual seringkali memakan waktu, subjektif, dan memerlukan keahlian tinggi dari patolog. Oleh karena itu, pengembangan sistem prediksi yang akurat dan efisien sangat penting untuk mendukung dokter dalam mengambil keputusan yang lebih cepat dan tepat. Proyek ini bertujuan untuk membangun model klasifikasi yang dapat memprediksi apakah seorang pasien menderita kanker payudara ganas (Malignant) atau jinak (Benign) menggunakan dataset *Breast Cancer Wisconsin (Diagnostic)*.

## Business Understanding

### Problem Statements

* **Subjektivitas Diagnosis:** Diagnosis kanker payudara secara manual dapat bervariasi antar ahli patologi, menyebabkan potensi misdiagnosis atau keterlambatan diagnosis.
* **Efisiensi Waktu:** Proses diagnosis manual yang kompleks membutuhkan waktu yang lama, yang dapat menunda penanganan pasien.
* **Kebutuhan Dukungan Klinis:** Dokter memerlukan alat bantu yang akurat dan cepat untuk memperkuat keputusan diagnosis mereka.

### Goals

* Membangun model klasifikasi yang mampu memprediksi diagnosis kanker payudara (ganas atau jinak) dengan akurasi tinggi.
* Mengembangkan solusi yang dapat membantu mengurangi subjektivitas dan mempercepat proses diagnosis.
* Menyediakan model yang dapat diintegrasikan ke dalam sistem pendukung keputusan klinis.

### Solution Statement

Untuk mencapai tujuan di atas, proyek ini akan mengimplementasikan solusi berbasis *machine learning* dengan langkah-langkah sebagai berikut:

1.  **Penggunaan Algoritma Klasifikasi:** Menggunakan algoritma *Logistic Regression* sebagai *baseline model* untuk klasifikasi biner. Algoritma ini dipilih karena kemudahannya dalam interpretasi dan performanya yang seringkali baik untuk masalah klasifikasi linear.
2.  **Penanganan Kelas Tidak Seimbang:** Menerapkan teknik *Synthetic Minority Over-sampling Technique (SMOTE)* pada data latih untuk mengatasi masalah ketidakseimbangan kelas antara diagnosis 'Malignant' dan 'Benign'. Hal ini bertujuan untuk mencegah bias model terhadap kelas mayoritas.
3.  **Evaluasi Model Terukur:** Mengukur kinerja model menggunakan metrik evaluasi yang relevan seperti *Accuracy*, *Precision*, *Recall*, dan *F1-Score*, serta memvisualisasikan *confusion matrix*. Metrik-metrik ini akan memberikan gambaran komprehensif mengenai kemampuan model dalam mengklasifikasikan kedua kelas.

## Data Understanding

Dataset yang digunakan dalam proyek ini adalah *Breast Cancer Wisconsin (Diagnostic) Dataset* yang dapat diunduh dari [Kaggle](https://www.kaggle.com/datasets/uciml/breast-cancer-wisconsin-data).

Dataset ini berisi 569 entri dan 33 kolom. Berikut adalah beberapa informasi penting tentang data:

* **Jumlah Data:** 569 baris.
* **Kondisi Data:**
    * Sebagian besar kolom adalah tipe data `float64`, yang mewakili fitur-fitur numerik dari sel kanker.
    * Ada satu kolom `id` dengan tipe `int64` yang kemungkinan adalah pengidentifikasi unik.
    * Kolom `diagnosis` bertipe `object` (string) yang merupakan label target.
    * Terdapat satu kolom `Unnamed: 32` yang seluruhnya `NaN` (null).
* **Distribusi Label Target (`diagnosis`):**
    * Benign (B): 357 kasus
    * Malignant (M): 212 kasus

Ini menunjukkan adanya ketidakseimbangan kelas, di mana jumlah kasus 'Benign' lebih banyak daripada 'Malignant'.

### Uraian Variabel (Fitur)

Dataset ini mengandung 30 fitur numerik yang dihitung dari citra digital massa payudara, ditambah kolom `id` dan `diagnosis`. Fitur-fitur ini menggambarkan karakteristik inti, rerata, dan "terburuk" (mean, standard error, dan "worst"/largest mean of three largest values) dari sepuluh karakteristik sel:

* `radius`: Jarak dari pusat ke perimeter
* `texture`: Standar deviasi nilai skala abu-abu
* `perimeter`: Keliling
* `area`: Luas area
* `smoothness`: Variasi panjang jari-jari
* `compactness`: Perimeter^2 / area - 1.0
* `concavity`: Tingkat keparahan bagian cekung
* `concave points`: Jumlah bagian cekung
* `symmetry`: Simetri
* `fractal_dimension`: "Approximation" dari garis pantai - 1

Setiap karakteristik diukur dalam tiga cara:
* `_mean`: Rerata dari nilai-nilai fitur
* `_se`: Kesalahan standar dari nilai-nilai fitur
* `_worst`: Rerata dari tiga nilai terbesar untuk setiap fitur

Kolom `diagnosis` adalah target, dengan `M` untuk Malignant (ganas) dan `B` untuk Benign (jinak).

### Visualisasi Distribusi Target

Visualisasi di bawah menunjukkan distribusi jumlah kasus 'Benign' dan 'Malignant', memperjelas adanya ketidakseimbangan data.

```python
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

# Asumsi df sudah dimuat dari data.csv
# df = pd.read_csv('data.csv')

sns.countplot(x='diagnosis', data=df)
plt.title('Distribusi Diagnosis (M = Malignant, B = Benign)')
plt.show()
```
hasilnya:
![picture 0](https://i.imgur.com/eA9Wvze.png)  

## Data Preparation

Tahap persiapan data adalah langkah krusial untuk memastikan kualitas dan kesesuaian data untuk proses pemodelan *machine learning*. Beberapa teknik data preparation yang dilakukan dalam proyek ini adalah sebagai berikut, disajikan secara berurutan:

1.  **Penghapusan Kolom Tidak Perlu dan Penanganan Nilai Kosong (NaN)**
    * **Proses:** Kolom `'Unnamed: 32'` dan `'id'` dihapus dari dataset.
    * **Alasan:** Kolom `'Unnamed: 32'` diketahui mengandung nilai `NaN` (null) di seluruh barisnya berdasarkan eksplorasi data (`df.info()`). Kolom ini tidak memberikan informasi yang berguna untuk prediksi dan dapat mengganggu proses pemodelan. Sementara itu, kolom `'id'` adalah pengidentifikasi unik pasien dan tidak relevan sebagai fitur prediktif. Penghapusan kedua kolom ini bertujuan untuk menyederhanakan data dan menghilangkan *noise*.

2.  **Pembuatan Fitur dan Label**
    * **Proses:** Dataset dibagi menjadi fitur (`X`) dan label target (`y`). Kolom `'diagnosis'` ditetapkan sebagai label (`y`), sementara semua kolom lainnya menjadi fitur (`X`).
    * **Alasan:** Pemisahan ini adalah langkah standar dalam *machine learning* untuk mendefinisikan variabel independen (fitur) yang akan digunakan untuk memprediksi variabel dependen (label).

3.  **Encoding Label Target**
    * **Proses:** Label `diagnosis` yang awalnya berupa string (`'M'` untuk Malignant dan `'B'` untuk Benign) diubah menjadi format numerik. Nilai `'M'` dipetakan menjadi `0` (Malignant) dan `'B'` dipetakan menjadi `1` (Benign).
    * **Alasan:** Sebagian besar algoritma *machine learning* memerlukan input numerik. Oleh karena itu, label kategori perlu dikonversi menjadi representasi numerik. Pemetaan ini juga konsisten dengan konvensi umum di mana `0` sering digunakan untuk kelas minoritas atau kelas negatif (dalam kasus ini, ganas).

4.  **Normalisasi Fitur**
    * **Proses:** Fitur-fitur dalam `X` dinormalisasi menggunakan `StandardScaler`. `StandardScaler` mengubah distribusi setiap fitur sehingga memiliki rata-rata nol (0) dan standar deviasi satu (1).
    * **Alasan:** Normalisasi sangat penting ketika fitur-fitur memiliki skala yang berbeda. Algoritma seperti *Logistic Regression* (dan banyak algoritma berbasis gradien lainnya) sensitif terhadap skala fitur. Tanpa normalisasi, fitur dengan rentang nilai yang lebih besar dapat mendominasi perhitungan jarak atau bobot, sehingga model mungkin tidak belajar dengan optimal.

5.  **Pembagian Data Latih dan Uji**
    * **Proses:** Data yang telah diproses (`X_scaled` dan `y`) dibagi menjadi data latih (80%) dan data uji (20%) menggunakan `train_test_split` dengan `random_state=42` untuk reproduktifitas.
    * **Alasan:** Pembagian data ini memungkinkan kita untuk melatih model pada sebagian data dan mengevaluasinya pada data yang belum pernah dilihat model sebelumnya. Ini adalah cara standar untuk mengestimasi kinerja model pada data baru dan mencegah *overfitting*.

6.  **Penanganan Kelas Tidak Seimbang dengan SMOTE**
    * **Proses:** Karena distribusi kelas target tidak seimbang (357 Benign vs 212 Malignant), teknik *Synthetic Minority Over-sampling Technique (SMOTE)* diterapkan pada *data latih* (`X_train`, `y_train`) untuk menyeimbangkan jumlah sampel di kedua kelas. SMOTE menghasilkan sampel sintetis dari kelas minoritas (`Malignant`).
    * **Alasan:** Ketidakseimbangan kelas dapat menyebabkan model bias terhadap kelas mayoritas, menghasilkan akurasi yang tinggi namun kinerja yang buruk pada kelas minoritas yang seringkali merupakan kelas yang paling penting untuk dideteksi (misalnya, kasus ganas dalam diagnosis kanker). SMOTE membantu menciptakan dataset latih yang lebih seimbang, memungkinkan model belajar karakteristik kedua kelas dengan lebih baik.

## Model Development

### Pembentukan Model Machine Learning

Dalam proyek ini, model klasifikasi *Logistic Regression* digunakan untuk memecahkan masalah prediksi kanker payudara. *Logistic Regression* adalah algoritma yang cocok untuk tugas klasifikasi biner dan memiliki interpretasi yang relatif mudah.

### Tahapan dan Parameter yang Digunakan

* **Algoritma:** *Logistic Regression* dari `sklearn.linear_model`.
* **Parameter:** Model diinisialisasi dengan `max_iter=1000`.
    * `max_iter`: Parameter ini menentukan jumlah iterasi maksimum yang digunakan oleh algoritma untuk konvergensi. Nilai `1000` dipilih untuk memastikan model memiliki cukup iterasi untuk menemukan solusi yang optimal.
* **Tahapan Pelatihan:** Model dilatih menggunakan data latih yang sudah diseimbangkan oleh SMOTE (`X_train_sm`, `y_train_sm`).

### Cara Kerja Algoritma Logistic Regression

**Logistic Regression** adalah algoritma klasifikasi linear yang digunakan untuk memprediksi probabilitas suatu kelas atau peristiwa. Meskipun namanya mengandung kata "regresi", ini merupakan model klasifikasi. Berikut adalah tahapan cara kerja Logistic Regression:

#### 1. Kombinasi Linear Fitur Input
Model mengambil fitur-fitur input (misalnya `radius_mean`, `texture_mean`, dll.) dan mengkombinasikannya secara linear. Setiap fitur memiliki bobot (`weight`) dan juga bias (`intercept`). Kombinasi linear ini dihitung dengan rumus:

\[
z = w_0 + w_1x_1 + w_2x_2 + \dots + w_nx_n
\]

- \( z \) : output linear (sering disebut *logit*)  
- \( w_0 \) : bias (intercept)  
- \( w_i \) : bobot untuk fitur ke-\( i \)  
- \( x_i \) : nilai fitur ke-\( i \)  

#### 2. Transformasi ke Probabilitas dengan Fungsi Sigmoid
Nilai \( z \) dikonversi menjadi probabilitas antara 0 dan 1 menggunakan fungsi sigmoid:

\[
P(y=1|X) = \frac{1}{1 + e^{-z}}
\]

- \( P(y=1|X) \) adalah probabilitas observasi termasuk dalam kelas positif (misalnya, *Benign*).  
- Semakin besar nilai \( z \), semakin mendekati 1 probabilitasnya.

#### 3. Pengambilan Keputusan Klasifikasi (Threshold)
Setelah mendapatkan probabilitas, keputusan klasifikasi diambil berdasarkan ambang batas (biasanya 0.5):

- Jika \( P(y=1|X) \geq 0.5 \), maka diklasifikasikan sebagai kelas positif (contoh: *Benign*)  
- Jika \( P(y=1|X) < 0.5 \), maka diklasifikasikan sebagai kelas negatif (contoh: *Malignant*)  

#### 4. Proses Pembelajaran (Optimasi Parameter)
Model mempelajari parameter \( w_i \) dan \( w_0 \) selama pelatihan dengan meminimalkan fungsi kerugian. Logistic Regression biasanya menggunakan **Log Loss** atau **Cross-Entropy Loss** sebagai fungsi kerugian, yang mengukur jarak antara prediksi probabilistik dan label aktual.

Proses optimasi dilakukan dengan algoritma seperti **Gradient Descent**, yang secara iteratif menyesuaikan parameter untuk mengurangi nilai fungsi kerugian. 

- Parameter `max_iter=1000` digunakan untuk membatasi jumlah iterasi maksimum agar proses konvergensi dapat tercapai secara stabil.

Dengan pendekatan ini, Logistic Regression dapat secara efektif memprediksi kelas pada data baru, terutama jika hubungan antara fitur dan target bersifat linear.


### Kelebihan dan Kekurangan Algoritma Logistic Regression

**Kelebihan:**
* **Sederhana dan Mudah Diinterpretasikan:** *Logistic Regression* memiliki persamaan matematis yang jelas, sehingga memudahkan pemahaman tentang bagaimana fitur memengaruhi probabilitas kelas. Koefisien model dapat diinterpretasikan sebagai bobot fitur.
* **Efisiensi Komputasi:** Algoritma ini relatif cepat dan efisien dalam hal komputasi, terutama untuk dataset berukuran sedang.
* **Performa Baik pada Data Linear:** Untuk masalah klasifikasi yang dapat dipisahkan secara linear, *Logistic Regression* seringkali memberikan performa yang kuat.
* **Memberikan Probabilitas:** Selain prediksi kelas, *Logistic Regression* juga menghasilkan probabilitas keanggotaan kelas, yang dapat berguna dalam pengambilan keputusan.

**Kekurangan:**
* **Asumsi Linearitas:** *Logistic Regression* mengasumsikan hubungan linear antara fitur dan log-odds dari target. Jika hubungan ini tidak linear, performa model bisa menurun.
* **Sensitif terhadap Outlier:** Model ini dapat terpengaruh secara signifikan oleh *outlier* dalam data, karena *outlier* dapat mengubah fungsi biaya secara drastis.
* **Tidak Cocok untuk Hubungan Kompleks:** Untuk masalah dengan hubungan fitur-target yang sangat kompleks atau non-linear, *Logistic Regression* mungkin tidak seakurat algoritma yang lebih canggih.

### Proses Improvement (jika hanya 1 algoritma)

Dalam proyek ini, hanya satu algoritma (*Logistic Regression*) yang digunakan sebagai *baseline model* sesuai dengan *Solution Statement*. Proses *improvement* terhadap model dilakukan melalui:

* **Penanganan Imbalance Data dengan SMOTE:** Ini adalah bentuk *improvement* krusial karena ketidakseimbangan kelas dapat secara drastis memengaruhi kemampuan model untuk mengklasifikasikan kelas minoritas dengan benar. Dengan menyeimbangkan data latih menggunakan SMOTE, model *Logistic Regression* memiliki kesempatan yang lebih baik untuk belajar dari kedua kelas, sehingga menghasilkan kinerja yang lebih robust dan tidak bias. Ini secara langsung bertujuan untuk meningkatkan *recall* pada kelas 'Malignant' yang krusial.

## Evaluation

### Metrik Evaluasi yang Digunakan

Untuk mengukur kinerja model *Logistic Regression* dalam memprediksi kanker payudara, beberapa metrik evaluasi klasifikasi yang relevan digunakan:

1.  **Confusion Matrix**
    * **Penjelasan:** *Confusion Matrix* adalah tabel yang menggambarkan performa model klasifikasi pada sekumpulan data uji yang label sebenarnya diketahui. Ini memvisualisasikan jumlah *True Positives (TP)*, *True Negatives (TN)*, *False Positives (FP)*, dan *False Negatives (FN)*.
        * **TP (True Positive):** Jumlah kasus `Benign` yang diprediksi dengan benar sebagai `Benign`.
        * **TN (True Negative):** Jumlah kasus `Malignant` yang diprediksi dengan benar sebagai `Malignant`.
        * **FP (False Positive):** Jumlah kasus `Malignant` yang salah diprediksi sebagai `Benign` (Kesalahan Tipe I).
        * **FN (False Negative):** Jumlah kasus `Benign` yang salah diprediksi sebagai `Malignant` (Kesalahan Tipe II).
    * **Tujuan:** Memberikan gambaran rinci tentang jenis-jenis kesalahan yang dilakukan model, yang sangat penting dalam konteks medis.

2.  **Accuracy**
    * **Formula:** $Accuracy = \frac{TP + TN}{TP + TN + FP + FN}$
    * **Penjelasan:** Mengukur proporsi total prediksi yang benar dari semua prediksi.
    * **Tujuan:** Memberikan gambaran umum tentang kinerja keseluruhan model. Namun, metrik ini bisa menyesatkan pada dataset yang tidak seimbang.

3.  **Precision**
    * **Formula:** $Precision = \frac{TP}{TP + FP}$
    * **Penjelasan:** Mengukur proporsi prediksi positif yang sebenarnya benar. Dalam konteks ini, untuk kelas 'Malignant' (0), ini berarti dari semua yang diprediksi sebagai 'Malignant', berapa banyak yang benar-benar 'Malignant'.
    * **Tujuan:** Penting untuk meminimalkan *false positives*. Dalam diagnosis kanker, *high precision* untuk kelas 'Malignant' berarti ketika model memprediksi ganas, kemungkinan besar itu memang ganas, mengurangi kekhawatiran yang tidak perlu atau tes lanjutan yang invasif.

4.  **Recall (Sensitivity)**
    * **Formula:** $Recall = \frac{TP}{TP + FN}$
    * **Penjelasan:** Mengukur proporsi kasus positif yang sebenarnya (aktual) yang berhasil diidentifikasi dengan benar oleh model. Untuk kelas 'Malignant' (0), ini berarti dari semua kasus 'Malignant' yang sebenarnya, berapa banyak yang berhasil dideteksi.
    * **Tujuan:** Metrik ini sangat kritis dalam diagnosis medis, karena meminimalkan *false negatives* adalah prioritas utama. Mendeteksi semua kasus 'Malignant' yang sebenarnya adalah yang paling penting untuk memastikan pasien mendapatkan perawatan yang tepat waktu.

5.  **F1-Score**
    * **Formula:** $F1-Score = 2 \times \frac{Precision \times Recall}{Precision + Recall}$
    * **Penjelasan:** Merupakan rata-rata harmonis dari *Precision* dan *Recall*.
    * **Tujuan:** Berguna ketika ada kebutuhan untuk menyeimbangkan antara *precision* dan *recall*, terutama pada dataset yang tidak seimbang.

### Hasil Proyek Berdasarkan Metrik Evaluasi

Setelah pelatihan model, evaluasi dilakukan pada data uji. Berikut adalah hasil *Classification Report* dan visualisasi *Confusion Matrix*:

![picture 1](https://i.imgur.com/6by1ZEO.png)  

Dari hasil di atas, model *Logistic Regression* menunjukkan kinerja yang sangat baik:

* **Accuracy:** 0.98. Ini berarti 98% dari total prediksi model adalah benar.
* **Precision (Malignant):** 0.98. Dari semua prediksi "Malignant" oleh model, 98% di antaranya benar-benar kasus "Malignant".
* **Recall (Malignant):** 0.98. Dari semua kasus "Malignant" yang sebenarnya, model berhasil mendeteksi 98% di antaranya.
* **F1-Score (Malignant):** 0.98. Menunjukkan keseimbangan yang sangat baik antara precision dan recall untuk kelas "Malignant".
* **Precision (Benign): 0.99.**
* **Recall (Benign): 0.99.**
* **F1-Score (Benign): 0.99.**

*Confusion Matrix* juga mengkonfirmasi hasil ini, dengan jumlah *True Negatives* dan *True Positives* yang sangat tinggi, serta jumlah *False Positives* dan *False Negatives* yang sangat rendah.

Metrik evaluasi yang digunakan sangat sesuai dengan konteks data dan *problem statement*. Dalam diagnosis kanker payudara, meminimalkan *false negatives* (pasien yang memiliki kanker ganas tetapi diprediksi jinak) adalah prioritas utama untuk menghindari penundaan pengobatan. Tingginya nilai *recall* untuk kelas 'Malignant' (0.98) menunjukkan bahwa model sangat efektif dalam mengidentifikasi kasus ganas. Meskipun demikian, *precision* yang tinggi untuk kelas 'Malignant' (0.98) juga penting untuk mengurangi *false positives* (pasien yang sehat tetapi diprediksi ganas), yang dapat menyebabkan kecemasan yang tidak perlu dan prosedur medis yang invasif. Dengan skor F1 yang tinggi (0.98) untuk kelas 'Malignant', model menunjukkan keseimbangan yang sangat baik antara kemampuan mendeteksi semua kasus ganas (recall) dan memastikan bahwa prediksi ganasnya benar (precision). Ini adalah indikasi bahwa model yang dikembangkan dapat menjadi alat pendukung keputusan yang andal dalam skenario klinis.