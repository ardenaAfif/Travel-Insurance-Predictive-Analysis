# -*- coding: utf-8 -*-
"""submission_dicoding_ml_terapan__predictive_analysis.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1UKOR8_AC-hx03Um-REYosO-lJIR-Y3s8
"""

# Import library
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import altair as alt
import zipfile

from zipfile import ZipFile
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.utils import resample
from sklearn.utils import resample
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Unzip file
dataset_zip = 'travel-insurance-data.zip'
with zipfile.ZipFile(dataset_zip, 'r') as zip:
  zip.extractall('/content')

df = pd.read_csv('/content/TravelInsurancePrediction.csv')
df.head()

"""## Exploratory Data Analysis"""

df.info()

# Menghapus kolom 'Unnamed' karena tidak dibutuhkan
df_new = df.drop(["Unnamed: 0"], axis=1)

# Mangubah kolom 'Employment Type', 'GraduateOrNot', 'FrequentFlyer', 'EverTravelledAbroad' menjadi nilai 0 dan 1
df_new['GovernmentSector'] = df['Employment Type'].map({'Private Sector/Self Employed': 0, 'Government Sector':1})
df_new['GraduateOrNot'] = df['GraduateOrNot'].map({'Yes':1, 'No':0})
df_new['FrequentFlyer'] = df['FrequentFlyer'].map({'Yes':1, 'No':0})
df_new['EverTravelledAbroad'] = df['EverTravelledAbroad'].map({'Yes':1, 'No':0})

# Menghapus kolom Employment Type yang sudah diganti GovernmentSector
df_new.drop(["Employment Type"], axis=1, inplace=True)

df_new.head()

# Visualisasi Korelasi
plt.figure(figsize=(8,8))
_ = sns.heatmap(df_new.corr(), cmap='coolwarm', annot=True)

# Menganalisis data menggunakan visualisasi

# Set style for seaborn
sns.set(style="whitegrid")

# # Membuat subplot
plt.figure(figsize=(15, 20))

# Subplot 1: Jumlah TravelInsurance
plt.subplot(4, 2, 1)
sns.countplot(x='TravelInsurance', data=df_new)
plt.title('Jumlah TravelInsurance')

# Distribusi umur
plt.subplot(4, 2, 2)
sns.countplot(x='Age', data=df_new, hue='TravelInsurance')
plt.title("Distribusi Umur dengan 'TravelInsurance'")

# Pengguna Asuransi berdasarkan kategori GovernmentSector
plt.subplot(4, 2, 3)
sns.countplot(x='GovernmentSector', hue='TravelInsurance', data=df_new)
plt.title("Pengguna Asuransi berdasarkan 'GovernmentSector'")

# Pengguna asuransi berdasarkan kategori 'GraduateOrNot'
plt.subplot(4, 2, 4)
sns.countplot(x='GraduateOrNot', data=df_new, hue='TravelInsurance')
plt.title("Pengguna asuransi berdasarkan kategori 'GraduateOrNot'")

# Pengguna Asuransi berdasarkan 'FamilyMembers'
plt.subplot(4, 2, 5)
sns.countplot(x='FamilyMembers', hue='TravelInsurance', data=df_new)
plt.title("Pengguna Asuransi berdasarkan 'FamilyMembers'")

# Pengguna Asuransi berdasarkan 'ChronicDiseases'
plt.subplot(4, 2, 6)
sns.countplot(x='ChronicDiseases', data=df_new, hue='TravelInsurance')
plt.title("Pengguna Asuransi berdasarkan 'ChronicDiseases'")

# Pengguna Asuransi berdasarkan kategori FrequentFlyer
plt.subplot(4, 2, 7)
sns.countplot(x='FrequentFlyer', data=df_new, hue='TravelInsurance')
plt.title("Pengguna Asuransi berdasarkan 'FrequentFlyer'")

# Pengguna Asuransi berdasarkan kategori EverTravelledAbroad
plt.subplot(4, 2, 8)
sns.countplot(x='EverTravelledAbroad', hue='TravelInsurance', data=df_new)
plt.title("Pengguna Asuransi berdasarkan 'EverTravelledAbroad'")

# Adjust layout
plt.tight_layout()

plt.show()

"""Hasil dari Explanatory Data Analysis:

1. Pada distribusi umur dalam penggunaan TravelInsurance, bahwa umur dibawah 33 tahun cenderung tidak menggunakan Asuransi saat bepergian
2. Pelanggan yang telah melakukan perjalan lebih dari 2 kali, cenderung menggunakan Asuransi
3. Pelanggan yang sering bepergian keluar negeri juga menggunakan Asuransi Travel

## Data Pre-processing

1. **Resampling dataset** untuk mencapai keseimbangan jumlah data. Resampling diperlukan untuk mengatasi bias dalam prediksi akibat ketidakseimbangan kuantitas data.
2. **Membagi dataset** menjadi dua bagian, yaitu data latih sebesar 80% dan data uji sebesar 20%. Pembagian dataset penting untuk menguji performa model terlatih pada data baru. Dalam kasus dataset ini, rasio 80:20 dianggap optimal karena jumlah datanya masih dalam skala ribuan (1987 baris).
3. **Normalisasi** dengan mengubah skala data sehingga memiliki distribusi yang relatif seragam atau mendekati distribusi normal. Langkah standarisasi bertujuan untuk membuat fitur numerik memiliki skala yang seragam, memudahkan proses pelatihan model.

### 1. Resample Dataset
"""

# Membagi feature dan label
label = 'TravelInsurance'

features = ['Age', 'GraduateOrNot', 'AnnualIncome', 'FamilyMembers',
            'ChronicDiseases', 'FrequentFlyer', 'EverTravelledAbroad',
            'GovernmentSector']

print('Label Coloumn:', label)
print('Features Coloumn:', features)

# Cek jumlah data pada kolom label 'TravelInsurance'
pd.value_counts(df_new['TravelInsurance'])

# Resampling untuk menangani ketidakseimbangan data pada class

yes = df_new[df_new['TravelInsurance']==1]
no = df_new[df_new['TravelInsurance']==0]

# Melakukan Data Balancing pada label == 1
df_resampled = resample(yes, replace = True, n_samples=1277)

# Menggabungkan data resample kedalam dataFrame
df_balanced = pd.concat([no, df_resampled])

pd.value_counts(df_balanced['TravelInsurance'])

"""### 2. Membagi Dataset"""

X_train, X_test, y_train, y_test = train_test_split(df_balanced[features], df_balanced[label], test_size=0.2, random_state=21, shuffle=True, stratify=df_balanced[label])

print(y_train.groupby(y_train).count())
print(y_test.groupby(y_test).count())

"""### 3. Normalisasi menggunakan Min-Max Scaler"""

scaler = MinMaxScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.fit_transform(X_test)

"""## Mengembangkan model

Algoritma machine learning yang diterapkan untuk menangani permasalahan dalam proyek ini mencakup K-Nearest Neighbors (KNN) dan Gradient Boosting.

1. **K-Nearest Neighbors (KNN)**

    Algoritma KNN yang diimplementasikan menggunakan library scikit-learn memilih nilai hyperparameter k=8, hasil dari eksperimen yang melibatkan beberapa nilai k lainnya dalam rentang 1-10. Pemilihan nilai k=8 didasarkan pada evaluasi performa, di mana nilai ini memberikan hasil terbaik. Model KNN dilatih dengan menggunakan data yang telah melalui tahap *pre-processing*.

2. **Gradient Boosting**

    Penerapan model Gradient Boosting juga melibatkan penggunaan library scikit-learn, khususnya GradientBoostingClassifier. Model ini dilatih dengan data yang telah melewati proses pre-processing untuk memastikan konsistensi dan kualitas data yang masuk ke dalam model.

### 1. Pegembangan menggunakan KNN Model
"""

knn = KNeighborsClassifier()

# Melatih model KNN
knn.fit(X_train, y_train)

# Menyimpan score hasil akurasi pada baseline model
score_knn = pd.DataFrame(columns=['Latih', 'Test'], index=['KNN'])
score_knn.loc['KNN', 'Latih'] = knn.score(X_train, y_train)
score_knn.loc['KNN', 'Test'] = knn.score(X_test, y_test)

score_knn

"""#### Melakukan Improvement pada KNN model"""

# Hyperparameter grid
param_grid = {'n_neighbors': [3, 5, 7, 9],
              'p': [1, 2],
              'weights': ["uniform","distance"],
              'algorithm':["ball_tree", "kd_tree", "brute"],
              }

# Pencarian parameter terbaik dengan GridSearchCV
knn_improved = GridSearchCV(knn,
                            param_grid,
                            cv = 5,
                            verbose = 1,
                            n_jobs = -1,
                            scoring='accuracy',
                            )
knn_improved.fit(X_train, y_train)

# Menggunakan parameter terbaik ketika improvement KNN model
knn_improved = KNeighborsClassifier(**knn_improved.best_params_)

# Melatih KNN model dengan parameter terbaik
knn_improved.fit(X_train, y_train)

# Membandingkan Hasil akurasi KNN model sebelum improve dan setelah mendapat parameter terbaik
score = pd.DataFrame(columns=['Latih', 'Test'], index=['KNN', 'KNN Improved'])

model_dict = {'KNN': knn,
              'KNN Improved': knn_improved}

for name, model in model_dict.items():
  score.loc[name, 'Latih'] = model.score(X_train, y_train)
  score.loc[name, 'Test'] = model.score(X_test, y_test)

score

"""### 2. Pegembangan menggunakan Gradient Boosting Model"""

gradient_boosting = GradientBoostingClassifier()

# Melatih Gradient Boosting Model
gradient_boosting.fit(X_train, y_train)

# Menyimpan score hasil akurasi pada baseline model
score_gradient_boosting = pd.DataFrame(columns=['latih', 'uji'], index=['Gradient Boosting'])
score_gradient_boosting.loc['Gradient Boosting', 'latih'] = gradient_boosting.score(X_train, y_train)
score_gradient_boosting.loc['Gradient Boosting', 'uji'] = gradient_boosting.score(X_test, y_test)

score_gradient_boosting

# Hyperparameter grid
param_grid = {'n_estimators': [10, 50, 100, 200, 500, 750, 1000],
              'max_depth': [3, 5, 10],
              'min_samples_leaf': [np.random.randint(1,10)],
              'max_features': [None, 'sqrt', 'log2']
              }

# Pencarian parameter terbaik dengan GridSearchCV
gradient_boosting_improved = GridSearchCV(gradient_boosting,
                            param_grid,
                            cv = 5,
                            verbose = 1,
                            n_jobs = -1,
                            scoring='roc_auc',
                            )
gradient_boosting_improved.fit(X_train, y_train)

# Pilih parameter terbaik untuk Gradient Boosting
gb_improved = GradientBoostingClassifier(**gradient_boosting_improved.best_params_)

# Latih model final
gb_improved.fit(X_train, y_train)

# Membandingkan Hasil akurasi Gradient Boosting model sebelum improve dan setelah mendapat parameter terbaik
score = pd.DataFrame(columns=['Latih', 'Test'], index=['Gradient Boosting', 'Gradient Boosting Improved'])

model_dict = {'Gradient Boosting': gradient_boosting,
              'Gradient Boosting Improved': gb_improved}

for name, model in model_dict.items():
  score.loc[name, 'Latih'] = model.score(X_train, y_train)
  score.loc[name, 'Test'] = model.score(X_test, y_test)

score

"""## Model Evaluation

### 1. Confusion Matrix
"""

# Menggunakan model yang sudah diprediksi
y_pred = gb_improved.predict(X_test)

# Menghasilkan confusion matrix
cm = confusion_matrix(y_test, y_pred)

# Membuat heatmap untuk confusion matrix
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', annot_kws={"size": 16})
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()

"""### 3. Accuracy, Precision, recall, f1-score"""

print('Akurasi: ', round(accuracy_score(y_pred, y_test), 3) * 100, '%')
print('Presisi: ', round(precision_score(y_pred, y_test), 3)* 100, '%')
print('Recall: ', round(recall_score(y_pred, y_test), 2)* 100, '%')
print('F1-score: ', round(f1_score(y_pred, y_test), 3)* 100, '%')