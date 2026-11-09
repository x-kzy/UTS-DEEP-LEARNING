# -*- coding: utf-8 -*-
"""Ekspor otomatis 14 sel kode dari notebook final. Jalankan berurutan di runtime Colab.
Berkas ini dokumentasi kode; notebook .ipynb tetap referensi utama eksperimen.
"""

# ================================================================================
# SECTION 01 — sumber: notebook sel 2
# ================================================================================
# ==============================================================================
# SECTION 1: KONFIGURASI LINGKUNGAN, IMPOR LIBRARY & AKUISISI DATASET
# ==============================================================================
import os
import sys
import glob
import json
import shutil
import platform
import warnings
warnings.filterwarnings('ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import tensorflow as tf
import sklearn
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support,
    classification_report, confusion_matrix, log_loss
)

# Instalasi otomatis keras-tuner jika dijalankan di sesi Google Colab baru
try:
    import keras_tuner as kt
except ImportError:
    import subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "keras-tuner"])
    import keras_tuner as kt

# Mengatur random seed global untuk reprodusibilitas eksperimen
SEED = 42
np.random.seed(SEED)
tf.keras.utils.set_random_seed(SEED)

# Mencetak spesifikasi lingkungan eksekusi secara lengkap untuk dokumentasi ilmiah
gpu_devices = tf.config.list_physical_devices('GPU')
accel_info = f"GPU Tersedia ({len(gpu_devices)} device)" if gpu_devices else "CPU Standard (x86_64)"

env_spec_df = pd.DataFrame([
    {'Komponen Lingkungan': 'Python Runtime', 'Versi / Spesifikasi': sys.version.split()[0]},
    {'Komponen Lingkungan': 'Platform OS', 'Versi / Spesifikasi': platform.platform()},
    {'Komponen Lingkungan': 'TensorFlow / Keras', 'Versi / Spesifikasi': tf.__version__},
    {'Komponen Lingkungan': 'KerasTuner', 'Versi / Spesifikasi': kt.__version__},
    {'Komponen Lingkungan': 'Scikit-Learn', 'Versi / Spesifikasi': sklearn.__version__},
    {'Komponen Lingkungan': 'Pandas & NumPy', 'Versi / Spesifikasi': f"pandas {pd.__version__} | numpy {np.__version__}"},
    {'Komponen Lingkungan': 'Joblib Serializer', 'Versi / Spesifikasi': joblib.__version__},
    {'Komponen Lingkungan': 'Akselerator Komputasi', 'Versi / Spesifikasi': accel_info},
    {'Komponen Lingkungan': 'Global Random Seed', 'Versi / Spesifikasi': f"SEED = {SEED}"}
])
print("=== SPESIFIKASI LINGKUNGAN KOMPUTASI & REPRODUSIBILITAS EKSPERIMEN ===")
display(env_spec_df)

# Akuisisi dataset menggunakan kagglehub dengan fallback otomatis ke file lokal
if not (os.path.exists('laptops_cleaned_v1.csv') and os.path.exists('laptops_uncleaned.csv')):
    try:
        try:
            import kagglehub
        except ImportError:
            import subprocess
            subprocess.check_call([sys.executable, '-m', 'pip', 'install', '-q', 'kagglehub'])
            import kagglehub
        path = kagglehub.dataset_download("anshsahu111/laptop-specifications-and-prices-dataset")
        print("Path to dataset files:", path)
        for csv_file in glob.glob(os.path.join(path, "*.csv")):
            shutil.copy(csv_file, ".")
    except Exception as e:
        print(f"Info pemuatan lokal digunakan ({e})")
else:
    print("Path to dataset files: Direktori Kerja Lokal / Google Colab (File CSV tersedia)")

# Jika unduhan tidak berhasil, berikan galat yang dapat ditindaklanjuti.
missing_csv = [f for f in ['laptops_uncleaned.csv', 'laptops_cleaned_v1.csv'] if not os.path.exists(f)]
if missing_csv:
    raise FileNotFoundError(
        'Dataset belum tersedia: ' + ', '.join(missing_csv) + '. '
        'Unduh kedua CSV dari tautan Kaggle pada header notebook atau periksa akses jaringan Colab.'
    )

# Memuat kedua file CSV dari sumber Kaggle untuk audit menyeluruh
df_uncleaned = pd.read_csv('laptops_uncleaned.csv')
df_cleaned = pd.read_csv('laptops_cleaned_v1.csv')

print("\n[1] INFORMASI DATASET MENTAH (laptops_uncleaned.csv)")
print(f"Dimensi Data : {df_uncleaned.shape[0]} baris x {df_uncleaned.shape[1]} kolom")
display(df_uncleaned.head(3))

print("\n[2] INFORMASI DATASET EKSTRAKSI AWAL (laptops_cleaned_v1.csv)")
print(f"Dimensi Data : {df_cleaned.shape[0]} baris x {df_cleaned.shape[1]} kolom")
display(df_cleaned.head(3))

print("\n[3] RINGKASAN TIPE DATA & MISSING VALUES PADA laptops_cleaned_v1.csv:")
info_df = pd.DataFrame({
    'Tipe Data': df_cleaned.dtypes.astype(str),
    'Jumlah Non-Null': df_cleaned.notnull().sum(),
    'Jumlah Missing (NaN)': df_cleaned.isnull().sum(),
    'Persentase Missing (%)': (df_cleaned.isnull().sum() / len(df_cleaned) * 100).round(2)
})
display(info_df)

# ================================================================================
# SECTION 02 — sumber: notebook sel 5
# ================================================================================
# ==============================================================================
# SECTION 2: EKSPLORASI DATA (EDA) MENDALAM, AUDIT KUALITAS DATA, PEMBERSIHAN
#            DETERMINISTIK & PELABELAN KELAS HARGA (DUAL-CURRENCY INR & IDR)
# ==============================================================================
df = df_cleaned.copy()

# Satu acuan kurs untuk SEMUA tampilan: snapshot kurs indikatif pada Info per November 2026.
# Kurs tidak diunduh saat Run all supaya tabel, grafik, dan aplikasi tetap konsisten.
# Target klasifikasi tetap dari kolom price asli dalam INR; padanan IDR hanya untuk dibaca.
KURS_IDR = 192.549  # 1 INR ≈ Rp192,549; kurs indikatif per 2026-11-09.
TANGGAL_KURS = '2026-11-09'
SUMBER_KURS = 'Kurs indikatif INR/IDR yang diperiksa pada Info per November 2026 (bukan kurs transaksi)'

def parse_inr_price(nilai):
    """Baca nilai INR numerik maupun teks mentah seperti '₹2,99,990'."""
    if pd.isna(nilai):
        return np.nan
    if isinstance(nilai, (int, float, np.number)):
        return float(nilai)
    import re
    bersih = re.sub(r'[^0-9.]', '', str(nilai))
    try:
        return float(bersih)
    except ValueError:
        return np.nan

def format_inr(nilai):
    angka = parse_inr_price(nilai)
    if pd.isna(angka):
        return '-'
    return '₹' + f'{angka:,.0f}'.replace(',', '.')

def format_idr(nilai):
    if pd.isna(nilai):
        return '-'
    return 'Rp' + f'{float(nilai):,.0f}'.replace(',', '.')

def format_dual_harga(nilai_inr):
    angka = parse_inr_price(nilai_inr)
    if pd.isna(angka):
        return '-'
    return f'{format_inr(angka)} (≈ {format_idr(angka * KURS_IDR)})'

def format_juta_idr(nilai_inr):
    return f"Rp{float(nilai_inr) * KURS_IDR / 1e6:.2f}".replace('.', ',') + ' juta'

def rentang_kelas_dual(kelas):
    if int(kelas) == 0:
        return f'< {format_dual_harga(50_000)}'
    if int(kelas) == 1:
        return f'{format_dual_harga(50_000)} s.d. {format_dual_harga(85_000)}'
    if int(kelas) == 2:
        return f'> {format_dual_harga(85_000)}'
    raise ValueError('Kategori harga tidak dikenal')

print(f'ACUAN DUAL PRICE: 1 INR ≈ Rp{str(KURS_IDR).replace(chr(46), chr(44))}, tanggal {TANGGAL_KURS}.')
print('IDR hanya ekuivalen ilustratif; kelas dan target tetap memakai harga INR.')
print('Batas kelas:')
for kelas, nama in enumerate(['Entry-Level', 'Mid-Range', 'High-End']):
    print(f'  {nama}: {rentang_kelas_dual(kelas)}')

# ------------------------------------------------------------------------------
# 1. AUDIT SELISIH 1.020 BARIS (laptops_uncleaned.csv) VS 1.018 BARIS (laptops_cleaned_v1.csv)
# ------------------------------------------------------------------------------
# Catatan: Kolom 'index' pada laptops_cleaned_v1.csv merepresentasikan nomor baris
# berbasis 1-indexed dari file mentah (yaitu Unnamed: 0 + 2).
df_uncleaned_check = df_uncleaned.copy()
df_uncleaned_check['expected_clean_index'] = df_uncleaned_check['Unnamed: 0'] + 2
dropped_indices = sorted(list(set(df_uncleaned_check['expected_clean_index']) - set(df_cleaned['index'])))
dropped_rows_df = df_uncleaned_check[df_uncleaned_check['expected_clean_index'].isin(dropped_indices)][
    ['Unnamed: 0', 'expected_clean_index', 'model', 'price', 'processor', 'core', 'ram', 'memory', 'graphic_card', 'display', 'os', 'warrenty']
]
# Audit raw: tunjukkan harga sumber beserta padanan tanpa mengubah dataset mentah.
dropped_rows_df = dropped_rows_df.copy()
dropped_rows_df['Harga (INR / IDR, ilustratif)'] = dropped_rows_df['price'].apply(format_dual_harga)
print(f"=== [AUDIT 1] IDENTIFIKASI {len(dropped_rows_df)} BARIS YANG DIHAPUS DARI laptops_uncleaned.csv (1.020 -> 1.018 BARIS) ===")
display(dropped_rows_df)

# ------------------------------------------------------------------------------
# 2. PERBAIKAN TYPO KOLOM & KONVERSI ILUSTRATIF DUAL-CURRENCY (1 INR ≈ Rp192,549)
# ------------------------------------------------------------------------------
df = df.rename(columns={
    'resolutioin_height': 'resolution_height',
    'resolutioin_width': 'resolution_width',
    'warrenty': 'warranty'
})

# Catatan Metodologis: Ambang kelas target didefinisikan pada mata uang asli dataset (INR).
# Konversi ke Rupiah menggunakan KURS_IDR yang telah didefinisikan di awal Section 2.
# Angka berasal dari kurs acuan bertanggal dan bukan harga jual aktual di Indonesia.
# Seluruh kolom harga dan kategori tetap TIDAK digunakan sebagai fitur X.
df['price_idr'] = df['price'] * KURS_IDR
df['price_juta_idr'] = (df['price_idr'] / 1e6).round(2)

# Statistik deskriptif fitur numerik sebelum imputasi (mempertahankan jumlah non-null asli)
num_cols_raw = ['price', 'price_idr', 'price_juta_idr', 'rating', 'threads_num', 'core_num', 'ram_num',
                'memory_size', 'display_size', 'resolution_height', 'resolution_width', 'warranty']
print("\n=== [AUDIT 2] STATISTIK DESKRIPTIF FITUR NUMERIK & HARGA (SEBELUM IMPUTASI) ===")
stats_summary = df[num_cols_raw].agg(['count', 'mean', 'median', 'min', 'max', 'std']).T.round(2)
display(stats_summary)
print(f'Kolom price = INR asli, price_idr = INR × {KURS_IDR} (IDR ilustratif), '
      'price_juta_idr = IDR ilustratif dalam jutaan rupiah. ')
print('Ketiga kolom harga dikecualikan dari fitur pelatihan.')

# ------------------------------------------------------------------------------
# 3. AUDIT BARIS DUPLIKAT & AUDIT OUTLIER SPESIFIKASI EKSTREM
# ------------------------------------------------------------------------------
spec_check_cols = [
    'brand_name', 'processor_brand', 'iter', 'core_num', 'threads_num',
    'ram_num', 'memory_type', 'memory_size', 'gpu_brand', 'gpu_type',
    'os', 'display_size', 'resolution_height', 'resolution_width',
    'Touch_Screen', 'warranty', 'rating'
]
dup_audit_df = pd.DataFrame([
    {'Kategori Audit Duplikasi': 'Duplikat Persis Seluruh Kolom (Tanpa Kolom ID index)', 'Jumlah Baris': int(df.drop(columns=['index']).duplicated().sum()), 'Persentase (%)': round(df.drop(columns=['index']).duplicated().mean()*100, 2)},
    {'Kategori Audit Duplikasi': 'Duplikat Nama Model (Kolom model)', 'Jumlah Baris': int(df['model'].duplicated().sum()), 'Persentase (%)': round(df['model'].duplicated().mean()*100, 2)},
    {'Kategori Audit Duplikasi': 'Duplikat Kombinasi 17 Fitur Spesifikasi + Harga (price)', 'Jumlah Baris': int(df.duplicated(subset=spec_check_cols + ['price']).sum()), 'Persentase (%)': round(df.duplicated(subset=spec_check_cols + ['price']).mean()*100, 2)},
    {'Kategori Audit Duplikasi': 'Duplikat Kombinasi 17 Fitur Spesifikasi Teknis Saja', 'Jumlah Baris': int(df.duplicated(subset=spec_check_cols).sum()), 'Persentase (%)': round(df.duplicated(subset=spec_check_cols).mean()*100, 2)}
])
print("\n=== [AUDIT 3] RINGKASAN AUDIT DUPLIKASI DATA PADA laptops_cleaned_v1.csv ===")
display(dup_audit_df)

outlier_df = df[(df['ram_num'] >= 64) | (df['memory_size'] >= 4000)][
    ['index', 'brand_name', 'model', 'iter', 'ram_num', 'memory_size', 'gpu_brand', 'price', 'price_idr', 'price_juta_idr']
].sort_values(by=['ram_num', 'price'], ascending=False)
outlier_df = outlier_df.copy()
outlier_df['Harga (INR / IDR, ilustratif)'] = outlier_df['price'].apply(format_dual_harga)
print("\n=== [AUDIT 4] VERIFIKASI OUTLIER SPESIFIKASI EKSTREM (RAM >= 64 GB ATAU STORAGE >= 4.000 GB) ===")
display(outlier_df)

# ------------------------------------------------------------------------------
# 4. PEMBERSIHAN DETERMINISTIK & FEATURE ENGINEERING (TANPA IMPUTASI STATISTIK!)
# ------------------------------------------------------------------------------
print("\nNilai unik 'gpu_brand' sebelum dibersihkan:", df['gpu_brand'].unique())
print("Nilai unik 'os' sebelum dibersihkan       :", df['os'].unique())

df['gpu_brand'] = df['gpu_brand'].replace({'Nvidia': 'NVIDIA', 'Arm': 'ARM'})
df['os'] = df['os'].replace({
    'Windows 10': 'Windows 10/11',
    'Windows 11': 'Windows 10/11',
    'Windows': 'Windows 10/11',
    'Backlit Keyboard': 'Windows 10/11', # Koreksi pergeseran kolom pada Samsung Galaxy Book 4 Ultra
    'DOS 3.0': 'DOS',
    'Ubuntu': 'Linux'
})

print("Nilai unik 'gpu_brand' setelah dibersihkan:", df['gpu_brand'].unique())
print("Nilai unik 'os' setelah dibersihkan       :", df['os'].unique())

# Feature Engineering Deterministik per-baris: Generasi RAM (ram_type) & Kerapatan Piksel (PPI)
def extract_ram_type(ram_str):
    s = str(ram_str).upper()
    if 'LPDDR5X' in s: return 'LPDDR5X'
    elif 'LPDDR5' in s: return 'LPDDR5'
    elif 'LPDDR4' in s: return 'LPDDR4'
    elif 'DDR5' in s: return 'DDR5'
    elif 'DDR4' in s: return 'DDR4'
    else: return 'Other'

df['ram_type'] = df['ram'].apply(extract_ram_type)
df['ppi'] = np.sqrt(df['resolution_height']**2 + df['resolution_width']**2) / df['display_size']

# PENTING (ANTI-DATA LEAKAGE): Imputasi missing values pada 'core_num' (22 NaN) dan
# 'threads_num' (77 NaN) TIDAK dilakukan di Section 2 sebelum pembagian data, melainkan
# dilakukan di dalam pipeline praproses pada Section 3 yang di-fit HANYA pada X_train!
print(f"\nStatus Missing Values sebelum Train/Val/Test Split (tetap utuh agar bebas leakage): "
      f"threads_num={df['threads_num'].isna().sum()} NaN, core_num={df['core_num'].isna().sum()} NaN")

# ------------------------------------------------------------------------------
# 5. PELABELAN TARGET KLASIFIKASI KELAS HARGA (3 KELAS BERBASIS INR & ILUSTRASI IDR)
# ------------------------------------------------------------------------------
def categorize_laptop(price):
    if price < 50000:
        return 0
    elif price <= 85000:
        return 1
    else:
        return 2

df['category'] = df['price'].apply(categorize_laptop)

label_names = ['Entry-Level (Budget)', 'Mid-Range (Mainstream)', 'High-End (Flagship)']
label_names_dual = [
    f'Kelas 0: {label_names[0]} - {rentang_kelas_dual(0)}',
    f'Kelas 1: {label_names[1]} - {rentang_kelas_dual(1)}',
    f'Kelas 2: {label_names[2]} - {rentang_kelas_dual(2)}',
]
kelas_harga_df = pd.DataFrame([
    {'Kelas': i, 'Nama Kelas': label_names[i],
     'Ambang Harga Asli (INR) dan Ekuivalen (IDR)': rentang_kelas_dual(i)}
    for i in range(3)
])

print("\n=== DISTRIBUSI & ANALISIS RENTANG HARGA AKTUAL DATASET (INR ASLI & EKUIVALEN ILUSTRATIF IDR) ===")
dual_currency_rows = []
for idx in sorted(df['category'].unique()):
    sub = df[df['category'] == idx]
    min_inr = float(sub['price'].min())
    max_inr = float(sub['price'].max())
    med_inr = float(sub['price'].median())
    print(f"{label_names_dual[idx]}: {len(sub)} sampel ({len(sub)/len(df)*100:.2f}%)")
    print('   -> Harga aktual: '
          f"termurah {format_dual_harga(min_inr)} | "
          f"median {format_dual_harga(med_inr)} | "
          f"termahal {format_dual_harga(max_inr)}")
    dual_currency_rows.append({
        'Kelas Target': f"Kelas {idx} - {label_names[idx]}",
        'Ambang Kelas (INR / IDR)': rentang_kelas_dual(idx),
        'Harga Minimum (INR / IDR)': format_dual_harga(min_inr),
        'Median Harga (INR / IDR)': format_dual_harga(med_inr),
        'Harga Maksimum (INR / IDR)': format_dual_harga(max_inr),
        'Jumlah Sampel': f"{len(sub)} ({len(sub)/len(df)*100:.2f}%)",
    })

dual_currency_df = pd.DataFrame(dual_currency_rows)
display(kelas_harga_df)
display(dual_currency_df)

# ------------------------------------------------------------------------------
# 6. TIGA VISUALISASI EKSPLORASI DATA (EDA) LENGKAP
# ------------------------------------------------------------------------------
# Visualisasi 2A: Distribusi Kelas Harga, Boxplot RAM per Kelas, dan Proporsi GPU
fig, axes = plt.subplots(1, 3, figsize=(17, 5.0))
sns.countplot(data=df, x='category', hue='category', palette='viridis', legend=False, ax=axes[0])
axes[0].set_xticks([0, 1, 2])
harga_ticks_dual = [
    f'Entry-Level\n(< ₹50rb / < {format_juta_idr(50_000)})',
    f'Mid-Range\n(₹50–85rb / {format_juta_idr(50_000)}–{format_juta_idr(85_000)})',
    f'High-End\n(> ₹85rb / > {format_juta_idr(85_000)})',
]
axes[0].set_xticklabels(harga_ticks_dual, fontsize=7.5)
axes[0].set_title('[A] Distribusi Kelas Harga Laptop (INR Asli & Ilustrasi IDR)', fontweight='bold')
axes[0].set_xlabel(f'Kelas harga dalam INR / IDR ilustratif (1 INR ≈ Rp192,549; {TANGGAL_KURS})')
axes[0].set_ylabel('Jumlah Laptop')

sns.boxplot(data=df, x='category', y='ram_num', hue='category', palette='viridis', legend=False, ax=axes[1])
axes[1].set_xticks([0, 1, 2])
axes[1].set_xticklabels(harga_ticks_dual, fontsize=7.5)
axes[1].set_title('[B] Distribusi Kapasitas RAM (GB) per Kelas Harga', fontweight='bold')
axes[1].set_xlabel('Kelas Harga Target')
axes[1].set_ylabel('Kapasitas RAM (GB)')

sns.countplot(data=df, x='gpu_type', hue='category', palette='viridis', ax=axes[2])
axes[2].set_title('[C] Proporsi Tipe GPU terhadap Kelas Harga', fontweight='bold')
axes[2].set_xlabel('Tipe GPU')
axes[2].set_ylabel('Jumlah Laptop')
axes[2].legend(title='Kelas harga (INR / IDR ilustratif)', labels=[
    f'Entry-Level (< ₹50rb / < {format_juta_idr(50_000)})',
    f'Mid-Range (₹50–85rb / {format_juta_idr(50_000)}–{format_juta_idr(85_000)})',
    f'High-End (> ₹85rb / > {format_juta_idr(85_000)})'
], fontsize=8, title_fontsize=8)
plt.tight_layout()
plt.show()

# Visualisasi 2B: Histogram & KDE Seluruh 10 Fitur Numerik Spesifikasi Laptop
num_features_eda = [
    'rating', 'threads_num', 'core_num', 'ram_num', 'memory_size',
    'display_size', 'resolution_height', 'resolution_width', 'ppi', 'warranty'
]
fig, axes = plt.subplots(2, 5, figsize=(18, 7.2))
axes = axes.flatten()
palette_hist = ['#1b7837', '#2166ac', '#2166ac', '#762a83', '#e08214', '#d6604d', '#4393c3', '#4393c3', '#1b7837', '#8c510a']
for idx, col in enumerate(num_features_eda):
    vals = df[col].dropna()
    sns.histplot(vals, kde=True, ax=axes[idx], color=palette_hist[idx], edgecolor='black', bins=18)
    axes[idx].axvline(vals.mean(), color='#d73027', linestyle='--', linewidth=1.4, label=f"Mean: {vals.mean():.1f}")
    axes[idx].axvline(vals.median(), color='#1a1a1a', linestyle=':', linewidth=1.4, label=f"Med: {vals.median():.1f}")
    na_cnt = df[col].isna().sum()
    na_str = f" ({na_cnt} NaN)" if na_cnt > 0 else ""
    axes[idx].set_title(f"{col}{na_str}", fontweight='bold', fontsize=10)
    axes[idx].legend(fontsize=7.5, loc='upper right')
    axes[idx].grid(alpha=0.25)
fig.suptitle('DISTRIBUSI HISTOGRAM & KURVA DENSITAS (KDE) SELURUH 10 FITUR NUMERIK SPESIFIKASI LAPTOP',
             fontsize=12.5, fontweight='bold', y=1.02)
plt.tight_layout()
plt.show()

# Visualisasi 2C: Correlation Heatmap Fitur Numerik terhadap Harga (INR) & Kelas Target
fig, ax = plt.subplots(figsize=(11.5, 8.5))
corr_cols = num_features_eda + ['price', 'category']
sns.heatmap(df[corr_cols].corr(), annot=True, fmt='.2f', cmap='RdBu_r', vmin=-0.5, vmax=1.0,
            linewidths=0.6, annot_kws={'size': 8.5, 'weight': 'bold'}, ax=ax)
ax.set_title('KORELASI FITUR TERHADAP HARGA ASLI INR (IDR = INR × KURS TETAP) & KELAS TARGET',
             fontweight='bold', fontsize=11.5, pad=12)
plt.tight_layout()
plt.show()
# Harga dalam INR dan IDR memiliki korelasi sama karena IDR = INR × konstanta positif.
# Kami sengaja menampilkan kolom price (INR) saja pada heatmap untuk menghindari duplikasi.


# ================================================================================
# SECTION 03 — sumber: notebook sel 8
# ================================================================================
# ==============================================================================
# SECTION 3: PEMBAGIAN DATA (TRAIN/VAL/TEST) & PIPELINE PRAPROSES BEBAS LEAKAGE
# ==============================================================================
# Daftar 10 fitur numerik dan 9 fitur kategorikal spesifikasi teknis
num_features = [
    'rating', 'threads_num', 'core_num', 'ram_num', 'memory_size',
    'display_size', 'resolution_height', 'resolution_width', 'ppi', 'warranty'
]
cat_features = [
    'brand_name', 'processor_brand', 'iter', 'ram_type',
    'memory_type', 'gpu_brand', 'gpu_type', 'os', 'Touch_Screen'
]

# Menyusun matriks fitur X (TANPA mengikutsertakan 'price', 'price_idr', 'price_juta_idr')
# Perhatikan bahwa X masih memiliki 77 NaN pada threads_num dan 22 NaN pada core_num!
X = df[num_features + cat_features].copy()
X['Touch_Screen'] = X['Touch_Screen'].astype(str)
y = df['category'].values

# 1. Pembagian data Stratified: 70% Train (712), 15% Validation (153), 15% Test (153)
# dilakukan SEBELUM proses imputasi missing values maupun normalisasi StandardScaler!
X_train, X_temp, y_train, y_temp = train_test_split(
    X, y, test_size=0.30, stratify=y, random_state=SEED
)
X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.50, stratify=y_temp, random_state=SEED
)

print(f"Jumlah Data Latih (Train - 70%)      : {X_train.shape[0]} sampel | Distribusi Kelas: {np.bincount(y_train).tolist()}")
print(f"Jumlah Data Validasi (Val - 15%)     : {X_val.shape[0]} sampel | Distribusi Kelas: {np.bincount(y_val).tolist()}")
print(f"Jumlah Data Uji (Test - 15%)         : {X_test.shape[0]} sampel | Distribusi Kelas: {np.bincount(y_test).tolist()}")

print("\n=== SEBARAN MISSING VALUES PADA MASING-MASING SUBSET SEBELUM IMPUTASI ===")
split_na_df = pd.DataFrame([
    {'Subset Data': 'X_train (712 sampel)', 'Missing threads_num': int(X_train['threads_num'].isna().sum()), 'Missing core_num': int(X_train['core_num'].isna().sum()), 'Total NaN': int(X_train.isna().sum().sum())},
    {'Subset Data': 'X_val (153 sampel)',   'Missing threads_num': int(X_val['threads_num'].isna().sum()),   'Missing core_num': int(X_val['core_num'].isna().sum()),   'Total NaN': int(X_val.isna().sum().sum())},
    {'Subset Data': 'X_test (153 sampel)',  'Missing threads_num': int(X_test['threads_num'].isna().sum()),  'Missing core_num': int(X_test['core_num'].isna().sum()),  'Total NaN': int(X_test.isna().sum().sum())}
])
display(split_na_df)

# ------------------------------------------------------------------------------
# 2. DEFINISI CUSTOM TRANSFORMER BEBAS LEAKAGE: GroupMedianImputer
# ------------------------------------------------------------------------------
# Custom transformer disimpan sebagai modul agar objek joblib bisa dimuat di proses lain.
from pathlib import Path
Path('preprocessors.py').write_text('# preprocessors.py - pemrosesan fitur yang dapat diimpor pada Colab/Spaces.\nimport numpy as np\nimport pandas as pd\nfrom sklearn.base import BaseEstimator, TransformerMixin\n\nclass GroupMedianImputer(BaseEstimator, TransformerMixin):\n    """\n    Mempelajari median fitur numerik (\'core_num\', \'threads_num\') per kelompok seri\n    prosesor (\'iter\') HANYA dari data latih (X_train) saat .fit() dipanggil.\n    Jika suatu kelompok prosesor memiliki kurang dari min_samples observasi non-null\n    pada X_train, maka digunakan fallback median global dari X_train.\n    """\n    def __init__(self, group_col=\'iter\', target_cols=(\'core_num\', \'threads_num\'), min_samples=3):\n        self.group_col = group_col\n        self.target_cols = target_cols\n        self.min_samples = min_samples\n\n    def fit(self, X, y=None):\n        X_df = pd.DataFrame(X).copy()\n        self.group_medians_ = {}\n        self.global_medians_ = {}\n        for col in self.target_cols:\n            self.global_medians_[col] = float(X_df[col].median())\n            col_map = {}\n            for grp_val, grp_df in X_df.groupby(self.group_col):\n                valid_vals = grp_df[col].dropna()\n                if len(valid_vals) >= self.min_samples:\n                    col_map[grp_val] = float(valid_vals.median())\n            self.group_medians_[col] = col_map\n        return self\n\n    def transform(self, X):\n        X_df = pd.DataFrame(X).copy()\n        for col in self.target_cols:\n            grp_map = self.group_medians_[col]\n            glob_med = self.global_medians_[col]\n            mask_na = X_df[col].isna()\n            if mask_na.any():\n                X_df.loc[mask_na, col] = (\n                    X_df.loc[mask_na, self.group_col].map(grp_map).fillna(glob_med)\n                )\n        return X_df\n', encoding='utf-8')
from preprocessors import GroupMedianImputer

# Membandingkan nilai imputasi GroupMedianImputer (fit pada X_train) vs SimpleImputer Global (fit pada X_train)
group_imputer_demo = GroupMedianImputer(group_col='iter', target_cols=('core_num', 'threads_num'), min_samples=3)
group_imputer_demo.fit(X_train)

global_core_med_train = float(X_train['core_num'].median())
global_thr_med_train  = float(X_train['threads_num'].median())

iters_with_missing = ['Core Ultra 5', 'Core Ultra 7', 'Core Ultra 9', 'Core i7', 'Core i9', 'Ryzen 9', 'Snapdragon', 'M1', 'M2', 'Other']
imp_comp_rows = []
for it_name in iters_with_missing:
    c_grp = group_imputer_demo.group_medians_['core_num'].get(it_name, global_core_med_train)
    t_grp = group_imputer_demo.group_medians_['threads_num'].get(it_name, global_thr_med_train)
    imp_comp_rows.append({
        'Seri Prosesor (iter)': it_name,
        'GroupMedianImputer core_num (X_train)': c_grp,
        'SimpleImputer Global core_num (X_train)': global_core_med_train,
        'GroupMedianImputer threads_num (X_train)': t_grp,
        'SimpleImputer Global threads_num (X_train)': global_thr_med_train
    })
imp_comp_df = pd.DataFrame(imp_comp_rows)
print("\n=== PERBANDINGAN NILAI IMPUTASI (MURNI DARI X_TRAIN): GROUP MEDIAN VS SIMPLE GLOBAL MEDIAN ===")
display(imp_comp_df)

# ------------------------------------------------------------------------------
# 3. MEMBANGUN PIPELINE PRAPROSES TERPADU (IMPUTER + SCALER + ONE-HOT ENCODER)
# ------------------------------------------------------------------------------
col_transformer = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), num_features),
        ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), cat_features)
    ]
)

preprocessor = Pipeline([
    ('group_imputer', GroupMedianImputer(group_col='iter', target_cols=('core_num', 'threads_num'), min_samples=3)),
    ('feature_encoder', col_transformer)
])

# Fit HANYA pada X_train, kemudian transform secara konsisten pada X_train, X_val, dan X_test
X_train_scaled = preprocessor.fit_transform(X_train).astype(np.float32)
X_val_scaled   = preprocessor.transform(X_val).astype(np.float32)
X_test_scaled  = preprocessor.transform(X_test).astype(np.float32)

INPUT_DIM = X_train_scaled.shape[1]
NUM_CLASSES = len(label_names)

print(f"\nDimensi Tensor Fitur setelah GroupMedianImputer + StandardScaler + OneHotEncoder: {INPUT_DIM} fitur")
print(f"Shape X_train_scaled : {X_train_scaled.shape} (dtype: {X_train_scaled.dtype}) | NaN count: {np.isnan(X_train_scaled).sum()}")
print(f"Shape X_val_scaled   : {X_val_scaled.shape} (dtype: {X_val_scaled.dtype}) | NaN count: {np.isnan(X_val_scaled).sum()}")
print(f"Shape X_test_scaled  : {X_test_scaled.shape} (dtype: {X_test_scaled.dtype}) | NaN count: {np.isnan(X_test_scaled).sum()}")

# 4. Audit Overlap Identitas Model & Spesifikasi Antar-Subset (Train vs Val vs Test)
train_models_set = set(df.loc[X_train.index, 'model'])
val_models_set   = set(df.loc[X_val.index, 'model'])
test_models_set  = set(df.loc[X_test.index, 'model'])

overlap_audit_df = pd.DataFrame([
    {'Pemeriksaan Overlap Antar-Subset': 'Model Unik pada Data Latih (X_train - 712 baris)', 'Jumlah Entri': len(train_models_set), 'Keterangan Metodologis': 'Basis pembelajaran parameter model & statistik praproses'},
    {'Pemeriksaan Overlap Antar-Subset': 'Model Unik pada Data Validasi (X_val - 153 baris)', 'Jumlah Entri': len(val_models_set), 'Keterangan Metodologis': 'Subset seleksi arsitektur & kalibrasi threshold'},
    {'Pemeriksaan Overlap Antar-Subset': 'Model Unik pada Data Uji (X_test - 153 baris)', 'Jumlah Entri': len(test_models_set), 'Keterangan Metodologis': 'Subset evaluasi akhir satu kali (blind test evaluation)'},
    {'Pemeriksaan Overlap Antar-Subset': 'Irisan Nama Model Train n Val', 'Jumlah Entri': len(train_models_set.intersection(val_models_set)), 'Keterangan Metodologis': 'Berasal dari varian SKU katalog berulang pada dataset asli'},
    {'Pemeriksaan Overlap Antar-Subset': 'Irisan Nama Model Train n Test', 'Jumlah Entri': len(train_models_set.intersection(test_models_set)), 'Keterangan Metodologis': 'Didokumentasikan sebagai karakteristik alami dataset katalog e-commerce'}
])
print("\n=== AUDIT OVERLAP IDENTITAS MODEL ANTAR-SUBSET (TRAIN / VAL / TEST) ===")
display(overlap_audit_df)

# ================================================================================
# SECTION 04 — sumber: notebook sel 11
# ================================================================================
# ==============================================================================
# SECTION 4: PEMBANGUNAN INPUT PIPELINE tf.data & VALIDASI INTEGRITAS TENSOR
# ==============================================================================
BATCH_SIZE = 32

def create_tf_dataset(features, labels, batch_size=32, is_training=False):
    dataset = tf.data.Dataset.from_tensor_slices((features, labels))
    if is_training:
        # reshuffle_each_iteration=True memastikan urutan batch diacak ulang di setiap epoch
        dataset = dataset.shuffle(buffer_size=len(features), seed=SEED, reshuffle_each_iteration=True)
    dataset = dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return dataset

train_ds = create_tf_dataset(X_train_scaled, y_train, batch_size=BATCH_SIZE, is_training=True)
val_ds   = create_tf_dataset(X_val_scaled, y_val, batch_size=BATCH_SIZE, is_training=False)
test_ds  = create_tf_dataset(X_test_scaled, y_test, batch_size=BATCH_SIZE, is_training=False)

# Pemeriksaan kelayakan nilai tensor (Bebas NaN, Bebas Infinity, dan Rentang Label 0-2)
for name_arr, x_arr, y_arr in [('Train', X_train_scaled, y_train), ('Validation', X_val_scaled, y_val), ('Test', X_test_scaled, y_test)]:
    assert not np.isnan(x_arr).any(), f"Ditemukan NaN pada tensor {name_arr}!"
    assert not np.isinf(x_arr).any(), f"Ditemukan Infinity pada tensor {name_arr}!"
    assert set(np.unique(y_arr)) == {0, 1, 2}, f"Label di luar rentang 0-2 pada {name_arr}!"

tensor_check_df = pd.DataFrame([
    {'Pipeline Subset': 'train_ds (Latih)',    'Dimensi Tensor X': str(X_train_scaled.shape), 'Tipe X': str(X_train_scaled.dtype), 'Tipe y': str(y_train.dtype), 'Rentang Label': str( sorted(np.unique(y_train).tolist()) ), 'Jumlah NaN / Inf': '0 / 0', 'Ukuran Batch': BATCH_SIZE, 'Mode Shuffle': 'Aktif (Tiap Epoch)'},
    {'Pipeline Subset': 'val_ds (Validasi)',   'Dimensi Tensor X': str(X_val_scaled.shape),   'Tipe X': str(X_val_scaled.dtype),   'Tipe y': str(y_val.dtype),   'Rentang Label': str( sorted(np.unique(y_val).tolist()) ),   'Jumlah NaN / Inf': '0 / 0', 'Ukuran Batch': BATCH_SIZE, 'Mode Shuffle': 'Nonaktif (Deterministik)'},
    {'Pipeline Subset': 'test_ds (Uji Akhir)', 'Dimensi Tensor X': str(X_test_scaled.shape),  'Tipe X': str(X_test_scaled.dtype),  'Tipe y': str(y_test.dtype),  'Rentang Label': str( sorted(np.unique(y_test).tolist()) ),  'Jumlah NaN / Inf': '0 / 0', 'Ukuran Batch': BATCH_SIZE, 'Mode Shuffle': 'Nonaktif (Deterministik)'}
])
print("=== RINGKASAN VERIFIKASI KELAYAKAN TENSOR & KONFIGURASI PIPELINE tf.data ===")
display(tensor_check_df)

# Inspeksi spesifikasi 1 mini-batch dari train_ds
for batch_x, batch_y in train_ds.take(1):
    print(f"\nRepresentasi Tensor Batch X : shape={batch_x.shape}, dtype={batch_x.dtype}")
    print(f"Representasi Tensor Batch y : shape={batch_y.shape}, dtype={batch_y.dtype}")
    print(f"Contoh 5 Label Pertama Batch: {batch_y[:5].numpy()}")

# ================================================================================
# SECTION 05 — sumber: notebook sel 14
# ================================================================================
# ==============================================================================
# SECTION 5: PERANCANGAN & PELATIHAN MODEL 1 (BASELINE) DAN MODEL 2 (REGULARIZED MLP)
# ==============================================================================

# ------------------------------------------------------------------------------
# 1. Arsitektur Model 1: Baseline MLP (Tanpa BatchNorm & Dropout)
# ------------------------------------------------------------------------------
def build_baseline_mlp(input_dim, num_classes):
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(input_dim,), name='input_layer'),
        tf.keras.layers.Dense(64, activation='relu', name='dense_1'),
        tf.keras.layers.Dense(32, activation='relu', name='dense_2'),
        tf.keras.layers.Dense(num_classes, activation='softmax', name='output_layer')
    ], name='Model_1_Baseline_MLP')
    return model

# ------------------------------------------------------------------------------
# 2. Arsitektur Model 2: Regularized & BatchNorm MLP (128 -> 64 -> 32)
# ------------------------------------------------------------------------------
def build_regularized_mlp(input_dim, num_classes):
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(input_dim,), name='input_layer'),

        tf.keras.layers.Dense(128, kernel_regularizer=tf.keras.regularizers.l2(1e-4), name='dense_1'),
        tf.keras.layers.BatchNormalization(name='bn_1'),
        tf.keras.layers.Activation('relu', name='relu_1'),
        tf.keras.layers.Dropout(0.3, name='dropout_1'),

        tf.keras.layers.Dense(64, kernel_regularizer=tf.keras.regularizers.l2(1e-4), name='dense_2'),
        tf.keras.layers.BatchNormalization(name='bn_2'),
        tf.keras.layers.Activation('relu', name='relu_2'),
        tf.keras.layers.Dropout(0.2, name='dropout_2'),

        tf.keras.layers.Dense(32, kernel_regularizer=tf.keras.regularizers.l2(1e-4), name='dense_3'),
        tf.keras.layers.BatchNormalization(name='bn_3'),
        tf.keras.layers.Activation('relu', name='relu_3'),

        tf.keras.layers.Dense(num_classes, activation='softmax', name='output_layer')
    ], name='Model_2_Regularized_MLP')
    return model

# Menampilkan ringkasan arsitektur kedua model
baseline_mlp = build_baseline_mlp(INPUT_DIM, NUM_CLASSES)
regularized_mlp = build_regularized_mlp(INPUT_DIM, NUM_CLASSES)

print("=== RINGKASAN ARSITEKTUR MODEL 1: BASELINE MLP (64 -> 32) ===")
baseline_mlp.summary()
print("\n=== RINGKASAN ARSITEKTUR MODEL 2: REGULARIZED & BATCHNORM MLP (128 -> 64 -> 32) ===")
regularized_mlp.summary()

# Pelatihan Model 1: Baseline MLP (SGD standar lr=0.01, 50 epoch)
tf.keras.utils.set_random_seed(SEED)
baseline_mlp.compile(
    optimizer=tf.keras.optimizers.SGD(learning_rate=0.01),
    loss=tf.keras.losses.SparseCategoricalCrossentropy(),
    metrics=['accuracy']
)
print("\n[Pelatihan 1/2] Melatih Model 1 (Baseline MLP - SGD)...")
hist_base = baseline_mlp.fit(train_ds, validation_data=val_ds, epochs=50, verbose=0)
print(f"-> Selesai (50 Epoch)! Peak Val Accuracy Model 1 (Baseline MLP) : {max(hist_base.history['val_accuracy'])*100:.2f}%")
print(f"-> Val Loss Terendah Model 1 (Baseline MLP)                     : {min(hist_base.history['val_loss']):.4f}")

# Pelatihan Model 2: Regularized MLP (Adam + BN + Dropout + L2 + Callbacks)
tf.keras.utils.set_random_seed(SEED)
regularized_mlp.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    loss=tf.keras.losses.SparseCategoricalCrossentropy(),
    metrics=['accuracy']
)
callbacks_reg = [
    tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-5, verbose=0),
    tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True, verbose=0)
]
print("\n[Pelatihan 2/2] Melatih Model 2 (Regularized MLP - Adam + BN + Dropout + L2)...")
hist_reg = regularized_mlp.fit(train_ds, validation_data=val_ds, epochs=100, callbacks=callbacks_reg, verbose=0)

best_epoch_reg = int(np.argmin(hist_reg.history['val_loss'])) + 1
restored_val_acc_reg = hist_reg.history['val_accuracy'][best_epoch_reg - 1] * 100
print(f"-> Selesai pada Epoch ke-{len(hist_reg.history['loss'])} (EarlyStopping)!")
print(f"-> Peak Val Accuracy selama pelatihan Model 2                   : {max(hist_reg.history['val_accuracy'])*100:.2f}% (Epoch ke-{int(np.argmax(hist_reg.history['val_accuracy']))+1})")
print(f"-> Bobot terbaik dipulihkan (restore_best_weights) dari Epoch ke-{best_epoch_reg}:")
print(f"   * Val Loss Terendah Model 2 (Epoch ke-{best_epoch_reg})                   : {min(hist_reg.history['val_loss']):.4f}")
print(f"   * Val Accuracy pada Bobot Dipulihkan (Epoch ke-{best_epoch_reg})          : {restored_val_acc_reg:.2f}%")

# Visualisasi Perbandingan Langsung Kurva Pembelajaran Model 1 vs Model 2
fig, axes = plt.subplots(1, 2, figsize=(15.5, 5.2))
ep1 = np.arange(1, len(hist_base.history['accuracy']) + 1)
ep2 = np.arange(1, len(hist_reg.history['accuracy']) + 1)

axes[0].plot(ep1, np.array(hist_base.history['accuracy'])*100, label='Model 1 Train Acc (SGD, No Reg)', color='#d73027', linestyle=':', linewidth=1.8)
axes[0].plot(ep1, np.array(hist_base.history['val_accuracy'])*100, label=f"Model 1 Val Acc (Peak: {max(hist_base.history['val_accuracy'])*100:.2f}%)", color='#d73027', linewidth=2.0)
axes[0].plot(ep2, np.array(hist_reg.history['accuracy'])*100, label='Model 2 Train Acc (Adam + BN + Dropout + L2)', color='#1b7837', linestyle='--', linewidth=1.8)
axes[0].plot(ep2, np.array(hist_reg.history['val_accuracy'])*100, label=f"Model 2 Val Acc (Peak: {max(hist_reg.history['val_accuracy'])*100:.2f}% | Ep {best_epoch_reg}: {restored_val_acc_reg:.2f}%)", color='#1b7837', linewidth=2.3)
axes[0].axvline(best_epoch_reg, color='#2166ac', linestyle='-.', alpha=0.85, label=f'Restored Best Weights (Epoch {best_epoch_reg})')
axes[0].set_title('[A] Perbandingan Kurva Akurasi: Model 1 (Baseline) vs Model 2 (Regularized)', fontweight='bold')
axes[0].set_xlabel('Epoch Pelatihan')
axes[0].set_ylabel('Accuracy (%)')
axes[0].legend(loc='lower right', fontsize=8.5)
axes[0].grid(True, alpha=0.3)

axes[1].plot(ep1, hist_base.history['loss'], label='Model 1 Train Loss (SGD)', color='#d73027', linestyle=':', linewidth=1.8)
axes[1].plot(ep1, hist_base.history['val_loss'], label=f"Model 1 Val Loss (Min: {min(hist_base.history['val_loss']):.4f})", color='#d73027', linewidth=2.0)
axes[1].plot(ep2, hist_reg.history['loss'], label='Model 2 Train Loss (Adam + Reg)', color='#2166ac', linestyle='--', linewidth=1.8)
axes[1].plot(ep2, hist_reg.history['val_loss'], label=f"Model 2 Val Loss (Min: {min(hist_reg.history['val_loss']):.4f} @ Epoch {best_epoch_reg})", color='#2166ac', linewidth=2.3)
axes[1].axvline(best_epoch_reg, color='#1b7837', linestyle='-.', alpha=0.85, label=f'Titik Val Loss Terendah (Epoch {best_epoch_reg})')
axes[1].set_title('[B] Perbandingan Kurva Loss & EarlyStopping: Model 1 vs Model 2', fontweight='bold')
axes[1].set_xlabel('Epoch Pelatihan')
axes[1].set_ylabel('Sparse Categorical Crossentropy Loss')
axes[1].legend(loc='upper right', fontsize=8.5)
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# ================================================================================
# SECTION 06 — sumber: notebook sel 17
# ================================================================================
# ==============================================================================
# SECTION 6: HYPERPARAMETER TUNING MLP DENGAN KERASTUNER (RANDOMSEARCH)
# ==============================================================================
def build_tunable_mlp(hp):
    model = tf.keras.Sequential(name='Model_3_Tuned_MLP')
    model.add(tf.keras.layers.Input(shape=(INPUT_DIM,), name='input_layer'))

    # Ruang pencarian jumlah hidden layer (2 atau 3 layer) dan kekuatan L2 (skala log)
    n_layers = hp.Int('n_layers', min_value=2, max_value=3, default=3)
    l2_reg = hp.Float('l2_reg', min_value=1e-5, max_value=1e-3, sampling='log', default=1e-4)
    dropout_1 = hp.Float('dropout_1', min_value=0.2, max_value=0.4, step=0.1, default=0.3)
    dropout_2 = hp.Float('dropout_2', min_value=0.1, max_value=0.3, step=0.1, default=0.2)

    # Layer 1
    units_1 = hp.Choice('units_1', values=[96, 128, 192, 256], default=192)
    model.add(tf.keras.layers.Dense(units_1, kernel_regularizer=tf.keras.regularizers.l2(l2_reg), name='dense_1'))
    model.add(tf.keras.layers.BatchNormalization(name='bn_1'))
    model.add(tf.keras.layers.Activation('relu', name='relu_1'))
    model.add(tf.keras.layers.Dropout(dropout_1, name='dropout_1'))

    # Layer 2
    units_2 = hp.Choice('units_2', values=[48, 64, 96, 128], default=96)
    model.add(tf.keras.layers.Dense(units_2, kernel_regularizer=tf.keras.regularizers.l2(l2_reg), name='dense_2'))
    model.add(tf.keras.layers.BatchNormalization(name='bn_2'))
    model.add(tf.keras.layers.Activation('relu', name='relu_2'))
    model.add(tf.keras.layers.Dropout(dropout_2, name='dropout_2'))

    # Layer 3 (Opsional sesuai hasil tuning n_layers)
    units_3 = hp.Choice('units_3', values=[24, 32, 48, 64], default=32)
    if n_layers == 3:
        model.add(tf.keras.layers.Dense(units_3, kernel_regularizer=tf.keras.regularizers.l2(l2_reg), name='dense_3'))
        model.add(tf.keras.layers.BatchNormalization(name='bn_3'))
        model.add(tf.keras.layers.Activation('relu', name='relu_3'))

    # Output Layer (3 Kelas Harga)
    model.add(tf.keras.layers.Dense(NUM_CLASSES, activation='softmax', name='output_layer'))

    # Ruang pencarian Optimizer & Learning Rate pada skala logaritmik
    opt_name = hp.Choice('optimizer', values=['adam', 'rmsprop'], default='adam')
    lr = hp.Float('learning_rate', min_value=1e-4, max_value=5e-3, sampling='log', default=1e-3)

    if opt_name == 'adam':
        optimizer = tf.keras.optimizers.Adam(learning_rate=lr)
    else:
        optimizer = tf.keras.optimizers.RMSprop(learning_rate=lr)

    model.compile(
        optimizer=optimizer,
        loss=tf.keras.losses.SparseCategoricalCrossentropy(),
        metrics=['accuracy']
    )
    return model

# Inisialisasi KerasTuner RandomSearch (10 Percobaan pada Data Validasi, 2 Eksekusi/Trial)
tuner = kt.RandomSearch(
    hypermodel=build_tunable_mlp,
    objective='val_accuracy',
    max_trials=10,
    executions_per_trial=2,
    seed=SEED,
    directory='kt_laptop_mlp_dir',
    project_name='laptop_price_mlp_tuning',
    overwrite=True
)

search_early_stop = tf.keras.callbacks.EarlyStopping(
    monitor='val_loss', patience=8, restore_best_weights=True
)

print("Memulai pencarian Hyperparameter dengan KerasTuner RandomSearch (10 Trials, 2 Executions/Trial)...")
tuner.search(
    train_ds,
    validation_data=val_ds,
    epochs=45,
    callbacks=[search_early_stop],
    verbose=0
)

best_hp = tuner.get_best_hyperparameters(num_trials=1)[0]
print("\n=== KONFIGURASI HYPERPARAMETER TERBAIK HASIL KERASTUNER ===")
print(f"- Jumlah Hidden Layer (n_layers) : {best_hp.get('n_layers')} layer")
print(f"- Jumlah Neuron Tiap Layer       : {best_hp.get('units_1')} -> {best_hp.get('units_2')}" +
      (f" -> {best_hp.get('units_3')}" if best_hp.get('n_layers') == 3 else ""))
print(f"- Dropout Rate Bertingkat        : L1={best_hp.get('dropout_1')} | L2={best_hp.get('dropout_2')}")
print(f"- Kekuatan Regularisasi L2       : {best_hp.get('l2_reg'):.6f} (Skala Log)")
print(f"- Optimizer Terpilih             : {best_hp.get('optimizer').upper()}")
print(f"- Learning Rate Awal             : {best_hp.get('learning_rate'):.5f} (Skala Log)")

# Menyusun ringkasan 10 trial untuk analisis dan visualisasi
trials_records = []
for idx, trial in enumerate(tuner.oracle.get_best_trials(num_trials=10)):
    hp_vals = trial.hyperparameters.values
    trials_records.append({
        'Peringkat': idx + 1,
        'Trial ID': f"Trial #{trial.trial_id}",
        'n_layers': hp_vals.get('n_layers'),
        'Susunan Unit': f"{hp_vals.get('units_1')}-{hp_vals.get('units_2')}" + (f"-{hp_vals.get('units_3')}" if hp_vals.get('n_layers') == 3 else ""),
        'Dropout (L1/L2)': f"{hp_vals.get('dropout_1')}/{hp_vals.get('dropout_2')}",
        'L2 Reg': round(hp_vals.get('l2_reg'), 6),
        'Optimizer': hp_vals.get('optimizer').upper(),
        'Learning Rate': round(hp_vals.get('learning_rate'), 5),
        'Val Accuracy (%)': round(trial.score * 100, 2)
    })
trials_summary_df = pd.DataFrame(trials_records)
print("\n=== TABEL PERINGKAT 10 PERCOBAAN HYPERPARAMETER TUNING (DATA VALIDASI) ===")
display(trials_summary_df)

# Visualisasi hasil eksplorasi RandomSearch
fig, axes = plt.subplots(1, 2, figsize=(15, 4.6))
colors_bar = ['#1b7837' if i == 0 else '#4393c3' for i in range(len(trials_summary_df))]
axes[0].barh(trials_summary_df['Trial ID'][::-1], trials_summary_df['Val Accuracy (%)'][::-1],
             color=colors_bar[::-1], edgecolor='black', linewidth=0.6)
axes[0].set_xlim(0, 100)  # skala penuh agar hasil trial tidak terpotong
axes[0].set_xlabel('Validation Accuracy (%) [Rata-rata 2 Eksekusi/Trial]')
axes[0].set_title('[A] Peringkat 10 Percobaan KerasTuner (RandomSearch)', fontweight='bold')
axes[0].grid(axis='x', alpha=0.3)

for opt, grp in trials_summary_df.groupby('Optimizer'):
    axes[1].scatter(grp['Learning Rate'], grp['Val Accuracy (%)'], s=110, label=f"Optimizer: {opt}", alpha=0.9, edgecolors='black')
axes[1].set_xscale('log')
axes[1].set_xlabel('Learning Rate (Skala Logaritmik)')
axes[1].set_ylabel('Validation Accuracy (%)')
axes[1].set_title('[B] Pengaruh Learning Rate (Skala Log) vs Val Accuracy', fontweight='bold')
axes[1].legend(loc='lower left')
axes[1].grid(True, which='both', alpha=0.3)
plt.tight_layout()
plt.show()

# ================================================================================
# SECTION 07 — sumber: notebook sel 20
# ================================================================================
# ==============================================================================
# SECTION 7: LATIH ULANG (RETRAINING) KONFIGURASI TERBAIK & CALLBACKS LENGKAP
# ==============================================================================
tf.keras.utils.set_random_seed(SEED)

# Konfigurasi tidak ditulis manual; ditentukan oleh best_hp hasil KerasTuner.
best_trial_id = tuner.oracle.get_best_trials(num_trials=1)[0].trial_id
tuned_mlp = tuner.hypermodel.build(best_hp)
tuned_units = [best_hp.get('units_1'), best_hp.get('units_2')]
if best_hp.get('n_layers') == 3:
    tuned_units.append(best_hp.get('units_3'))
tuned_structure = ' -> '.join(str(units) for units in tuned_units)
print(f"=== MODEL UTAMA SESUAI JUDUL: MLP HASIL HYPERPARAMETER TUNING ===")
print(f"Trial terbaik: {best_trial_id} | Hidden layer: {best_hp.get('n_layers')} | Unit: {tuned_structure}")
tuned_mlp.summary()

# Menyusun Callbacks lengkap: patience ReduceLROnPlateau (5) < patience EarlyStopping (15)
callbacks_tuned = [
    tf.keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss', factor=0.5, patience=5, min_delta=1e-4, min_lr=1e-5, verbose=1
    ),
    tf.keras.callbacks.EarlyStopping(
        monitor='val_loss', patience=15, min_delta=1e-4, restore_best_weights=True, verbose=1
    ),
    tf.keras.callbacks.ModelCheckpoint(
        filepath='Model_Tuned_MLP_Terbaik.keras', monitor='val_loss', save_best_only=True, verbose=0
    ),
    tf.keras.callbacks.CSVLogger('log_training_tuned_mlp.csv', append=False)
]

print("\nMelatih ulang (Retraining) Model 3 (Tuned MLP) dengan Callbacks lengkap...")
hist_tuned = tuned_mlp.fit(
    train_ds,
    validation_data=val_ds,
    epochs=100,
    callbacks=callbacks_tuned,
    verbose=0
)

best_epoch = int(np.argmin(hist_tuned.history['val_loss'])) + 1
restored_val_acc_tuned = hist_tuned.history['val_accuracy'][best_epoch - 1] * 100
print(f"\n-> Training berhenti pada Epoch ke-{len(hist_tuned.history['loss'])} (EarlyStopping)!")
print(f"-> Peak Val Accuracy selama Retraining Tuned MLP              : {max(hist_tuned.history['val_accuracy'])*100:.2f}% (Epoch ke-{int(np.argmax(hist_tuned.history['val_accuracy']))+1})")
print(f"-> Bobot terbaik dipulihkan dari Epoch ke-{best_epoch}:")
print(f"   * Val Loss Terendah Tuned MLP (Epoch ke-{best_epoch})               : {min(hist_tuned.history['val_loss']):.4f}")
print(f"   * Val Accuracy pada Bobot Dipulihkan (Epoch ke-{best_epoch})        : {restored_val_acc_tuned:.2f}%")
print("   * File checkpoint tersimpan                               : Model_Tuned_MLP_Terbaik.keras")
print("   * Catatan: skor latih ulang dapat berbeda dari skor rata-rata trial tuning.")
print("   * File log pemantauan tersimpan                           : log_training_tuned_mlp.csv")

# ================================================================================
# SECTION 08 — sumber: notebook sel 23
# ================================================================================
# ============================================================================
# SECTION 8 - EVALUASI MLP HASIL TUNING (MODEL UTAMA) DAN PEMBANDING
# ============================================================================
# Semua metrik di sini dihitung dari model yang tersedia dalam memori.
# Model utama sudah ditetapkan: MLP yang hyperparameternya dipilih KerasTuner.
# Model 1 dan 2 hanya pembanding. Data test TIDAK dipakai mengubah model utama.
from sklearn.metrics import f1_score

model_candidates = [
    ('Model 1: Baseline MLP', baseline_mlp, hist_base),
    ('Model 2: Regularized MLP', regularized_mlp, hist_reg),
    ('Model 3: Tuned MLP', tuned_mlp, hist_tuned),
]

validation_rows = []
for name, model, history in model_candidates:
    val_prob = model.predict(X_val_scaled, verbose=0)
    val_pred = val_prob.argmax(axis=1)
    val_metrics = model.evaluate(X_val_scaled, y_val, verbose=0, return_dict=True)
    validation_rows.append({
        'Model': name,
        'Jumlah Parameter': int(model.count_params()),
        'Peak Val Accuracy (%)': float(max(history.history['val_accuracy']) * 100),
        'Val Accuracy Bobot Akhir (%)': float(accuracy_score(y_val, val_pred) * 100),
        'Val Macro F1': float(f1_score(y_val, val_pred, average='macro', zero_division=0)),
        'Val Loss Bobot Akhir': float(val_metrics['loss']),
        'Min Val Loss Selama Training': float(min(history.history['val_loss'])),
        'Epoch Dilatih': int(len(history.history['loss'])),
    })

val_selection_df = pd.DataFrame(validation_rows)
# Peringkat F1 validasi untuk membandingkan kandidat; bukan dasar mengganti
# model utama. Judul penelitian menetapkan fokus pada MLP hasil tuning KerasTuner.
val_ranked = val_selection_df.sort_values(
    ['Val Macro F1', 'Val Accuracy Bobot Akhir (%)', 'Val Loss Bobot Akhir'],
    ascending=[False, False, True], kind='stable'
).reset_index(drop=True)
final_model_name = 'Model 3: Tuned MLP'
final_model = tuned_mlp
assert any(name == final_model_name and model is final_model for name, model, _ in model_candidates)
final_selection_basis = (
    'Model utama MLP hasil hyperparameter tuning KerasTuner; '
    'best_hp dipilih memakai val_accuracy selama pencarian, bukan hasil test.'
)
print('=== PERBANDINGAN VALIDASI TIGA MODEL (BUKAN PEMILIHAN DARI TEST) ===')
display(val_ranked.round(4))
print('Model utama sesuai judul:', final_model_name)
print('Dasar penetapan:', final_selection_basis)
print('Catatan: jika model pembanding lebih tinggi, selisih akan dibahas tanpa mengganti model utama.')

# Perbandingan tiga kandidat pada test bersifat DESKRIPTIF; hasil test tidak
# mengubah MLP hasil tuning yang sudah ditetapkan sebagai model utama.
results_rows = []
predictions_dict, test_probabilities_dict = {}, {}
for name, model, history in model_candidates:
    probs = model.predict(X_test_scaled, verbose=0)
    pred = probs.argmax(axis=1)
    predictions_dict[name] = pred
    test_probabilities_dict[name] = probs
    pr, rc, f1, _ = precision_recall_fscore_support(
        y_test, pred, average='macro', zero_division=0
    )
    results_rows.append({
        'Iterasi Model': name,
        'Test Accuracy (%)': float(accuracy_score(y_test, pred) * 100),
        'Test Macro Precision': float(pr),
        'Test Macro Recall': float(rc),
        'Test Macro F1-Score': float(f1),
        'Jumlah Salah Uji': int(np.sum(y_test != pred)),
        'Jumlah Data Uji': int(len(y_test)),
    })
results_df = pd.DataFrame(results_rows)
print('\n=== EVALUASI DESKRIPTIF DATA TEST - TUNED MLP TETAP MODEL UTAMA ===')
display(results_df.round(4))

probs_test_final = test_probabilities_dict[final_model_name]
y_pred_final = predictions_dict[final_model_name]
print('\n=== LAPORAN KLASIFIKASI MODEL UTAMA: MLP HASIL TUNING ===')
print(classification_report(y_test, y_pred_final, target_names=label_names, zero_division=0))

# Kurva model final serta confusion matrix; judul tidak menggunakan nilai manual.
selected_history = next(h for name, _, h in model_candidates if name == final_model_name)
fig, axes = plt.subplots(1, 3, figsize=(16, 4.7))
axes[0].plot(selected_history.history['accuracy'], label='Latih')
axes[0].plot(selected_history.history['val_accuracy'], label='Validasi')
axes[0].set_title('Akurasi model final')
axes[0].set_xlabel('Epoch'); axes[0].set_ylabel('Accuracy'); axes[0].legend(); axes[0].grid(alpha=0.3)
axes[1].plot(selected_history.history['loss'], label='Latih')
axes[1].plot(selected_history.history['val_loss'], label='Validasi')
axes[1].set_title('Loss model final')
axes[1].set_xlabel('Epoch'); axes[1].set_ylabel('Loss'); axes[1].legend(); axes[1].grid(alpha=0.3)
cm_final = confusion_matrix(y_test, y_pred_final, labels=[0, 1, 2])
sns.heatmap(cm_final, annot=True, fmt='d', cmap='Blues', ax=axes[2],
            xticklabels=['Entry', 'Mid', 'High'], yticklabels=['Entry', 'Mid', 'High'])
axes[2].set_title(f'Confusion matrix: {final_model_name}')
axes[2].set_xlabel('Prediksi'); axes[2].set_ylabel('Aktual')
plt.tight_layout(); plt.show()

# Contoh salah dan benar HARUS berasal dari baris pengujian yang benar-benar ada.
real_test_rows = df.loc[X_test.index, ['model', 'price', 'price_idr', 'category']].reset_index(drop=True).copy()
real_test_rows['Harga Aktual (INR / IDR)'] = real_test_rows['price'].apply(format_dual_harga)
real_test_rows['Kelas Aktual'] = y_test
real_test_rows['Kelas Prediksi'] = y_pred_final
real_test_rows['Confidence (%)'] = np.max(probs_test_final, axis=1) * 100
real_test_rows['Benar?'] = real_test_rows['Kelas Aktual'] == real_test_rows['Kelas Prediksi']
real_test_rows['Jarak Harga ke Batas Terdekat (INR)'] = real_test_rows['price'].apply(
    lambda p: min(abs(float(p)-50000), abs(float(p)-85000))
)
real_test_rows['Jarak ke Ambang (INR / IDR)'] = real_test_rows[
    'Jarak Harga ke Batas Terdekat (INR)'].apply(format_dual_harga)
# Beberapa contoh kelas benar, lalu kesalahan paling dekat ambang harga.
benar = real_test_rows[real_test_rows['Benar?']].groupby('Kelas Aktual', group_keys=False).head(1)
salah = real_test_rows[~real_test_rows['Benar?']].sort_values(
    'Jarak Harga ke Batas Terdekat (INR)'
).head(8)
error_analysis_df = pd.concat([benar, salah], ignore_index=True)
print('\n=== CONTOH PREDIKSI BENAR DAN SALAH, BUKAN DATA DIKETIK MANUAL ===')
display(error_analysis_df.round(2))

# Simulasi spesifikasi laptop baru dipertahankan dari eksperimen sebelumnya (tanpa label aktual).
# 'sampel_laptop_baru' adalah input simulasi; BUKAN contoh benar/salah pada test.
sampel_laptop_baru = pd.DataFrame([
    {'nama_simulasi':'Laptop kuliah (Core i3, 8GB, Integrated)', 'brand_name':'Asus', 'processor_brand':'Intel', 'iter':'Core i3',
     'rating':4.2, 'threads_num':8, 'core_num':6, 'ram_num':8, 'ram_type':'DDR4', 'memory_type':'SSD',
     'memory_size':512, 'gpu_brand':'Intel', 'gpu_type':'Integrated', 'os':'Windows 10/11',
     'display_size':14.0, 'resolution_height':1920, 'resolution_width':1080, 'Touch_Screen':'False', 'warranty':1},
    {'nama_simulasi':'Laptop kreator (Ryzen 5, 16GB, Dedicated)', 'brand_name':'Lenovo', 'processor_brand':'AMD', 'iter':'Ryzen 5',
     'rating':4.4, 'threads_num':12, 'core_num':6, 'ram_num':16, 'ram_type':'DDR5', 'memory_type':'SSD',
     'memory_size':512, 'gpu_brand':'NVIDIA', 'gpu_type':'Dedicated', 'os':'Windows 10/11',
     'display_size':15.6, 'resolution_height':1920, 'resolution_width':1080, 'Touch_Screen':'False', 'warranty':1},
    {'nama_simulasi':'Laptop gaming (Core i9, 32GB, Dedicated)', 'brand_name':'Asus', 'processor_brand':'Intel', 'iter':'Core i9',
     'rating':4.6, 'threads_num':24, 'core_num':16, 'ram_num':32, 'ram_type':'DDR5', 'memory_type':'SSD',
     'memory_size':1000, 'gpu_brand':'NVIDIA', 'gpu_type':'Dedicated', 'os':'Windows 10/11',
     'display_size':16.0, 'resolution_height':2560, 'resolution_width':1600, 'Touch_Screen':'False', 'warranty':2},
])
sampel_laptop_baru['ppi'] = np.hypot(
    sampel_laptop_baru['resolution_height'], sampel_laptop_baru['resolution_width']
) / sampel_laptop_baru['display_size']
sim_probs = final_model.predict(
    preprocessor.transform(sampel_laptop_baru[num_features + cat_features]).astype(np.float32), verbose=0
)
sim_pred = sim_probs.argmax(axis=1)
simulasi_df = pd.DataFrame({
    'Deskripsi': sampel_laptop_baru['nama_simulasi'],
    'Kelas': [label_names[i] for i in sim_pred],
    'Confidence (%)': np.max(sim_probs, axis=1) * 100,
    'Rentang Kelas Prediksi (INR / IDR)': [rentang_kelas_dual(i) for i in sim_pred],
})
print('\n=== SIMULASI SPESIFIKASI BARU (TANPA HARGA AKTUAL / GROUND TRUTH) ===')
print('Model menghasilkan kelas harga (INR / IDR ilustratif), bukan nominal harga spesifik laptop.')
display(simulasi_df.round(2))


# ================================================================================
# SECTION 09 — sumber: notebook sel 26
# ================================================================================
# ==============================================================================
# SECTION 9: AUDIT & VISUALISASI KONDISI AWAL (PEMISAHAN 3 LAPISAN BASELINE)
# ==============================================================================

# 1. Mengambil metrik kondisi awal dari Model 1 (Baseline MLP: 64 -> 32, SGD lr=0.01)
y_pred_base = predictions_dict['Model 1: Baseline MLP']
acc_base = accuracy_score(y_test, y_pred_base) * 100
prec_base, rec_base, f1_base, _ = precision_recall_fscore_support(y_test, y_pred_base, average='macro')
prec_cls_base, rec_cls_base, f1_cls_base, _ = precision_recall_fscore_support(y_test, y_pred_base, average=None)

# Menyusun Tabel Pemisahan 3 Kondisi Awal (Data Mentah vs Skala Fitur vs Arsitektur Baseline MLP)
before_summary_df = pd.DataFrame([
    {
        'Lapisan Kondisi Awal': '1. Kualitas Dataset Mentah (Sebelum Pembersihan)',
        'Status / Kondisi Awal': f"{df_cleaned.isnull().sum().sum()} nilai hilang (threads_num: {int(df_cleaned['threads_num'].isna().sum())}, core_num: {int(df_cleaned['core_num'].isna().sum())}); nama kolom dan kategori diperiksa",
        'Dampak Teknis & Penanganan': 'Diselesaikan secara deterministik & melalui GroupMedianImputer bebas leakage pada X_train'
    },
    {
        'Lapisan Kondisi Awal': '2. Skala Fitur Numerik (Sebelum StandardScaler)',
        'Status / Kondisi Awal': f"Rentang timpang: rating ({df['rating'].min()}-{df['rating'].max()}) vs memory_size ({df['memory_size'].min()}-{df['memory_size'].max()} GB)",
        'Dampak Teknis & Penanganan': 'Diselesaikan untuk SEMUA model melalui StandardScaler (Mean ~ 0, Std ~ 1) yang di-fit pada X_train'
    },
    {
        'Lapisan Kondisi Awal': '3. Arsitektur & Optimasi Baseline (Model 1 pada Data Terpraproses)',
        'Status / Kondisi Awal': 'MLP Dasar (64 -> 32), Optimizer SGD (lr=0.01) Konstan, Tanpa BatchNorm / Dropout / L2',
        'Dampak Teknis & Penanganan': f"Meski memakai tensor {INPUT_DIM} fitur yang sudah bersih, konvergensi lambat (Min Val Loss: {min(hist_base.history['val_loss']):.4f})"
    },
    {
        'Lapisan Kondisi Awal': '4. Performa Evaluasi Baseline MLP (Data Uji / Test Set)',
        'Status / Kondisi Awal': f"Test Accuracy = {acc_base:.2f}% | Macro Precision = {prec_base:.4f} | Macro F1 = {f1_base:.4f}",
        'Dampak Teknis & Penanganan': f"Terdapat {int((y_test != y_pred_base).sum())} sampel laptop salah klasifikasi dari {len(y_test)} data uji"
    }
])

print("=== TABEL PEMISAHAN 3 KONDISI AWAL: DATA MENTAH, SKALA FITUR & PERFORMA ARSITEKTUR BASELINE MLP ===")
display(before_summary_df)

# 2. Mencetak Dashboard Grafik Kondisi Awal (4 Panel Visualisasi)
fig, axes = plt.subplots(2, 2, figsize=(16, 11))
fig.suptitle('AUDIT KONDISI AWAL: KUALITAS DATA MENTAH, SKALA FITUR SEBELUM SCALING & PERFORMA ARSITEKTUR BASELINE MLP',
             fontsize=12.5, fontweight='bold', y=1.02)

# Panel A: Profil Masalah Data Mentah Sebelum Pembersihan
masalah_labels = ['Missing:\nthreads_num', 'Missing:\ncore_num', 'Label Kotor:\ngpu_brand', 'Label Kotor:\nos', 'Typo Nama\nKolom']
masalah_counts = [
    int(df_cleaned['threads_num'].isnull().sum()),
    int(df_cleaned['core_num'].isnull().sum()),
    int(df_cleaned['gpu_brand'].nunique()),
    int(df_cleaned['os'].nunique()),
    3
]
bars_a = axes[0, 0].bar(masalah_labels, masalah_counts,
                        color=['#d73027', '#f46d43', '#fdae61', '#fee08b', '#abd9e9'], edgecolor='black')
axes[0, 0].set_title('[A] Kondisi 1: Profil Masalah Dataset Mentah Sebelum Pembersihan', fontweight='bold')
axes[0, 0].set_ylabel('Jumlah Temuan Masalah / Variasi Label')
axes[0, 0].set_ylim(0, max(masalah_counts) * 1.18)
axes[0, 0].grid(axis='y', alpha=0.3)

# Panel B: Ketimpangan Skala Fitur Numerik Mentah Sebelum StandardScaler
raw_means = X_train[num_features].mean()
raw_stds = X_train[num_features].std()
x_pos = np.arange(len(num_features))
axes[0, 1].bar(x_pos - 0.18, raw_means, width=0.35, label='Mean Mentah (Sebelum StandardScaler)', color='#e08214', edgecolor='black')
axes[0, 1].bar(x_pos + 0.18, raw_stds, width=0.35, label='Std Dev Mentah (Sebelum StandardScaler)', color='#8073ac', edgecolor='black')
axes[0, 1].set_yscale('log')
axes[0, 1].set_xticks(x_pos)
axes[0, 1].set_xticklabels(num_features, rotation=30, ha='right')
axes[0, 1].set_title('[B] Kondisi 2: Ketimpangan Skala Numerik Sebelum StandardScaler (Log Scale)', fontweight='bold')
axes[0, 1].set_ylabel('Besaran Nilai Asli (Skala Log10)')
axes[0, 1].legend(loc='upper left')
axes[0, 1].grid(True, which='both', alpha=0.3)

# Panel C: Learning Curve Baseline MLP (Pada Data yang Sudah Terpraproses)
epochs_base = np.arange(1, len(hist_base.history['accuracy']) + 1)
axes[1, 0].plot(epochs_base, np.array(hist_base.history['accuracy']) * 100,
                label='Train Accuracy (Model 1 Baseline SGD)', color='#b2182b', linewidth=2)
axes[1, 0].plot(epochs_base, np.array(hist_base.history['val_accuracy']) * 100,
                label=f'Val Accuracy (Model 1 Baseline SGD - Peak: {max(hist_base.history["val_accuracy"])*100:.2f}%)', color='#ef8a62', linewidth=2, linestyle='--')
axes[1, 0].plot(epochs_base, np.array(hist_base.history['loss']) * 100,
                label='Train Loss (x100, hanya untuk tampilan)', color='#4d4d4d', linewidth=1.5, linestyle=':')
axes[1, 0].set_title('[C] Kondisi 3: Kurva Pembelajaran Arsitektur Baseline MLP (Pada Data Terpraproses)', fontweight='bold')
axes[1, 0].set_xlabel('Epoch (Tanpa EarlyStopping & Tanpa LR Scheduler)')
axes[1, 0].set_ylabel('Persentase (%) / Skala Metrik')
axes[1, 0].legend(loc='lower right')
axes[1, 0].grid(True, alpha=0.3)

# Panel D: Confusion Matrix Awal pada Model 1 (Baseline MLP)
cm_base = confusion_matrix(y_test, y_pred_base)
sns.heatmap(cm_base, annot=True, fmt='d', cmap='OrRd',
            xticklabels=['Entry-Level', 'Mid-Range', 'High-End'],
            yticklabels=['Entry-Level', 'Mid-Range', 'High-End'],
            annot_kws={'size': 12, 'weight': 'bold'}, ax=axes[1, 1])
axes[1, 1].set_title(f"[D] Confusion Matrix Arsitektur Baseline MLP (Test Acc: {acc_base:.2f}%)", fontweight='bold')
axes[1, 1].set_xlabel('Prediksi Model 1 (Baseline MLP)')
axes[1, 1].set_ylabel('Kelas Aktual (Ground Truth)')

plt.tight_layout()
plt.show()

# ================================================================================
# SECTION 10 — sumber: notebook sel 29
# ================================================================================
# ============================================================================
# SECTION 10 - RINGKASAN PERBANDINGAN DARI HASIL EKSEKUSI
# ============================================================================
# Ambil dari val_selection_df dan results_df yang baru saja dihitung.
all_candidates_summary_df = val_selection_df.merge(
    results_df, left_on='Model', right_on='Iterasi Model', how='inner'
).drop(columns=['Iterasi Model'])
print('=== PERBANDINGAN METRIK MODEL YANG DIHITUNG, TANPA NILAI MANUAL ===')
display(all_candidates_summary_df.round(4))

baseline_row = all_candidates_summary_df.loc[
    all_candidates_summary_df['Model'] == 'Model 1: Baseline MLP'
].iloc[0]
final_row = all_candidates_summary_df.loc[
    all_candidates_summary_df['Model'] == final_model_name
].iloc[0]
progress_comparison_df = pd.DataFrame([
    {
        'Indikator': 'Akurasi validasi bobot akhir (%)',
        'Baseline': baseline_row['Val Accuracy Bobot Akhir (%)'],
        'Model hasil tuning': final_row['Val Accuracy Bobot Akhir (%)']
    },
    {
        'Indikator': 'F1 makro validasi',
        'Baseline': baseline_row['Val Macro F1'],
        'Model hasil tuning': final_row['Val Macro F1']
    },
    {
        'Indikator': 'Akurasi data uji (%)',
        'Baseline': baseline_row['Test Accuracy (%)'],
        'Model hasil tuning': final_row['Test Accuracy (%)']
    },
    {
        'Indikator': 'F1 makro data uji',
        'Baseline': baseline_row['Test Macro F1-Score'],
        'Model hasil tuning': final_row['Test Macro F1-Score']
    },
])
progress_comparison_df['Selisih (tuned - baseline)'] = (
    progress_comparison_df['Model hasil tuning'] - progress_comparison_df['Baseline']
)
print('\n=== PERBANDINGAN BASELINE DENGAN MODEL UTAMA HASIL HYPERPARAMETER TUNING ===')
display(progress_comparison_df.round(4))

# Grafik dimulai dari nol agar perbedaan nilai tidak tampak berlebihan.
plot_df = all_candidates_summary_df[['Model', 'Val Accuracy Bobot Akhir (%)', 'Test Accuracy (%)']].copy()
ax = plot_df.set_index('Model').plot(kind='bar', figsize=(10, 5), ylim=(0, 100), rot=0)
ax.set_ylabel('Akurasi (%)'); ax.set_xlabel('')
ax.set_title('Perbandingan akurasi: validation dan testing')
ax.grid(axis='y', alpha=0.3)
plt.tight_layout(); plt.show()
print('Catatan: Model 3 adalah model utama hasil KerasTuner. Pembanding tidak mengganti model utama berdasarkan hasil test.')


# ================================================================================
# SECTION 11 — sumber: notebook sel 32
# ================================================================================
# ============================================================================
# SECTION 11 - DUA FITUR UNIK, DENGAN PENGUKURAN NYATA
# ============================================================================
# (1) Pilih ambang confidence pada VAL; (2) permutation importance pada VAL.
# Data test hanya dipakai SETELAH ambang telah dibekukan.
probs_val_final = final_model.predict(X_val_scaled, verbose=0)
probs_test_final = final_model.predict(X_test_scaled, verbose=0)
max_conf_val = np.max(probs_val_final, axis=1)
max_conf_test = np.max(probs_test_final, axis=1)
pred_cls_val = np.argmax(probs_val_final, axis=1)
pred_cls_test = np.argmax(probs_test_final, axis=1)

threshold_rows = []
for tau in np.arange(0.50, 0.951, 0.05):
    mask = max_conf_val >= tau
    accepted = int(mask.sum())
    if accepted:
        acc = accuracy_score(y_val[mask], pred_cls_val[mask])
        f1 = f1_score(y_val[mask], pred_cls_val[mask], average='macro', zero_division=0)
    else:
        acc, f1 = np.nan, np.nan
    threshold_rows.append({
        'Threshold': float(round(tau, 2)),
        'Jumlah Diterima': accepted,
        'Jumlah Ditinjau': int((~mask).sum()),
        'Coverage': float(mask.mean()),
        'Accepted Accuracy': float(acc),
        'Accepted Macro F1': float(f1),
    })
thresh_val_df = pd.DataFrame(threshold_rows)
# Syarat operasional: setidaknya 80% validasi diterima otomatis.
# Maksimalkan F1 pada prediksi yang diterima; seri dipecahkan oleh coverage.
eligible = thresh_val_df[
    (thresh_val_df['Coverage'] >= 0.80) & thresh_val_df['Accepted Macro F1'].notna()
].copy()
if eligible.empty:
    raise RuntimeError('Tidak ada ambang yang lolos syarat coverage 80%; periksa data validasi.')
chosen_threshold_row = eligible.sort_values(
    ['Accepted Macro F1', 'Coverage', 'Threshold'],
    ascending=[False, False, True], kind='stable'
).iloc[0]
OPTIMAL_TAU = float(chosen_threshold_row['Threshold'])
print('=== FITUR 1: HASIL KALIBRASI AMBANG PADA DATA VALIDASI ===')
display(thresh_val_df.round(4))
print(f'Ambang dipilih dari validasi: tau={OPTIMAL_TAU:.2f}, '
      f'coverage={chosen_threshold_row["Coverage"]:.2%}, '
      f'accepted F1={chosen_threshold_row["Accepted Macro F1"]:.4f}')

# Satu evaluasi ambang terpilih pada data test, tanpa mengubah tau berdasarkan hasil test.
mask_test = max_conf_test >= OPTIMAL_TAU
n_accept = int(mask_test.sum())
if n_accept == 0:
    raise RuntimeError('Ambang terpilih tidak menerima satu pun sampel pada data test.')
base_f1 = f1_score(y_test, pred_cls_test, average='macro', zero_division=0)
mask_err = y_test != pred_cls_test
threshold_test_df = pd.DataFrame([{
    'Threshold dari validasi': OPTIMAL_TAU,
    'Total test': int(len(y_test)),
    'Diterima otomatis': n_accept,
    'Ditinjau': int((~mask_test).sum()),
    'Coverage test': float(mask_test.mean()),
    'Akurasi semua test': float(accuracy_score(y_test, pred_cls_test)),
    'F1 makro semua test': float(base_f1),
    'Akurasi test yang diterima': float(accuracy_score(y_test[mask_test], pred_cls_test[mask_test])),
    'F1 makro test yang diterima': float(f1_score(
        y_test[mask_test], pred_cls_test[mask_test], average='macro', zero_division=0
    )),
    'Total kesalahan awal': int(mask_err.sum()),
    'Kesalahan yang ditandai untuk tinjauan': int(np.sum(mask_err & (~mask_test))),
    'Kesalahan yang masih diterima otomatis': int(np.sum(mask_err & mask_test)),
}])
print('\n=== FITUR 1: PENGUJIAN AMBANG TETAP PADA DATA TEST ===')
display(threshold_test_df.round(4))

# Fungsi yang sama dipakai dalam uji dan dapat digunakan oleh antarmuka SDK Static.
def predict_with_confidence_review(df_input, model_obj, prep_obj, tau=None):
    tau = OPTIMAL_TAU if tau is None else float(tau)
    raw = df_input.copy()
    if 'ppi' not in raw.columns:
        raw['ppi'] = np.hypot(raw['resolution_height'], raw['resolution_width']) / raw['display_size']
    X_input = prep_obj.transform(raw[num_features + cat_features]).astype(np.float32)
    prob = model_obj.predict(X_input, verbose=0)
    idxs = prob.argmax(axis=1)
    return pd.DataFrame({
        'Kelas Prediksi': [label_names[int(i)] for i in idxs],
        'Confidence': np.max(prob, axis=1),
        'Status': np.where(np.max(prob, axis=1) >= tau, 'Prediksi otomatis', 'Perlu ditinjau'),
    })

# Fitur 2: Permutation importance pada 19 FITUR ASLI (bukan kolom one-hot terpisah).
# Semua nilai yang ditampilkan berasal dari hasil prediksi yang baru dihitung.
all_original_features = num_features + cat_features
K_REPEATS = 10
baseline_val_f1 = f1_score(y_val, pred_cls_val, average='macro', zero_division=0)
rng_perm = np.random.default_rng(SEED)
permutation_rows = []
for col_name in all_original_features:
    drops = []
    for repeat in range(K_REPEATS):
        X_perm = X_val.copy()
        X_perm[col_name] = rng_perm.permutation(X_perm[col_name].to_numpy())
        transformed = preprocessor.transform(X_perm).astype(np.float32)
        perm_pred = final_model.predict(transformed, verbose=0).argmax(axis=1)
        perm_f1 = f1_score(y_val, perm_pred, average='macro', zero_division=0)
        drops.append(baseline_val_f1 - perm_f1)
    permutation_rows.append({
        'Nama Fitur Asli': col_name,
        'Jenis Fitur': 'Numerik' if col_name in num_features else 'Kategorikal',
        'Penurunan F1 Rata-rata': float(np.mean(drops)),
        'Simpangan Baku': float(np.std(drops, ddof=1)),
        'Jumlah Ulangan': K_REPEATS,
    })
perm_importance_df = pd.DataFrame(permutation_rows).sort_values(
    'Penurunan F1 Rata-rata', ascending=False
).reset_index(drop=True)
perm_importance_df.insert(0, 'Peringkat', np.arange(1, len(perm_importance_df)+1))
print('\n=== FITUR 2: PERMUTATION IMPORTANCE NYATA, VALIDASI 10 KALI/FITUR ===')
display(perm_importance_df.round(5))
print('Nilai negatif berarti pengacakan tidak mengurangi F1; bukan bukti fitur merugikan secara kausal.')

fig, axes = plt.subplots(1, 2, figsize=(14, 5.8))
axes[0].plot(thresh_val_df['Threshold'], thresh_val_df['Coverage'], 'o-', label='Coverage')
axes[0].plot(thresh_val_df['Threshold'], thresh_val_df['Accepted Accuracy'], 's-', label='Accepted accuracy')
axes[0].plot(thresh_val_df['Threshold'], thresh_val_df['Accepted Macro F1'], '^-', label='Accepted F1')
axes[0].axvline(OPTIMAL_TAU, linestyle='--', color='black', label=f'Tau terpilih={OPTIMAL_TAU:.2f}')
axes[0].set_ylim(0, 1); axes[0].set_xlabel('Confidence threshold'); axes[0].set_ylabel('Proporsi / skor')
axes[0].set_title('Trade-off: coverage dan kualitas prediksi'); axes[0].legend(fontsize=8); axes[0].grid(alpha=0.3)
view = perm_importance_df.sort_values('Penurunan F1 Rata-rata')
axes[1].barh(view['Nama Fitur Asli'], view['Penurunan F1 Rata-rata'], xerr=view['Simpangan Baku'])
axes[1].set_title('Permutation importance pada data validasi')
axes[1].set_xlabel('Penurunan F1 makro (semakin besar semakin penting)'); axes[1].grid(axis='x', alpha=0.3)
plt.tight_layout(); plt.show()


# ================================================================================
# SECTION 12 — sumber: notebook sel 35
# ================================================================================
# ============================================================================
# SECTION 12 - EKSPERIMEN ABLASI AKTUAL, MULTI-SEED & DIAGNOSTIK GROUP-SPLIT
# ============================================================================
# Eksperimen pendukung menggunakan VALIDASI; test utama tidak dipakai memilih model.
import gc
from sklearn.model_selection import GroupShuffleSplit

RUN_ABLATION = True
RUN_MULTI_SEED = True
RUN_GROUP_SENSITIVITY = True

# Replikasi konfigurasi yang diukur: semua eksperimen A-E mempunyai
# arsitektur 128-64-32. Perubahan ditambahkan satu per satu.
def build_controlled_ablation_mlp(input_dim, num_classes, opt_name='sgd',
                                  lr=0.01, use_bn=False, use_dropout=False, l2_val=0.0):
    reg = tf.keras.regularizers.l2(l2_val) if l2_val else None
    layers = [tf.keras.layers.Input(shape=(input_dim,))]
    for idx, units in enumerate([128, 64, 32]):
        layers.append(tf.keras.layers.Dense(units, kernel_regularizer=reg))
        if use_bn:
            layers.append(tf.keras.layers.BatchNormalization())
        layers.append(tf.keras.layers.Activation('relu'))
        if use_dropout and idx < 2:
            layers.append(tf.keras.layers.Dropout([0.3, 0.2][idx]))
    layers.append(tf.keras.layers.Dense(num_classes, activation='softmax'))
    model = tf.keras.Sequential(layers)
    opt = tf.keras.optimizers.Adam(learning_rate=lr) if opt_name == 'adam' \
        else tf.keras.optimizers.SGD(learning_rate=lr)
    model.compile(optimizer=opt, loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model

def _callbacks():
    return [
        tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5,
                                             patience=5, min_lr=1e-5, verbose=0),
        tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=15,
                                         restore_best_weights=True, verbose=0),
    ]

def _fit_and_val(model, seed, epochs=100, with_callbacks=True,
                 x_train=None, yy_train=None, x_val=None, yy_val=None):
    tx = X_train_scaled if x_train is None else x_train
    ty = y_train if yy_train is None else yy_train
    vx = X_val_scaled if x_val is None else x_val
    vy = y_val if yy_val is None else yy_val
    tf.keras.utils.set_random_seed(int(seed))
    data = tf.data.Dataset.from_tensor_slices((tx, ty)).shuffle(
        len(tx), seed=int(seed), reshuffle_each_iteration=True
    ).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
    history = model.fit(data, validation_data=(vx, vy), epochs=epochs,
                        callbacks=_callbacks() if with_callbacks else [], verbose=0)
    pred = model.predict(vx, verbose=0).argmax(axis=1)
    ev = model.evaluate(vx, vy, verbose=0, return_dict=True)
    return {
        'Val Accuracy (%)': float(accuracy_score(vy, pred) * 100),
        'Val Macro F1': float(f1_score(vy, pred, average='macro', zero_division=0)),
        'Val Loss': float(ev['loss']),
        'Epoch Dilatih': int(len(history.history['loss'])),
        'Jumlah Parameter': int(model.count_params()),
    }

ablation_configs = [
    ('A: Kapasitas 3-layer + SGD', dict(opt_name='sgd', lr=0.01, use_bn=False, use_dropout=False, l2_val=0)),
    ('B: A + Adam', dict(opt_name='adam', lr=0.001, use_bn=False, use_dropout=False, l2_val=0)),
    ('C: B + BatchNorm', dict(opt_name='adam', lr=0.001, use_bn=True, use_dropout=False, l2_val=0)),
    ('D: C + Dropout', dict(opt_name='adam', lr=0.001, use_bn=True, use_dropout=True, l2_val=0)),
    ('E: D + L2', dict(opt_name='adam', lr=0.001, use_bn=True, use_dropout=True, l2_val=1e-4)),
]
ablation_rows = []
if RUN_ABLATION:
    for name, cfg in ablation_configs:
        tf.keras.utils.set_random_seed(SEED)
        ab_model = build_controlled_ablation_mlp(INPUT_DIM, NUM_CLASSES, **cfg)
        m = _fit_and_val(ab_model, SEED, epochs=100)
        ablation_rows.append({'Eksperimen': name, **cfg, **m})
        del ab_model
        gc.collect()
    # F adalah arsitektur hasil tuning, bukan satu perubahan terkendali dari E.
    final_val = val_selection_df.loc[val_selection_df['Model'] == final_model_name].iloc[0]
    ablation_rows.append({
        'Eksperimen': 'F: Kandidat hasil tuning (pembanding, bukan ablasi satu variabel)',
        'opt_name': str(best_hp.get('optimizer')),
        'lr': float(best_hp.get('learning_rate')),
        'use_bn': True, 'use_dropout': True, 'l2_val': float(best_hp.get('l2_reg')),
        'Val Accuracy (%)': float(final_val['Val Accuracy Bobot Akhir (%)']),
        'Val Macro F1': float(final_val['Val Macro F1']),
        'Val Loss': float(final_val['Val Loss Bobot Akhir']),
        'Epoch Dilatih': int(final_val['Epoch Dilatih']),
        'Jumlah Parameter': int(final_val['Jumlah Parameter']),
    })
ablation_study_df = pd.DataFrame(ablation_rows)
print('=== ABLASI: METRIK VALIDASI HASIL PELATIHAN A-E + PEMBANDING F ===')
display(ablation_study_df.round(5))

# Lima seed untuk setiap model. Setiap baris berasal dari .fit() terpisah.
# Test tidak dievaluasi ulang pada bagian ini untuk mencegah pengambilan keputusan dari test.
SEEDS_LIST = [42, 52, 62, 72, 82]
seed_rows = []
if RUN_MULTI_SEED:
    for model_name, _, _ in model_candidates:
        for seed in SEEDS_LIST:
            tf.keras.utils.set_random_seed(seed)
            if model_name.startswith('Model 1'):
                candidate = build_baseline_mlp(INPUT_DIM, NUM_CLASSES)
                candidate.compile(optimizer=tf.keras.optimizers.SGD(learning_rate=0.01),
                                  loss='sparse_categorical_crossentropy', metrics=['accuracy'])
                stats = _fit_and_val(candidate, seed, epochs=50, with_callbacks=False)
            elif model_name.startswith('Model 2'):
                candidate = build_regularized_mlp(INPUT_DIM, NUM_CLASSES)
                candidate.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
                                  loss='sparse_categorical_crossentropy', metrics=['accuracy'])
                stats = _fit_and_val(candidate, seed, epochs=100)
            else:
                candidate = tuner.hypermodel.build(best_hp)
                stats = _fit_and_val(candidate, seed, epochs=100)
            seed_rows.append({'Model': model_name, 'Seed': seed, **stats})
            del candidate
            gc.collect()
multiseed_detail_df = pd.DataFrame(seed_rows)
if not multiseed_detail_df.empty:
    multiseed_summary_df = multiseed_detail_df.groupby('Model').agg(
        Val_Accuracy_Mean=('Val Accuracy (%)','mean'),
        Val_Accuracy_Std=('Val Accuracy (%)','std'),
        Val_F1_Mean=('Val Macro F1','mean'),
        Val_F1_Std=('Val Macro F1','std'),
        Val_Loss_Mean=('Val Loss','mean'),
        Val_Loss_Std=('Val Loss','std'),
        Runs=('Seed','count')
    ).reset_index()
else:
    multiseed_summary_df = pd.DataFrame()
print('\n=== HASIL LIMA SEED: VALIDASI (SETIAP SEED DILATIH ULANG) ===')
display(multiseed_detail_df.round(5))
print('\n=== RATA-RATA DAN SIMPANGAN BAKU HASIL SEED NYATA ===')
display(multiseed_summary_df.round(5))

if not ablation_study_df.empty:
    ax = ablation_study_df.plot.bar(x='Eksperimen', y='Val Macro F1', figsize=(11, 4.5),
                                   legend=False, rot=25)
    ax.set_ylim(0, 1); ax.set_ylabel('F1 makro validasi')
    ax.set_title('Ablasi pada validasi (F adalah pembanding tuned)')
    ax.grid(axis='y', alpha=0.3); plt.tight_layout(); plt.show()

# Pemeriksaan sensitivitas terhadap variasi laptop yang belum pernah muncul dalam training.
# Ini EKSPERIMEN PENDAMPING, bukan nilai test final utama. Best_hp sudah dipilih pada
# split acak sebelumnya; oleh sebab itu hasil group split tidak disebut benchmark independen.
# Pengelompokan memakai nama model yang dinormalisasi, bukan spesifikasi/harga.
group_sensitivity_df = pd.DataFrame()
if RUN_GROUP_SENSITIVITY:
    X_all = df[num_features + cat_features].copy()
    X_all['Touch_Screen'] = X_all['Touch_Screen'].astype(str)
    yy_all = df['category'].to_numpy()
    groups = df['model'].fillna('tidak-diketahui').astype(str).str.strip().str.lower().to_numpy()
    outer = GroupShuffleSplit(n_splits=1, test_size=0.15, random_state=SEED)
    idx_tv, idx_gtest = next(outer.split(X_all, yy_all, groups))
    inner = GroupShuffleSplit(n_splits=1, test_size=0.17647, random_state=SEED)
    idx_gtr_local, idx_gval_local = next(inner.split(
        X_all.iloc[idx_tv], yy_all[idx_tv], groups[idx_tv]
    ))
    idx_gtr, idx_gval = idx_tv[idx_gtr_local], idx_tv[idx_gval_local]
    assert not (set(groups[idx_gtr]) & set(groups[idx_gtest]))
    assert not (set(groups[idx_gval]) & set(groups[idx_gtest]))
    assert not (set(groups[idx_gtr]) & set(groups[idx_gval]))
    from sklearn.base import clone
    group_prep = clone(preprocessor)
    gx_tr = group_prep.fit_transform(X_all.iloc[idx_gtr]).astype(np.float32)
    gx_val = group_prep.transform(X_all.iloc[idx_gval]).astype(np.float32)
    gx_test = group_prep.transform(X_all.iloc[idx_gtest]).astype(np.float32)
    # Build with the tuned hyperparameters but with the new input dimension.
    # Hypermodel uses the global input dimension; setting and restoring it is isolated here.
    old_input_dim = INPUT_DIM
    try:
        INPUT_DIM = int(gx_tr.shape[1])
        tf.keras.utils.set_random_seed(SEED)
        group_model = build_tunable_mlp(best_hp)
    finally:
        INPUT_DIM = old_input_dim
    gr = _fit_and_val(group_model, SEED, epochs=100, x_train=gx_tr,
                      yy_train=yy_all[idx_gtr], x_val=gx_val, yy_val=yy_all[idx_gval])
    grp_pred = group_model.predict(gx_test, verbose=0).argmax(axis=1)
    group_sensitivity_df = pd.DataFrame([{
        'Tipe pengujian': 'Sensitivitas: group-disjoint nama model (bukan benchmark independen)',
        'Jumlah train': int(len(idx_gtr)), 'Jumlah val': int(len(idx_gval)),
        'Jumlah test': int(len(idx_gtest)),
        'Overlap grup': 0,
        'Akurasi validasi': float(gr['Val Accuracy (%)'] / 100),
        'F1 makro validasi': float(gr['Val Macro F1']),
        'Akurasi test grup': float(accuracy_score(yy_all[idx_gtest], grp_pred)),
        'F1 makro test grup': float(f1_score(yy_all[idx_gtest], grp_pred, average='macro', zero_division=0)),
    }])
    del group_model
    gc.collect()
print('\n=== DIAGNOSTIK GENERALISASI TERHADAP NAMA MODEL YANG TIDAK BERTUMPAH TINDIH ===')
display(group_sensitivity_df.round(5))
print('Catatan: pemilihan hyperparameter awal sudah menggunakan split acak; '
      'eksperimen group-disjoint ini hanya pengujian sensitivitas tambahan, bukan estimasi bebas bias.')


# ================================================================================
# SECTION 13 — sumber: notebook sel 38
# ================================================================================
# ============================================================================
# SECTION 13 - SIMPAN, MUAT ULANG, DAN PAKET DEPLOYMENT SDK STATIC
# ============================================================================
import hashlib
import subprocess
from pathlib import Path
import zipfile
ARTIFACT_DIR = Path('artifacts')
ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)

# Model utama hasil tuning dan transformer disimpan dari objek evaluasi.
assert final_model is tuned_mlp, "Deployment wajib menggunakan Model 3 hasil KerasTuner."
model_path = ARTIFACT_DIR / 'model_final.keras'
prep_path = ARTIFACT_DIR / 'preprocessor.joblib'
final_model.save(model_path)
joblib.dump(preprocessor, prep_path)

# Transformer didefinisikan di preprocessors.py, bukan __main__.
assert GroupMedianImputer.__module__ == 'preprocessors', (
    'GroupMedianImputer harus berasal dari modul preprocessors.py agar joblib portable.'
)
import shutil
shutil.copy2('preprocessors.py', ARTIFACT_DIR / 'preprocessors.py')

feature_schema = {
    'project_title': 'Klasifikasi Kelas Harga Laptop Berdasarkan Spesifikasi Teknis Menggunakan Arsitektur Multi-Layer Perceptron (MLP) dengan Hyperparameter Tuning',
    'method': 'Multi-Layer Perceptron (MLP) dengan KerasTuner RandomSearch',
    'model_role': 'model utama hasil hyperparameter tuning',
    'selected_final_model': final_model_name,
    'input_tensor_dim': int(INPUT_DIM),
    'confidence_threshold_tau': float(OPTIMAL_TAU),
    'currency_reference': {
        'label_currency': 'INR',
        'display_currency': 'IDR',
        'idr_per_inr': float(KURS_IDR),
        'as_of': TANGGAL_KURS,
        'basis': SUMBER_KURS,
        'interpretation': 'Ekuivalen ilustratif, bukan harga pasar Indonesia',
    },
    'confidence_selection_rule': 'maksimum accepted F1 di validation, coverage minimal 80%',
    'numerical_features': list(num_features),
    'categorical_features': list(cat_features),
    'numerical_defaults': {
        c: float(X_train[c].median()) for c in num_features
    },
    'training_ranges': {
        c: {'min': float(X_train[c].min()), 'max': float(X_train[c].max())}
        for c in num_features
    },
    'categorical_options': {
        c: sorted(X_train[c].fillna('Tidak diketahui').astype(str).unique().tolist())
        for c in cat_features
    },
    'feature_engineering': {
        'ppi': 'sqrt(resolution_height**2 + resolution_width**2) / display_size',
        'ram_type': 'kategori generasi RAM dari spesifikasi',
    },
}
(ARTIFACT_DIR / 'feature_schema.json').write_text(
    json.dumps(feature_schema, ensure_ascii=False, indent=2), encoding='utf-8'
)
class_labels_meta = {
    'currency': 'INR',
    'display_currency': 'INR / IDR',
    'illustrative_idr_per_inr': float(KURS_IDR),
    'exchange_rate_as_of': TANGGAL_KURS,
    'exchange_rate_source': SUMBER_KURS,
    'warning': 'Konversi IDR hanya ilustrasi; bukan prediksi harga pasar Indonesia.',
    'classes': {
        '0': {'short_label': label_names[0], 'price_rule_inr': '<50000',
              'price_range_dual': rentang_kelas_dual(0)},
        '1': {'short_label': label_names[1], 'price_rule_inr': '50000..85000',
              'price_range_dual': rentang_kelas_dual(1)},
        '2': {'short_label': label_names[2], 'price_rule_inr': '>85000',
              'price_range_dual': rentang_kelas_dual(2)},
    },
}
(ARTIFACT_DIR / 'class_labels.json').write_text(
    json.dumps(class_labels_meta, ensure_ascii=False, indent=2), encoding='utf-8'
)
# Catat ruang pencarian dan hyperparameter dari proses tuning yang benar-benar dijalankan.
(ARTIFACT_DIR / 'tuning_hyperparameters.json').write_text(
    json.dumps({
        'research_title': feature_schema['project_title'],
        'algorithm': 'KerasTuner RandomSearch',
        'trials': int(len(tuner.oracle.trials)),
        'executions_per_trial': 2,
        'objective': 'val_accuracy',
        'best_trial_id': str(best_trial_id),
        'best_hyperparameters': dict(best_hp.values),
    }, ensure_ascii=False, indent=2, allow_nan=False),
    encoding='utf-8'
)

# Rekam seluruh nilai dari tabel hasil KOMPUTASI, bukan angka yang diketik.
# Dataset yang digunakan evaluasi utama tetap split stratified acak.
evaluation_meta = {
    'dataset_rows_cleaned': int(len(df)),
    'split_sizes': {'train': int(len(X_train)), 'val': int(len(X_val)), 'test': int(len(X_test))},
    'primary_model_from_tuning': final_model_name,
    'model_design_selection_rule': final_selection_basis,
    'best_hyperparameters': dict(best_hp.values),
    'validation_model_comparison': json.loads(val_selection_df.to_json(orient='records')),
    'test_model_comparison_descriptive': json.loads(results_df.to_json(orient='records')),
    'threshold_val_tuning': json.loads(thresh_val_df.to_json(orient='records')),
    'threshold_test_locked': json.loads(threshold_test_df.to_json(orient='records')),
    'permutation_importance_val': json.loads(perm_importance_df.to_json(orient='records')),
    'ablation_val': json.loads(ablation_study_df.to_json(orient='records')),
    'multi_seed_val': json.loads(multiseed_detail_df.to_json(orient='records')),
    'multi_seed_val_summary': json.loads(multiseed_summary_df.to_json(orient='records')),
    'group_split_sensitivity_not_independent_benchmark': json.loads(group_sensitivity_df.to_json(orient='records')),
    'limitations': [
        'Label merupakan kelas harga yang dibuat dari ambang INR pada dataset asal. Padanan IDR memakai kurs ilustratif 1 INR = Rp192,549 per 2026-11-09, bukan harga pasar Indonesia.',
        'Pembagian data utama bersifat acak dan terdapat risiko model laptop serupa lintas subset.',
        'Eksperimen group-disjoint adalah uji sensitivitas, bukan benchmark independen untuk tuning.',
        'Confidence softmax tidak otomatis merupakan probabilitas yang terkalibrasi.',
    ],
}
(ARTIFACT_DIR / 'evaluation_results.json').write_text(
    json.dumps(evaluation_meta, ensure_ascii=False, indent=2, allow_nan=False),
    encoding='utf-8'
)

# Inferensi pada Hugging Face SDK Static berjalan di JavaScript.
# Model Keras dan joblib tetap disimpan untuk arsip ilmiah; di browser digunakan
# bobot numerik, scaler/encoder, dan median training yang diekspor ke JSON.

# 1. Ekspor preprocessing scikit-learn -> JSON (fit hanya pada data latih).
learned_imputer = preprocessor.named_steps['group_imputer']
learned_encoder = preprocessor.named_steps['feature_encoder']
learned_scaler = learned_encoder.named_transformers_['num']
learned_onehot = learned_encoder.named_transformers_['cat']
assert list(learned_encoder.transformers_[0][2]) == list(num_features)
assert list(learned_encoder.transformers_[1][2]) == list(cat_features)

preprocessing_browser = {
    'input_tensor_dim': int(INPUT_DIM),
    'numerical_features': list(num_features),
    'categorical_features': list(cat_features),
    'numerical_means': [float(z) for z in learned_scaler.mean_],
    'numerical_scales': [float(z) for z in learned_scaler.scale_],
    'numerical_defaults': feature_schema['numerical_defaults'],
    'categories': [[str(v) for v in arr] for arr in learned_onehot.categories_],
    'imputation': {
        'group_col': str(learned_imputer.group_col),
        'global_medians': {k: float(v) for k, v in learned_imputer.global_medians_.items()},
        'group_medians': {
            k: {str(g): float(v) for g, v in mapping.items()}
            for k, mapping in learned_imputer.group_medians_.items()
        },
    },
}
assert len(preprocessing_browser['numerical_means']) == len(num_features)
assert (sum(len(x) for x in preprocessing_browser['categories']) + len(num_features)) == INPUT_DIM
(ARTIFACT_DIR / 'preprocessing_browser.json').write_text(
    json.dumps(preprocessing_browser, indent=2, ensure_ascii=False, allow_nan=False), encoding='utf-8'
)

# 2. Ekspor seluruh lapisan dari MLP hasil KerasTuner.
#    BatchNormalization memakai moving mean/variance pada inference,
#    sedangkan Dropout dilewati karena nonaktif saat inference.
web_layers = []
for layer in final_model.layers:
    layer_type = layer.__class__.__name__
    if layer_type == 'InputLayer':
        continue
    if layer_type == 'Dense':
        kernel, *remaining = layer.get_weights()
        bias = remaining[0] if remaining else np.zeros(kernel.shape[1], dtype=np.float32)
        web_layers.append({
            'type': 'Dense', 'name': layer.name,
            'input_dim': int(kernel.shape[0]), 'output_dim': int(kernel.shape[1]),
            'activation': layer.activation.__name__,
            'kernel': kernel.tolist(), 'bias': bias.tolist(),
        })
    elif layer_type == 'BatchNormalization':
        weights = layer.get_weights()
        assert len(weights) == 4 and layer.center and layer.scale, (
            'Konfigurasi BatchNormalization baru perlu dicek sebelum ekspor JS.'
        )
        gamma, beta, moving_mean, moving_variance = weights
        web_layers.append({
            'type': 'BatchNormalization', 'name': layer.name,
            'gamma': gamma.tolist(), 'beta': beta.tolist(),
            'mean': moving_mean.tolist(), 'variance': moving_variance.tolist(),
            'epsilon': float(layer.epsilon),
        })
    elif layer_type == 'Activation':
        web_layers.append({'type': 'Activation', 'name': layer.name,
                           'activation': layer.activation.__name__})
    elif layer_type == 'ReLU':
        web_layers.append({'type': 'Activation', 'name': layer.name, 'activation': 'relu'})
    elif layer_type == 'Dropout':
        web_layers.append({'type': 'Dropout', 'name': layer.name})
    else:
        raise ValueError('Layer belum didukung pada Static SDK: ' + layer_type)
assert web_layers[0]['type'] == 'Dense'
model_browser = {
    'model': 'Model 3: Tuned MLP',
    'project_title': feature_schema['project_title'],
    'input_tensor_dim': int(INPUT_DIM),
    'output_classes': int(NUM_CLASSES),
    'layers': web_layers,
}
(ARTIFACT_DIR / 'model_browser.json').write_text(
    json.dumps(model_browser, ensure_ascii=False, allow_nan=False, separators=(',', ':')), encoding='utf-8'
)

# 3. Aset frontend statis: HTML, CSS, JavaScript; tidak ada backend Python.
# Jika Section 13 pernah dijalankan pada runtime sebelumnya, bersihkan berkas
# aplikasi server lama agar folder artifacts tidak memuat dua jenis deployment.
for legacy_name in ('app.py', 'requirements.txt'):
    (ARTIFACT_DIR / legacy_name).unlink(missing_ok=True)
STATIC_INFERENCE_JS = "/* Inferensi MLP dan preprocessing di browser, tanpa backend Python.\n   Seluruh parameter dibaca dari artefak hasil Section 13 notebook. */\n(function (root) {\n  'use strict';\n  function assert(cond, message) { if (!cond) throw new Error(message); }\n  function isBlank(value) { return value === null || value === undefined || value === ''; }\n\n  function transformItem(raw, prep) {\n    const item = Object.assign({}, raw);\n    const numeric = prep.numerical_features;\n    const cats = prep.categorical_features;\n    assert(prep.numerical_means.length === numeric.length, 'Dimensi mean numerik tidak cocok.');\n    assert(prep.numerical_scales.length === numeric.length, 'Dimensi skala numerik tidak cocok.');\n    assert(prep.categories.length === cats.length, 'Dimensi encoder kategorikal tidak cocok.');\n    for (const col of cats) {\n      item[col] = String(isBlank(item[col]) ? '' : item[col]);\n    }\n    for (const col of numeric) {\n      if (col === 'ppi') continue;\n      if (isBlank(item[col])) {\n        if (Object.prototype.hasOwnProperty.call(prep.imputation.global_medians, col)) {\n          const group = item[prep.imputation.group_col];\n          const candidate = prep.imputation.group_medians[col][group];\n          item[col] = candidate === undefined ? prep.imputation.global_medians[col] : candidate;\n        } else {\n          item[col] = prep.numerical_defaults[col];\n        }\n      } else {\n        item[col] = Number(item[col]);\n      }\n      assert(Number.isFinite(item[col]), 'Fitur numerik tidak valid: ' + col);\n    }\n    if (numeric.includes('ppi')) {\n      assert(item.display_size > 0, 'Ukuran layar harus lebih dari nol.');\n      item.ppi = Math.hypot(item.resolution_height, item.resolution_width) / item.display_size;\n    }\n    let vector = [];\n    for (let i = 0; i < numeric.length; i++) {\n      let value = Number(item[numeric[i]]);\n      assert(Number.isFinite(value), 'Fitur numerik tidak valid: ' + numeric[i]);\n      vector.push(Math.fround((value - prep.numerical_means[i]) / prep.numerical_scales[i]));\n    }\n    for (let i = 0; i < cats.length; i++) {\n      for (const category of prep.categories[i]) {\n        vector.push(item[cats[i]] === String(category) ? 1 : 0);\n      }\n    }\n    assert(vector.length === prep.input_tensor_dim, 'Input model tidak sesuai dimensi preprocessing.');\n    return { vector: vector, completed: item };\n  }\n\n  function runLayers(features, net) {\n    let x = features.slice();\n    for (const layer of net.layers) {\n      if (layer.type === 'Dense') {\n        assert(x.length === layer.input_dim, 'Dimensi Dense tidak sesuai.');\n        const y = new Array(layer.output_dim);\n        for (let j = 0; j < layer.output_dim; j++) {\n          let sum = layer.bias[j];\n          for (let i = 0; i < x.length; i++) sum += x[i] * layer.kernel[i][j];\n          y[j] = sum;\n        }\n        x = y;\n        if (layer.activation === 'relu') x = x.map(v => Math.max(0, v));\n        else if (layer.activation === 'sigmoid') x = x.map(v => 1 / (1 + Math.exp(-v)));\n        else if (layer.activation === 'tanh') x = x.map(v => Math.tanh(v));\n        else if (layer.activation === 'softmax') x = softmax(x);\n        else assert(!layer.activation || layer.activation === 'linear', 'Aktivasi Dense tidak didukung: ' + layer.activation);\n      } else if (layer.type === 'BatchNormalization') {\n        assert(x.length === layer.mean.length, 'Dimensi BatchNormalization tidak cocok.');\n        x = x.map((v, i) => (v - layer.mean[i]) * layer.gamma[i] /\n          Math.sqrt(layer.variance[i] + layer.epsilon) + layer.beta[i]);\n      } else if (layer.type === 'Activation') {\n        if (layer.activation === 'relu') x = x.map(v => Math.max(0, v));\n        else if (layer.activation === 'softmax') x = softmax(x);\n        else if (layer.activation === 'tanh') x = x.map(v => Math.tanh(v));\n        else if (layer.activation === 'linear') {}\n        else throw new Error('Aktivasi tidak didukung: ' + layer.activation);\n      } else if (layer.type === 'Dropout') {\n        // Dropout nonaktif pada mode inference; tidak mengubah nilai aktivasi.\n      } else {\n        throw new Error('Jenis layer tidak didukung: ' + layer.type);\n      }\n    }\n    assert(x.every(Number.isFinite), 'Probabilitas prediksi tidak valid.');\n    return x;\n  }\n\n  function softmax(z) {\n    const mx = Math.max(...z), e = z.map(v => Math.exp(v - mx));\n    const sum = e.reduce((a, b) => a + b, 0);\n    return e.map(v => v / sum);\n  }\n\n  function predict(raw, prep, net, labels, schema) {\n    const t = transformItem(raw, prep);\n    const probabilities = runLayers(t.vector, net);\n    let classIndex = probabilities.indexOf(Math.max(...probabilities));\n    const confidence = probabilities[classIndex];\n    const outside = [];\n    for (const col of prep.numerical_features) {\n      if (col === 'ppi') continue;\n      const limits = schema.training_ranges && schema.training_ranges[col];\n      if (limits && (t.completed[col] < limits.min || t.completed[col] > limits.max)) outside.push(col);\n    }\n    return {\n      classIndex,\n      probabilities,\n      confidence,\n      classInfo: labels.classes[String(classIndex)],\n      requiresReview: confidence < Number(schema.confidence_threshold_tau),\n      outsideTrainingRange: outside,\n      featureVector: t.vector\n    };\n  }\n  root.LaptopInference = { transformItem, runLayers, predict };\n})(typeof globalThis !== 'undefined' ? globalThis : this);\n"
STATIC_APP_JS = "/* Aplikasi SDK Static dengan inferensi JavaScript dan tanpa CDN. */\n(async function(){\n  'use strict';\n  const $ = id => document.getElementById(id);\n  let prep, model, schema, labels;\n  const nice = {rating:'Rating (0–5)',threads_num:'Jumlah Threads',core_num:'Jumlah Core',ram_num:'RAM (GB)',memory_size:'Penyimpanan (GB)',display_size:'Ukuran layar (inci)',resolution_height:'Resolusi tinggi (px)',resolution_width:'Resolusi lebar (px)',warranty:'Garansi (tahun)',brand_name:'Merek laptop',processor_brand:'Merek prosesor',iter:'Seri prosesor',ram_type:'Tipe RAM',memory_type:'Tipe penyimpanan',gpu_brand:'Merek GPU',gpu_type:'Jenis GPU',os:'Sistem operasi',Touch_Screen:'Layar sentuh'};\n  try {\n    const [p,m,s,l,evaluation] = await Promise.all(['preprocessing_browser.json','model_browser.json','feature_schema.json','class_labels.json','evaluation_results.json'].map(async f => {\n      const response = await fetch(f, {cache:'no-store'});\n      if(!response.ok) throw new Error('Gagal membaca '+f+' (HTTP '+response.status+').');\n      return response.json();\n    }));\n    [prep,model,schema,labels] = [p,m,s,l];\n    const formFields = [];\n    for(const name of prep.numerical_features) {\n      if(name === 'ppi') continue;\n      const input = document.createElement('input');\n      input.type = 'number'; input.step = 'any'; input.name = name;\n      const defaultValue = prep.numerical_defaults[name];\n      input.value = Number.isFinite(defaultValue) ? String(defaultValue) : '';\n      input.placeholder = (name === 'core_num' || name === 'threads_num') ? 'Otomatis bila kosong' : 'Masukkan angka';\n      formFields.push([name,input]);\n    }\n    for(let idx=0;idx<prep.categorical_features.length;idx++) {\n      const name=prep.categorical_features[idx];\n      const input=document.createElement('select');input.name=name;\n      for(const v of prep.categories[idx]) {\n        const option=document.createElement('option');option.value=String(v);option.textContent=String(v);input.appendChild(option);\n      }\n      formFields.push([name,input]);\n    }\n    for(const [name,input] of formFields) {\n      const label=document.createElement('label');\n      const text=document.createElement('span');text.textContent=nice[name] || name;\n      label.append(text,input);$('fields').appendChild(label);\n    }\n    const rate=Number(labels.illustrative_idr_per_inr).toLocaleString('id-ID', {maximumFractionDigits:3});\n    $('currency-note').textContent='Acuan konversi ilustratif: 1 INR ≈ Rp'+rate+' ('+labels.exchange_rate_as_of+'). Bukan harga jual laptop di Indonesia. Batas kelas ditentukan dari harga INR asli dataset.';\n    const importance = Array.isArray(evaluation.permutation_importance_val) ? evaluation.permutation_importance_val.slice() : [];\n    importance.sort((a,b) => Number(b['Penurunan F1 Rata-rata']) - Number(a['Penurunan F1 Rata-rata']));\n    const target = $('importance'); target.replaceChildren();\n    if (!importance.length) { target.textContent = 'Hasil permutation importance belum tersedia.'; }\n    else for (const f of importance.slice(0, 6)) {\n      const line = document.createElement('div');line.className='barrow';\n      const name = document.createElement('span');name.textContent=String(f['Nama Fitur Asli'] || 'fitur');\n      const bar=document.createElement('div');bar.className='bar';\n      const val=Math.max(0,Number(f['Penurunan F1 Rata-rata'] || 0));\n      const inner=document.createElement('span');inner.style.width=Math.min(100,val/0.12*100).toFixed(1)+'%';bar.appendChild(inner);\n      const txt=document.createElement('span');txt.textContent=val.toFixed(4);\n      line.append(name,bar,txt);target.appendChild(line);\n    }\n    $('predict').disabled = false;\n    $('predict').textContent = 'Klasifikasikan laptop';\n    $('status').textContent='Model siap. Isi spesifikasi dan tekan tombol klasifikasi.';\n    $('feature-form').addEventListener('submit',function(event){\n      event.preventDefault();\n      try{\n        const row={};for(const [name,input] of formFields) row[name]=input.value;\n        const pred=LaptopInference.predict(row,prep,model,labels,schema);\n        $('status').textContent='Hasil dari MLP hasil hyperparameter tuning';$('result').hidden=false;\n        $('class-name').textContent=pred.classInfo.short_label;\n        $('price-range').textContent='Rentang harga INR / IDR: '+pred.classInfo.price_range_dual;\n        $('confidence').textContent='Keyakinan model: '+(100*pred.confidence).toFixed(2)+'%';\n        $('review').textContent=pred.requiresReview?'Perlu ditinjau: keyakinan prediksi di bawah ambang '+(100*schema.confidence_threshold_tau).toFixed(0)+'% yang ditetapkan melalui validasi.':'Prediksi memenuhi ambang keyakinan validasi.';\n        $('probabilities').replaceChildren();\n        pred.probabilities.forEach((p,i)=>{\n          const wrap=document.createElement('div');wrap.className='barrow';\n          const name=document.createElement('span');name.textContent=labels.classes[String(i)].short_label.split(' ')[0];\n          const bar=document.createElement('div');bar.className='bar';const inner=document.createElement('span');inner.style.width=(p*100).toFixed(2)+'%';bar.appendChild(inner);\n          const pct=document.createElement('span');pct.textContent=(100*p).toFixed(1)+'%';wrap.append(name,bar,pct);$('probabilities').appendChild(wrap);\n        });\n        $('warnings').textContent=pred.outsideTrainingRange.length?'Peringatan: nilai di luar rentang data latih pada '+pred.outsideTrainingRange.join(', ')+'.':'Nilai fitur numerik berada dalam rentang data latih.';\n      }catch(err){$('status').textContent='Tidak dapat menghitung prediksi: '+err.message;$('status').className='error';$('result').hidden=true;}\n    });\n  }catch(err){$('status').textContent='Aplikasi belum siap: '+err.message+' Pastikan seluruh artefak hasil Section 13 diunggah ke Space.';$('status').className='error';}\n})();\n"
STATIC_INDEX_HTML = '<!doctype html>\n<html lang="id">\n<head>\n  <meta charset="utf-8">\n  <meta name="viewport" content="width=device-width,initial-scale=1">\n  <meta name="description" content="Klasifikasi kelas harga laptop berbasis MLP hasil hyperparameter tuning, dijalankan secara lokal di browser.">\n  <title>MLP Laptop Price Classifier | Kelompok 4</title>\n  <style>\n  :root {color-scheme:light; --bg:#f5f5f2;--paper:#fff;--ink:#172438;--muted:#56657a;--line:#dce2e8;--navy:#14263e;--blue:#346fc5;--soft:#f0f5fc}\n  *{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--ink);font:15px/1.55 system-ui,-apple-system,"Segoe UI",sans-serif}\n  .container{max-width:1120px;margin:auto;padding:26px 18px 70px}header{padding:22px 0 16px;border-bottom:1px solid var(--line)}\n  h1{font-size:clamp(24px,4vw,38px);line-height:1.18;margin:8px 0 12px;max-width:900px}h2{font-size:19px;margin:0 0 16px}\n  .eyebrow{font-weight:800;letter-spacing:.12em;text-transform:uppercase;font-size:11px;color:var(--blue)}p{margin:8px 0;color:var(--muted)}\n  .cols{display:grid;grid-template-columns:minmax(0,1.4fr) minmax(0,1fr);gap:18px;margin-top:24px}\n  .panel{background:var(--paper);border:1px solid var(--line);border-radius:14px;padding:22px;box-shadow:0 4px 20px #14263e0a}\n  .fields{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:12px}label{display:flex;flex-direction:column;gap:5px;font-weight:650;font-size:13px}\n  input,select{width:100%;font:inherit;min-width:0;border:1px solid #c8d1db;background:#fff;border-radius:8px;padding:10px;min-height:43px;color:#14263e}\n  input:focus,select:focus{outline:2px solid #7fa6dc;outline-offset:1px}\n  button{border:none;background:var(--navy);color:#fff;border-radius:9px;padding:12px 20px;font:inherit;font-weight:700;cursor:pointer;margin-top:16px;width:100%}\n  button:disabled{opacity:.5;cursor:not-allowed}.meta{font-size:12px;color:var(--muted);padding-top:12px}\n  .result-label{font-weight:800;font-size:23px;margin-top:10px}.class-range{font-weight:650;margin:10px 0}\n  .barrow{display:grid;grid-template-columns:90px 1fr 62px;align-items:center;gap:8px;margin:12px 0;font-size:12px}\n  .bar{height:10px;background:#e5ebf2;border-radius:10px;overflow:hidden}.bar>span{display:block;background:#346fc5;height:100%;border-radius:10px}\n  .notice{background:var(--soft);padding:13px;border-radius:9px;margin:12px 0;color:var(--ink);font-size:13px}\n  .error{color:#a21d34}.muted{color:var(--muted)}.footer{margin-top:22px;color:var(--muted);font-size:13px}\n  @media(max-width:760px){.cols{grid-template-columns:1fr}.fields{grid-template-columns:1fr}.container{padding:18px 12px 50px}.panel{padding:17px}}\n  </style>\n</head>\n<body><main class="container">\n  <header><div class="eyebrow">UTS Deep Learning · Kelompok 4 · SDK Static</div>\n    <h1>Klasifikasi Kelas Harga Laptop Berdasarkan Spesifikasi Teknis Menggunakan Arsitektur Multi-Layer Perceptron (MLP) dengan Hyperparameter Tuning</h1>\n    <p>Inferensi berjalan sepenuhnya di browser menggunakan bobot MLP hasil pelatihan. Tidak memerlukan backend Python atau layanan prediksi berbayar.</p>\n  </header>\n  <div class="cols"><section class="panel"><h2>Masukkan spesifikasi laptop</h2><form id="feature-form"><div id="fields" class="fields"></div><button id="predict" type="submit" disabled>Mempersiapkan model...</button></form><div class="meta">Input tidak dikirim ke server; kalkulasi dilakukan pada perangkat ini.</div></section>\n    <aside class="panel"><h2>Hasil klasifikasi</h2><div id="status" class="muted" role="status" aria-live="polite">Memuat artefak model...</div><div id="result" hidden>\n    <div class="result-label" id="class-name"></div><div class="class-range" id="price-range"></div><div id="confidence" class="muted"></div>\n    <div id="review" class="notice"></div><div id="probabilities"></div><div class="notice" id="warnings"></div>\n    </div><div class="footer" id="currency-note"></div></aside></div>\n<section class="panel" style="margin-top:18px"><h2>Fitur teknis yang paling berpengaruh</h2><p>Ringkasan <em>permutation importance</em> dari pengujian validasi pada notebook. Penurunan F1 merupakan ukuran sensitivitas model, bukan hubungan sebab-akibat.</p><div id="importance" class="muted">Memuat hasil evaluasi...</div></section>\n</main><script src="inference.js"></script><script src="app.js"></script></body></html>\n'
for filename, contents in {
    'inference.js': STATIC_INFERENCE_JS,
    'app.js': STATIC_APP_JS,
    'index.html': STATIC_INDEX_HTML,
}.items():
    (ARTIFACT_DIR / filename).write_text(contents, encoding='utf-8')

readme_static = '''---
title: Laptop Price Tier Classifier - Kelompok 4
emoji: 💻
colorFrom: blue
colorTo: gray
sdk: static
app_file: index.html
---

# Klasifikasi Kelas Harga Laptop Berdasarkan Spesifikasi Teknis Menggunakan Arsitektur Multi-Layer Perceptron (MLP) dengan Hyperparameter Tuning

Aplikasi ini menggunakan **Hugging Face Spaces SDK Static** dengan `index.html`,
`app.js`, dan `inference.js`. Tidak menggunakan server Python, token pribadi,
atau endpoint inferensi berbayar. Bobot Model 3 (MLP hasil KerasTuner) diekspor ke
`model_browser.json`. Statistik imputasi, scaler, dan OneHotEncoder yang dipelajari
**hanya dari data latih** disimpan di `preprocessing_browser.json`.

## Cara menggunakan
1. Buat Hugging Face Space baru dan pilih **Static** / HTML.
2. Unggah **semua berkas** dari ZIP hasil Section 13 ke direktori utama Space.
3. Pastikan README.md memiliki `sdk: static` dan `app_file: index.html`.
4. Buka tab App. Pilih spesifikasi, lalu tekan **Klasifikasikan laptop**.
5. Uji contoh prediksi dan dua fitur: *confidence threshold* serta
   ringkasan *permutation importance* (dari evaluasi validasi notebook).

Seluruh kalkulasi prediksi berjalan di browser. Berkas model dan preprocessing dapat
dibaca publik melalui browser; ini memang karakteristik deployment statis.
Jika dibuka melalui `file://` dan `fetch` ditolak browser, gunakan Hugging Face Space
atau server statis lokal (`python -m http.server 8000`).

## Rentang kelas dual-price (konversi ilustratif)
- Entry-Level: di bawah ₹50.000 (≈ Rp9.627.450)
- Mid-Range: ₹50.000 sampai ₹85.000 (≈ Rp9.627.450 sampai Rp16.366.665)
- High-End: di atas ₹85.000 (≈ Rp16.366.665)

Acuan tetap: **1 INR ≈ Rp192,549** pada **Info per November 2026**. IDR hanya
padanan ilustratif, bukan harga aktual di Indonesia. Target dan kelas menggunakan INR.
Model utama penelitian adalah Model 3 hasil hyperparameter tuning; Model 1/2 pembanding.

## Catatan ilmiah
`model_final.keras` dan `preprocessor.joblib` tetap tersimpan dalam folder
`artifacts/` pada Colab untuk kebutuhan reproduksi Python, tetapi tidak diperlukan
oleh Static Space. Tidak ada skor hasil penelitian yang dihitung ulang pada browser.
Metrik penelitian tetap merujuk output training/validasi/testing dari notebook.
'''
(ARTIFACT_DIR / 'README.md').write_text(readme_static, encoding='utf-8')

# 4. Buat sampel parity-check pada data UJI asli, tanpa memodifikasi model.
#    Bagian browser harus menghasilkan kelas dan probabilitas yang konsisten dengan Keras.
probe_rows = X_test.iloc[:min(12, len(X_test))].copy()
probe_rows = probe_rows[num_features + cat_features]
python_vectors = preprocessor.transform(probe_rows).astype(np.float32)
python_predictions = final_model.predict(python_vectors, verbose=0)
probe_data = json.loads(probe_rows.to_json(orient='records'))
(ARTIFACT_DIR / 'browser_test_inputs.json').write_text(
    json.dumps(probe_data, ensure_ascii=False, indent=2), encoding='utf-8'
)
(ARTIFACT_DIR / 'browser_expected_probs.json').write_text(
    json.dumps(python_predictions.tolist()), encoding='utf-8'
)

# 5. Smoke test Python reload: berkas penelitian tetap dapat dibuka pada proses baru.
reload_prep = joblib.load(prep_path)
reload_model = tf.keras.models.load_model(model_path, compile=False)
orig_x = preprocessor.transform(probe_rows).astype(np.float32)
other_x = reload_prep.transform(probe_rows).astype(np.float32)
orig_p = final_model.predict(orig_x, verbose=0)
other_p = reload_model.predict(other_x, verbose=0)
assert np.allclose(orig_x, other_x, atol=1e-6)
assert np.allclose(orig_p, other_p, atol=1e-5)
print('Uji ulang model Keras dan preprocessor joblib: lulus.')

# 6. Smoke test inferensi browser melalui Node jika tersedia di lingkungan Colab.
#    Pengujian dilakukan pada model hasil training asli, bukan bobot contoh.
static_smoke_script = '''const fs = require('fs');
require('./inference.js');
const read = n => JSON.parse(fs.readFileSync(n, 'utf8'));
const p = read('preprocessing_browser.json');
const m = read('model_browser.json');
const schema = read('feature_schema.json');
const labels = read('class_labels.json');
const inputs = read('browser_test_inputs.json');
const predictions = inputs.map(row => globalThis.LaptopInference.predict(row, p, m, labels, schema).probabilities);
console.log('BROWSER_PROBS_JSON=' + JSON.stringify(predictions));
'''
(ARTIFACT_DIR / 'smoke_static.js').write_text(static_smoke_script, encoding='utf-8')
node_executable = shutil.which('node')
if node_executable:
    proc = subprocess.run([node_executable, 'smoke_static.js'],
                          cwd=str(ARTIFACT_DIR.resolve()),
                          capture_output=True, text=True, timeout=120)
    if proc.returncode != 0:
        raise RuntimeError('Smoke test Static JavaScript gagal:\n' + proc.stderr[-3000:])
    lines = [line for line in proc.stdout.splitlines() if line.startswith('BROWSER_PROBS_JSON=')]
    if not lines:
        raise RuntimeError('Node tidak mencetak prediksi untuk uji Static.')
    browser_probs = np.asarray(json.loads(lines[-1].split('=', 1)[1]), dtype=np.float64)
    max_diff = float(np.max(np.abs(python_predictions - browser_probs)))
    assert browser_probs.shape == python_predictions.shape
    assert np.allclose(browser_probs, python_predictions, atol=2e-3, rtol=1e-3), (
        f'Prediksi browser berbeda terlalu jauh dari Keras (maks: {max_diff}).'
    )
    assert np.array_equal(np.argmax(browser_probs, axis=1), np.argmax(python_predictions, axis=1)), (
        'Label prediksi di browser tidak sama dengan prediksi Keras.'
    )
    print(f'Uji JavaScript vs Keras pada {len(probe_rows)} data uji: LULUS; '
          f'selisih probabilitas maksimum {max_diff:.8f}.')
else:
    print('Node.js tidak tersedia: kode JS lolos ekspor, tetapi perbandingan probabilitas '
          'browser vs Keras perlu dijalankan di lingkungan dengan Node.js atau browser.')

# 7. Paket khusus Hugging Face Static: hanya berkas yang dibutuhkan browser.
static_files = [
    'README.md', 'index.html', 'app.js', 'inference.js',
    'model_browser.json', 'preprocessing_browser.json',
    'feature_schema.json', 'class_labels.json',
    'evaluation_results.json', 'tuning_hyperparameters.json',
]
for filename in static_files:
    assert (ARTIFACT_DIR / filename).is_file(), 'Artefak Static belum tersedia: '+filename
zip_path = Path('HuggingFace_Spaces_Kelompok4_Static_Final.zip')
with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as z:
    for filename in static_files:
        z.write(ARTIFACT_DIR / filename, arcname=filename)

artifact_files_df = pd.DataFrame([
    {'File': name, 'Ukuran (KB)': round((ARTIFACT_DIR / name).stat().st_size / 1024, 2),
     'SHA256': hashlib.sha256((ARTIFACT_DIR / name).read_bytes()).hexdigest()[:20]}
    for name in static_files
])
print('\n=== BERKAS DEPLOYMENT SDK STATIC ===')
display(artifact_files_df)
print('Paket ZIP:', zip_path.resolve(), '| ukuran KB:', round(zip_path.stat().st_size/1024, 2))
print('sdk: static; app_file: index.html. Hosting publik harus diuji setelah upload.')


# ================================================================================
# SECTION 14 — sumber: notebook sel 40
# ================================================================================
# ============================================================================
# SECTION 14 - PETA MATERI W01–W07 DAN BUKTI HASIL UNTUK LAPORAN
# Semua metrik diperoleh dari objek yang dihasilkan kode setelah Run all.
# ============================================================================
from pathlib import Path
import zipfile

# 1) Audit konsistensi data, pelatihan, dan model utama sesuai judul.
assert final_model is tuned_mlp, 'Model utama harus MLP hasil hyperparameter tuning.'
assert final_model_name == 'Model 3: Tuned MLP'
assert NUM_CLASSES == 3 and final_model.output_shape[-1] == 3
assert INPUT_DIM == X_train_scaled.shape[1] == X_val_scaled.shape[1] == X_test_scaled.shape[1]
assert len(X_train) + len(X_val) + len(X_test) == len(df)
assert not (set(X_train.index) & set(X_val.index))
assert not (set(X_train.index) & set(X_test.index))
assert not (set(X_val.index) & set(X_test.index))
assert not {'price', 'price_idr', 'price_juta_idr', 'category'} & set(num_features + cat_features)
assert len(tuner.oracle.trials) > 0 and bool(best_hp.values)
assert np.isfinite(X_train_scaled).all() and np.isfinite(X_val_scaled).all() and np.isfinite(X_test_scaled).all()
assert int(confusion_matrix(y_test, y_pred_final, labels=[0,1,2]).sum()) == len(y_test)
assert len(perm_importance_df) == len(num_features) + len(cat_features)
assert len(threshold_test_df) == 1 and np.isclose(
    float(threshold_test_df.iloc[0]['Threshold dari validasi']), float(OPTIMAL_TAU))
assert np.isclose(
    float(results_df.loc[results_df['Iterasi Model'] == final_model_name, 'Test Accuracy (%)'].iloc[0]),
    100 * accuracy_score(y_test, y_pred_final), atol=1e-8)
if RUN_ABLATION:
    assert len(ablation_study_df) == 6, 'Eksperimen A–E dan pembanding F harus berasal dari hasil pelatihan.'
if RUN_MULTI_SEED:
    assert len(multiseed_detail_df) == len(SEEDS_LIST) * len(model_candidates)
if RUN_GROUP_SENSITIVITY:
    assert len(group_sensitivity_df) == 1

# 2) Tabel sumber materi; ini adalah peta penerapan, bukan metrik penelitian.
materi_rows = [
    ('W01', 'MLP, tensor, tf.data', 'Section 4–7', 'Diterapkan', 'W01 - halaman 27–33; 41–47'),
    ('W02', 'Pemisahan data, shuffle, input pipeline', 'Section 3–4', 'Diterapkan', 'W02 - halaman 6–11; 24–30'),
    ('W03', 'Cross-entropy, overfitting, Dropout, L2', 'Section 5; 7; 8; 12', 'Diterapkan', 'W03 - halaman 9–24'),
    ('W04', 'SGD, Adam/RMSprop, Batch Normalization', 'Section 5–7; 12', 'Diterapkan', 'W04 - halaman 16–35'),
    ('W05', 'RandomSearch KerasTuner, ReduceLROnPlateau, EarlyStopping, Checkpoint',
     'Section 6–8; 12', 'Diterapkan, metode utama', 'W05 - halaman 13–44'),
    ('W06', 'Konvolusi dan pooling pada citra', 'Tidak diterapkan', 'Tidak diperlukan untuk data tabular', 'W06 - halaman 2–5'),
    ('W07', 'Transfer learning CNN untuk citra', 'Tidak diterapkan', 'Tidak diperlukan untuk data tabular', 'W07 - halaman 2; 20–27'),
]
peta_materi_df = pd.DataFrame(
    materi_rows, columns=['Pertemuan', 'Materi', 'Bukti pada notebook', 'Status', 'Halaman PDF'])
print('=== PETA MATERI DEEP LEARNING W01–W07 ===')
display(peta_materi_df)

# 3) Kumpulkan hanya data dan metrik yang dibentuk dalam proses eksperimen ini.
hasil_dir = Path('Hasil_Laporan_Kelompok4')
hasil_dir.mkdir(exist_ok=True)
tables = {
    '01_peta_materi_w01_w07': peta_materi_df,
    '02a_ambang_kelas_inr_idr': kelas_harga_df,
    '02b_statistik_harga_aktual_dual': dual_currency_df,
    '02_distribusi_kelas': df['category'].value_counts().sort_index()
        .rename_axis('Kelas').reset_index(name='Jumlah'),
    '03_missing_values': df[num_features + cat_features].isna().sum()
        .rename_axis('Fitur').reset_index(name='Jumlah NaN'),
    '04_pembagian_data': pd.DataFrame([
        {'Subset':'Training', 'Jumlah':len(X_train)},
        {'Subset':'Validation', 'Jumlah':len(X_val)},
        {'Subset':'Testing', 'Jumlah':len(X_test)}]),
    '05_peringkat_hyperparameter_tuning': trials_summary_df,
    '06_metrik_validasi': val_selection_df,
    '07_metrik_pengujian': results_df,
    '08_perbandingan_model': all_candidates_summary_df,
    '09_threshold_validasi': thresh_val_df,
    '10_threshold_pengujian': threshold_test_df,
    '11_permutation_importance': perm_importance_df,
    '12_ablasi_validasi': ablation_study_df,
    '13_multiseed_detail': multiseed_detail_df,
    '14_multiseed_ringkasan': multiseed_summary_df,
    '15_sensitivitas_group_split': group_sensitivity_df,
    '16_contoh_benar_salah_test': error_analysis_df,
    '16a_simulasi_kelas_harga_dual': simulasi_df,
}
for filename, table in tables.items():
    table.to_csv(hasil_dir / f'{filename}.csv', index=False, encoding='utf-8-sig')

# 4) Grafik evaluasi khusus model utama MLP hasil tuning, bukan hasil sintetis.
fig, ax = plt.subplots(figsize=(8.8, 4.6))
ax.plot(range(1, len(hist_tuned.history['accuracy'])+1),
        np.asarray(hist_tuned.history['accuracy'])*100, label='Training')
ax.plot(range(1, len(hist_tuned.history['val_accuracy'])+1),
        np.asarray(hist_tuned.history['val_accuracy'])*100, label='Validation')
ax.set(title='Akurasi MLP hasil tuning', xlabel='Epoch', ylabel='Akurasi (%)', ylim=(0,100))
ax.legend(); ax.grid(alpha=.25); fig.tight_layout()
fig.savefig(hasil_dir/'17_akurasi_mlp_hasil_tuning.png',dpi=180);plt.show()

fig, ax = plt.subplots(figsize=(8.8, 4.6))
ax.plot(range(1,len(hist_tuned.history['loss'])+1), hist_tuned.history['loss'],label='Training')
ax.plot(range(1,len(hist_tuned.history['val_loss'])+1), hist_tuned.history['val_loss'],label='Validation')
ax.set(title='Loss MLP hasil tuning',xlabel='Epoch',ylabel='Loss')
ax.legend();ax.grid(alpha=.25);fig.tight_layout()
fig.savefig(hasil_dir/'18_loss_mlp_hasil_tuning.png',dpi=180);plt.show()

fig, ax = plt.subplots(figsize=(6.4,5.2))
sns.heatmap(confusion_matrix(y_test,y_pred_final,labels=[0,1,2]), annot=True,fmt='d',
    cmap='Blues', xticklabels=label_names, yticklabels=label_names,ax=ax)
ax.set(title='Confusion matrix - MLP hasil tuning',xlabel='Prediksi',ylabel='Aktual')
fig.tight_layout();fig.savefig(hasil_dir/'19_confusion_matrix_mlp_hasil_tuning.png',dpi=180);plt.show()

fig, ax = plt.subplots(figsize=(8.8,4.8))
plot_rank=perm_importance_df.sort_values('Penurunan F1 Rata-rata')
ax.barh(plot_rank['Nama Fitur Asli'],plot_rank['Penurunan F1 Rata-rata'],
        xerr=plot_rank['Simpangan Baku'])
ax.set(xlabel='Penurunan F1 makro validasi',title='Permutation importance (10 pengacakan/fitur)')
ax.grid(axis='x',alpha=.25);fig.tight_layout()
fig.savefig(hasil_dir/'20_permutation_importance.png',dpi=180);plt.show()

fig, ax = plt.subplots(figsize=(8.5,4.6))
ax.plot(thresh_val_df['Threshold'],thresh_val_df['Coverage'],'o-',label='Coverage')
ax.plot(thresh_val_df['Threshold'],thresh_val_df['Accepted Macro F1'],'s-',label='Accepted F1')
ax.axvline(OPTIMAL_TAU,color='black',linestyle='--',label=f'Ambang={OPTIMAL_TAU:.2f}')
ax.set(xlabel='Confidence threshold',ylabel='Proporsi/skor',title='Kalibrasi ambang pada validasi',ylim=(0,1))
ax.legend();ax.grid(alpha=.25);fig.tight_layout()
fig.savefig(hasil_dir/'21_threshold_validasi.png',dpi=180);plt.show()

# 5) Ringkasan yang dapat disalin ke laporan, seluruh angka diperoleh dari model/target nyata.
final_metric = results_df.loc[results_df['Iterasi Model'] == final_model_name].iloc[0]
ringkasan = f"""# Ringkasan hasil notebook Kelompok 4 (dibuat otomatis setelah Run all)

Judul: {feature_schema['project_title']}
Dataset: {len(df)} baris data setelah proses ekstraksi yang digunakan.
Pembagian data: training {len(X_train)}, validasi {len(X_val)}, testing {len(X_test)}.
Dimensi fitur model: {INPUT_DIM}; kelas keluaran: {NUM_CLASSES}.
Metode utama: MLP hasil KerasTuner RandomSearch ({len(tuner.oracle.trials)} trial).
Hyperparameter terbaik: {best_hp.values}.
Model utama: {final_model_name} (jumlah parameter {final_model.count_params()}).
Akurasi testing: {final_metric['Test Accuracy (%)']:.4f}%.
F1 makro testing: {final_metric['Test Macro F1-Score']:.6f}.
Jumlah kesalahan testing: {int(final_metric['Jumlah Salah Uji'])} dari {len(y_test)} sampel.
Ambang confidence validasi: {OPTIMAL_TAU:.2f}.

Sumber angka: objek hasil prediksi, evaluasi, dan pelatihan dalam notebook ini; bukan hasil ketikan manual.
Harga / nilai tukar: 1 INR ≈ Rp{str(KURS_IDR).replace('.', ',')} (acuan {TANGGAL_KURS}; hanya ilustrasi).
Ambang kelas harga asli beserta ekuivalen IDR:
- Entry-Level: {rentang_kelas_dual(0)}
- Mid-Range: {rentang_kelas_dual(1)}
- High-End: {rentang_kelas_dual(2)}
Harga aktual per kelas dan harga setiap contoh uji terdapat pada tabel CSV ekspor dual-price.
Semua kelas ditentukan dari INR asli; model tidak memperkirakan harga nominal atau harga pasar Indonesia.
Materi yang diterapkan: W01, W02, W03, W04, W05; W06/W07 tidak digunakan karena fokusnya CNN/citra.
Hasil sensitivitas group split bukan benchmark independen untuk memilih hyperparameter.
"""
(hasil_dir/'00_ringkasan_hasil_otomatis.md').write_text(ringkasan,encoding='utf-8')
zip_path=Path('Hasil_Laporan_Kelompok4.zip')
with zipfile.ZipFile(zip_path,'w',zipfile.ZIP_DEFLATED) as zipf:
    for path in sorted(hasil_dir.iterdir()):
        if path.is_file():
            zipf.write(path,arcname=path.name)
print('=== AUDIT AKHIR LULUS: MODEL, INPUT, HASIL EVALUASI DAN SUMBER BUKTI ===')
print(ringkasan)
print(f'File bukti penelitian tersimpan: {zip_path.resolve()}')
print('Catatan: kelulusan audit ini tidak dengan sendirinya membuktikan hosting Hugging Face sudah aktif.')

