# Membaca dan Menjalankan Ulang Eksperimen

## Cara tercepat: hasil tanpa melatih ulang

1. Buka [notebook final](../notebooks/UTS_DEEP_LEARNING_KELOMPOK_4_FINAL.ipynb) untuk melihat 14 section beserta output tersimpan.
2. Buka [galeri](GALLERY.md), [tabel output](TABLE_INDEX.md), dan [ringkasan hasil](EXPERIMENTS.md) untuk meninjau metrik yang tercatat.
3. Buka laporan [PDF](../reports/Laporan_UTS_Deep_Learning_Kelompok_4.pdf) atau [DOCX](../reports/Laporan_UTS_Deep_Learning_Kelompok_4.docx) untuk pembahasan akademis.

## Menjalankan kembali pada Google Colab

1. Unggah notebook `.ipynb` dan pilih runtime dengan Python serta TensorFlow.
2. Jalankan **Runtime → Run all** dari Section 1 ke 14. Dataset diunduh menggunakan `kagglehub` bila file CSV belum ada.
3. Jika Kaggle tidak tersedia, tempatkan `laptops_uncleaned.csv` dan `laptops_cleaned_v1.csv` di direktori kerja sebelum menjalankan Section 1.
4. Setelah pelatihan, periksa bahwa model utama tetap Model 3, input tensor 83 fitur, dan callback serta threshold dibangun dari validation.
5. Section 13 bertanggung jawab mengekspor model untuk SDK Static. File bobot hasil pelatihan **tidak disertakan** dalam repository GitHub ini. Jalankan ekspor untuk menghasilkan artefak yang cocok dengan model runtime terbaru.

## Perbedaan yang mungkin terjadi

GPU/CPU, versi library, urutan floating-point, dan variasi pelatihan dapat menghasilkan sedikit perbedaan meskipun seed sama. `requirements.txt` adalah panduan dependensi, bukan salinan lingkungan Colab yang telah diverifikasi ulang pada mesin ini. Grafik dan tabel arsip di repository adalah keluaran tersimpan dari notebook unggahan, bukan hasil eksekusi baru.

## Pemeriksaan integrity paket

Dari root folder repository yang diekstrak jalankan:

```bash
python scripts/verify_repository.py
```

Pemeriksaan ini menguji keberadaan file, angka dasar pada metadata, jumlah skenario 18/40/72, format gambar, tautan lokal Markdown, dan konsistensi tabel yang tersedia. Ini **bukan** pengganti menjalankan TensorFlow dari awal.
