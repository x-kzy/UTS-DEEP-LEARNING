# Mengapa Memilih Hugging Face Spaces dan Bagaimana Deploy

[README](../README.md) · [Source frontend](WEB_SOURCE.md) · [Bukti website](HUGGING_FACE.md) · [LAPTOP-CLASSIFIER](https://huggingface.co/spaces/Xkazuya/LAPTOP-CLASSIFIER)

## Alasan memilih Hugging Face Spaces SDK Static

Kami memerlukan sarana agar model hasil eksperimen dapat dicoba secara publik tanpa mengharuskan dosen menyiapkan Python, membuka Colab, atau melatih MLP ulang. Hugging Face Spaces menyediakan halaman aplikasi dan repository khusus untuk setiap Space, sehingga UI, source, dan status deployment dapat diperiksa dari satu alamat.

SDK **Static** dipilih karena arsitektur inferensi proyek sudah diekspor ke JavaScript. Penghitungan dilakukan oleh browser, bukan API Gradio atau server Python. Manfaatnya adalah alur penggunaan yang sederhana, tidak memerlukan GPU server untuk setiap prediksi, dan cukup mengunggah HTML/CSS/JS beserta JSON model. Trade-off: bobot dikirim ke browser pengguna, performa bergantung perangkat, dan seluruh pemrosesan browser harus diuji terhadap hasil TensorFlow asli.

## Langkah deployment yang digunakan sebagai acuan

1. **Latih dan evaluasi model di Google Colab**, pilih Model 3 Tuned MLP 96→64→48 yang sesuai judul, lalu pastikan preprocessing dipelajari dari training.
2. **Ekspor bobot dan preprocessing asli**, termasuk nama/urutan fitur, statistik StandardScaler, kategori OneHotEncoder, statistik imputasi, label kelas, dan ambang confidence. Jangan mengisi JSON dengan angka dummy.
3. **Jalankan pemeriksaan kesesuaian** antara beberapa probabilitas model Keras dan forward-pass JavaScript. Jika tidak lulus, jangan menyebut deployment terverifikasi.
4. **Siapkan berkas root Space**, sekurang-kurangnya `README.md` dengan metadata `sdk: static` dan `app_file: index.html`, kemudian `index.html`, `style.css`, `app.js`, `inference.js`, `xlsx-lite.js`, serta artefak JSON dari langkah dua.
5. **Buat Space bernama LAPTOP-CLASSIFIER** di akun Hugging Face, pilih template **Blank** dan **SDK Static**, lalu unggah isi folder aplikasi melalui Files → Contribute → Upload files → Commit changes. File ZIP tidak diunggah sebagai satu berkas.
6. **Buka tab App**, periksa status *Model 3 Tuned MLP siap · 83 fitur · inferensi di browser*.
7. **Uji input kustom** (dropdown, radio, slider dan preset), pergantian tema, informasi INR/IDR, probabilitas dan status *Perlu Ditinjau*.
8. **Uji tiga format batch**: CSV 18, XLSX 40, JSON 72. Baca jumlah kelas pada antarmuka dan periksa file hasil ekspor.
9. **Simpan screenshot dan catatan hasil demonstrasi**; screenshot yang disertakan dalam repository berasal dari hasil pengguna menjalankan Space, bukan hasil pelatihan baru di GitHub.

## Susunan berkas pada Space produksi

```text
LAPTOP-CLASSIFIER (root Space)
├── README.md                 # YAML SDK Static dan metadata
├── index.html                # UI terbaru
├── style.css                 # Light/Dark dan responsif
├── app.js                    # Form, batch, render hasil
├── inference.js              # Inferensi dengan bobot model
├── xlsx-lite.js              # Import/export XLSX
├── model_browser.json        # Bobot MLP ekspor Colab (tidak disimpan di ZIP GitHub ini)
├── preprocessing_browser.json
├── feature_schema.json
├── class_labels.json
├── evaluation_results.json
├── presets.json
├── catalog_presets.json
└── model_parity_reference.json  # Bila diproduksi saat ekspor
```

Berkas JSON hasil training pada struktur di atas **tidak disertakan dalam repository GitHub dokumentasi ini** karena tidak tersedia sebagai unggahan yang dapat diverifikasi. Untuk mencocokkan versi produksi, kunjungi [Files di Space](https://huggingface.co/spaces/Xkazuya/LAPTOP-CLASSIFIER/tree/main). Tidak perlu mengunggah `.ipynb` ke Space sebagai aplikasi.

## Pengujian sesudah deploy

| Area | Bukti yang disimpan | Interpretasi |
| --- | --- | --- |
| Input kustom Budget gelap | [Screenshot](huggingface/kustom_budget_dark_9859.png) | Entry-Level, confidence 98,59%; contoh inferensi |
| Input kustom High-End terang | [Screenshot](huggingface/kustom_highend_light_9997.png) | High-End, confidence 99,97%; contoh inferensi |
| Preset training 48 laptop | [Screenshot](huggingface/batch_preset_training_48_light.png) | Distribusi 17/17/14, **bukan accuracy** |
| CSV 18 skenario | [Screenshot](huggingface/batch_csv_18.png) | Distribusi 0/13/5 |
| XLSX 40 skenario | [Screenshot](huggingface/batch_xlsx_40.png) | Distribusi 2/16/22 |
| JSON 72 skenario | [Screenshot](huggingface/batch_json_72_dark_667.png) | Distribusi 4/20/48 |

## Batas klaim

Bukti tampilan web tidak dengan sendirinya membuktikan kecocokan floating-point terhadap TensorFlow. Notebook sumber menyiapkan pemeriksaan *parity*, tetapi log sukses pemeriksaan dari deployment aktual **tidak tercatat di lampiran yang tersedia**. Prediksi online yang terlihat pada screenshot digunakan hanya untuk menjelaskan demo, bukan menggantikan evaluasi 153 sampel test.
