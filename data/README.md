# Dataset dan pengujian input

Sumber penelitian: [Laptop Specifications and Prices Dataset](https://www.kaggle.com/datasets/anshsahu111/laptop-specifications-and-prices-dataset), oleh `anshsahu111`.

Notebook mengunduh `laptops_uncleaned.csv` (1.020 baris × 12 kolom) dan `laptops_cleaned_v1.csv` (1.018 baris × 22 kolom) melalui `kagglehub`. **Dataset Kaggle penuh tidak disalin** ke repository ini karena sumber publik tetap menjadi acuan atribusi dan versi. Dataset atau ketentuan lisensinya perlu diperiksa langsung pada Kaggle.

Folder `testing/` berisi **input simulasi**, bukan sampel uji berlabel pada eksperimen 153 data. File CSV/XLSX/JSON mempunyai jumlah dan isi berbeda: 18, 40, dan 72 laptop. Model menghitung label prediksi, sehingga jangan menyebut kelas skenario buatan sebagai label sebenarnya.

Untuk menguji: buka [LAPTOP-CLASSIFIER](https://huggingface.co/spaces/Xkazuya/LAPTOP-CLASSIFIER), pilih **Batch Upload File**, unggah satu file, klik **Jalankan Prediksi Massal**, lalu unduh hasilnya bila perlu.

Berkas: [`batch_uji_18_laptop.csv`](testing/batch_uji_18_laptop.csv) · [`batch_uji_40_laptop.xlsx`](testing/batch_uji_40_laptop.xlsx) · [`batch_uji_72_laptop.json`](testing/batch_uji_72_laptop.json).

## Perbedaan preset training dan file simulasi

Screenshot 48 laptop berasal dari tombol **Preset Katalog (Data Latih)** pada aplikasi dan bukan dari ketiga file simulasi. File CSV 18, XLSX 40, dan JSON 72 adalah **skenario pengujian sintetis** yang disertakan untuk mendemokan tab batch; distribusi prediksinya ada pada [dokumentasi Hugging Face](../docs/HUGGING_FACE.md). Pada template upload, 18 input spesifikasi ditambah identitas opsional `model`, sehingga berkas demo memuat 19 kolom; fitur PPI dihitung otomatis.
