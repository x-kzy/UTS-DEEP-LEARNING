# Peta Section, Materi Kuliah, dan Bukti

[Kembali ke README](../README.md) · [Galeri gambar](GALLERY.md) · [Hasil eksperimen](EXPERIMENTS.md)

Notebook memuat **14 section**, berurutan dari pemuatan dataset hingga ringkasan penelitian. Tabel ini membantu pembaca menuju kode, grafik, dan penjelasan setiap tahap tanpa membuka seluruh notebook terlebih dahulu.

| Notebook | Peran dalam alur | Keterkaitan materi |
| --- | --- | --- |
| [Section 01](stages/01.md) | Lingkungan dan akuisisi dataset | W01 |
| [Section 02](stages/02.md) | EDA, audit kualitas, dan pembentukan label | W02 |
| [Section 03](stages/03.md) | Stratified split dan preprocessing train-only | W02, W05 |
| [Section 04](stages/04.md) | TensorFlow tf.data dan representasi tensor | W01–W02 |
| [Section 05](stages/05.md) | Dua model pembanding: baseline dan regularized MLP | W01–W04 |
| [Section 06](stages/06.md) | Hyperparameter tuning dengan KerasTuner | W05 |
| [Section 07](stages/07.md) | Retraining model utama dan callback | W04–W05 |
| [Section 08](stages/08.md) | Evaluasi data test dan contoh prediksi | W03–W05 |
| [Section 09](stages/09.md) | Audit baseline sebelum optimasi | W02–W04 |
| [Section 10](stages/10.md) | Ringkasan perbandingan model | W03–W05 |
| [Section 11](stages/11.md) | Dua fitur unik: confidence threshold dan permutation importance | W03–W05 |
| [Section 12](stages/12.md) | Ablasi, lima seed, dan sensitivitas generalisasi | W03–W05 |
| [Section 13](stages/13.md) | Ekspor dan rancangan aplikasi Hugging Face SDK Static | Implementasi proyek |
| [Section 14](stages/14.md) | Peta materi kuliah dan simpulan yang tercatat | W01–W07 (evaluasi kesesuaian) |

Metode **MLP dengan KerasTuner** selaras dengan karakteristik data tabular. Materi CNN dan transfer learning W06–W07 dibahas dalam pemetaan materi, tetapi tidak dipaksakan ke model laptop berbasis tabel. Status bukti per stage tersedia langsung di halaman masing-masing.


## Dokumentasi terbaru website

[Bukti Light/Dark, input kustom, batch 48/18/40/72](HUGGING_FACE.md) · [Source UI terbaru](WEB_SOURCE.md) · [Alasan serta tahap deployment](DEPLOYMENT.md).
