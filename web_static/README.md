# Source Antarmuka LAPTOP-CLASSIFIER (SDK Static)

[README utama](../README.md) · [Bukti demo](../docs/HUGGING_FACE.md) · [Penjelasan berkas frontend](../docs/WEB_SOURCE.md) · [Space publik](https://huggingface.co/spaces/Xkazuya/LAPTOP-CLASSIFIER)

Folder ini memuat kode yang mendasari antarmuka Hugging Face **tanpa Gradio**. `index.html` dan `app.js` disalin **persis dari file terbaru yang diunggah tim**, yaitu `index(3).html` dan `app(1).js`. Keduanya juga cocok secara checksum dengan ZIP pembaruan footer dan sumber output yang dilampirkan. Jadi, perubahan **“UTS ● DEEP LEARNING ● KELOMPOK 4 ● LAPTOP CLASSIFIER”** dan **“Sumber output: Notebook UTS”** sudah ada.

Tiga berkas pendamping (`style.css`, `inference.js`, `xlsx-lite.js`) berasal dari paket sumber statis yang lebih dahulu disiapkan untuk proyek ini. **Kami tidak dapat mengunduh snapshot commit Space publik pada sesi penyusunan**; karena itu, kami tidak menyatakan bahwa kelima file ini identik byte-for-byte dengan versi yang *sedang online*. Jika Space telah mengalami perubahan CSS/inferensi setelah paket awal, ambil versi pendamping terbaru dari tab [Files di Hugging Face](https://huggingface.co/spaces/Xkazuya/LAPTOP-CLASSIFIER/tree/main) sebelum memakai folder ini untuk redeploy.

| Berkas | Fungsi |
| --- | --- |
| [`index.html`](index.html) | Layout dua tab, dropdown/radio/slider, kartu prediksi, ringkasan batch, panel penelitian, footer |
| [`app.js`](app.js) | Penghubung UI, pemuatan JSON, preset, inferensi kustom/batch, parsing CSV/JSON/XLSX, penghitungan PPI, ekspor hasil, tema |
| [`inference.js`](inference.js) | Transformasi fitur dan perhitungan forward MLP di browser dengan bobot yang diekspor |
| [`style.css`](style.css) | Tampilan responsif, warna light/dark, grid, tabel, tombol |
| [`xlsx-lite.js`](xlsx-lite.js) | Pembaca dan penulis XLSX yang berjalan pada browser tanpa backend |

**Bukan paket deployment produksi yang lengkap:** file bobot dan parameter asli seperti `model_browser.json`, `preprocessing_browser.json`, `feature_schema.json`, `class_labels.json`, dan referensi parity tidak tersedia di lampiran. Berkas-berkas tersebut sengaja **tidak diganti dengan model contoh atau data rekaan**. Aplikasi publik memuat artefaknya secara terpisah melalui Space. Untuk pemakaian frontend lokal dengan prediksi sungguhan, unduh artefak produksi asli dari Space milik kelompok.

Teks asli yang kamu kirim pada file `app.js` dan `index.html` tidak diubah saat dimasukkan ke repository ini.
