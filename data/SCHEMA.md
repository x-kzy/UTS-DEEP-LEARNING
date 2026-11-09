# Atribut Input Model

Notebook memakai **19 atribut input** yang ditransformasi menjadi **83 dimensi float32**. Kolom harga, ekuivalen IDR, dan identitas model **tidak** diberikan kepada MLP.

| Jenis | Kolom | Pengolahan |
| --- | --- | --- |
| Numerik (10) | `rating`, `threads_num`, `core_num`, `ram_num`, `memory_size`, `display_size`, `resolution_height`, `resolution_width`, `ppi`, `warranty` | `GroupMedianImputer` untuk core/thread bila kosong, lalu `StandardScaler` |
| Kategorikal (9) | `brand_name`, `processor_brand`, `iter`, `ram_type`, `memory_type`, `gpu_brand`, `gpu_type`, `os`, `Touch_Screen` | `OneHotEncoder(handle_unknown="ignore")` |
| Target | `price_class` (0/1/2), dibuat dari `price` (INR) | Keluaran tiga probabilitas `Softmax` |
| Audit saja | `model`, `price`, `price_idr` | Bukan fitur prediksi |

`ppi` dihitung dari resolusi dan diagonal layar. Pilihan kategori di aplikasi idealnya bersumber dari encoder yang telah dilatih; termasuk `gpu_brand` dan `gpu_type` namun **belum** membedakan RTX/GTX atau nomor seri GPU.

**Batas harga:** 0 = < ₹50.000; 1 = ₹50.000–₹85.000; 2 = > ₹85.000. Konversi IDR hanya konteks: 1 INR ≈ Rp192,549 (acuan tetap 9 November 2026).
