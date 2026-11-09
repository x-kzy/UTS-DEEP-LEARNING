# Keterbatasan dan Rencana Penyempurnaan

Poin yang perlu dipertimbangkan setelah UTS:

1. **Model 2 lebih tinggi dari Model 3 pada testing.** Model utama tetap Model 3 untuk menguji metode hyperparameter tuning pada judul; jangan mengklaim tuning menghasilkan akurasi tertinggi.
2. **Kemiripan produk.** Duplikasi spesifikasi dan nama laptop antar-split acak dapat membuat metrik terlalu optimistis untuk produk yang benar-benar baru. Pengujian group-disjoint membantu menilai sensitivitas, tetapi belum menghilangkan seluruh risiko.
3. **Kelas Mid-Range.** Recall 40/56 (~71,43%) menunjukkan kesulitan membedakan laptop kelas menengah dari dua kelas lainnya.
4. **GPU belum per-seri.** Input memakai `gpu_brand` dan `gpu_type`, belum nomor RTX, GTX, atau GPU individual. Dropdown tambahan yang tidak ada pada fitur pelatihan tidak boleh dianggap mengubah inferensi.
5. **Confidence bukan probabilitas kebenaran terkalibrasi.** Ambang 0,70 menandai input yang perlu ditinjau, tetapi probabilitas softmax sendiri belum membuktikan kalibrasi pada semua domain.
6. **Harga IDR hanya padanan ilustratif.** Label asli adalah INR; model bukan kalkulator harga jual pasar Indonesia.
7. **Domain dataset.** Harga, merek dan spesifikasi berasal dari satu katalog; perubahan harga dan tipe perangkat baru perlu evaluasi ulang.
8. **Rilis frontend.** Screenshot publik membuktikan alur UI pada tanggal yang ditunjukkan, tetapi integritas versi bobot dan kesetaraan keluaran JavaScript–Keras perlu diperiksa ulang setiap kali model diperbarui.

Pengembangan lanjutan yang relevan adalah feature engineering GPU lebih detail, evaluasi pada waktu/produk baru, validasi kalibrasi probabilitas, perluasan data, dan rangkaian uji otomatis frontend.
