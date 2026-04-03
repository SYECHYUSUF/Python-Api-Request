# Analisis IHSG & Geopolitik menggunakan Apache Spark 📈

Proyek ini bertujuan untuk menarik data Indeks Harga Saham Gabungan (IHSG - `^JKSE`) dari Yahoo Finance API, memprosesnya menggunakan teknologi **Apache Spark**, dan melakukan komparasi pergerakan harga terkait transisi fase geopolitik (Sebelum Perang vs Era Perang mulai April 2024 hingga saat ini).

## 🚀 Fitur Utama
- **Ekstraksi Data API:** Menarik data harian perdagangan IHSG sejak 1 Januari 2023 secara terkini via `yfinance`.
- **Pemrosesan Data dengan PySpark:** Pengolahan dataset dalam jumlah besar secara efisien, melakukan agregasi bulanan, dan iterasi window function untuk mendapat nilai pertumbuhan MoM (Month-over-Month).
- **Analisis Sentimen Geopolitik:** Melabeli dan memfilter tren harga berdasarkan fase "Sebelum Perang" serta "Era Perang".
- **Prediksi Kedepan (Forecasting):** Memberikan estimasi angka penutupan IHSG untuk 3 bulan ke depan bersumber dari rata-rata tren penurunan/kenaikan era perang.
- **Visualisasi Grafik:** Pembuatan plot secara otomatis menggunakan matplotlib dengan penanda visual bergesernya rezim ekonomi (`Grafik_IHSG_Geopolitik.png`).
- **Ekspor CSV:** Hasil bersih siap pakai akan diekspor dalam format csv.

## 🛠️ Persyaratan (Prerequisites)
Pastikan Python 3 sudah terpasang, lalu unduh library yang dibutuhkan dengan menjalankan:
```bash
pip install pyspark yfinance pandas matplotlib
```

## 📖 Cara Menjalankan (How to Run)
1. Buka terminal atau Command Prompt.
2. Navigasi menuju folder direktori proyek ini.
3. Jalankan script analisa:
```bash
python tugas_spark.py
```
*(Gunakan perintah `python3 tugas_spark.py` jika default OS Anda menggunakan alias python3).*

## 📂 Struktur Output File
Setelah script berjalan sukses, program akan mendaur bentuk laporan berupa:
- `Dataset_IHSG_Geopolitik_Final/`: Direktori output Spark yang berisi file berekstensi `.csv` hasil ringkasan pergerakan IHSG. Sangat berguna jika data ini ingin dilanjutkan pada *tools* BI (Business Intelligence).
- `Grafik_IHSG_Geopolitik.png`: Gambar grafik linear komparasi rata-rata IHSG bulanan sebelum era eskalasi dan sesudah zona perang geopolitik.

## 📝 Konteks Proyek
Modul dikerjakan sebagai bagian dari Tugas / Praktikum Mata Kuliah Big Data (Semester 4), menargetkan cara mengkolaborasikan sumber data interaktif (API), pemrosesan terdistribusi (Spark), dan penarikan wawasan berbasis anomali global.
