import requests
import datetime
import pandas as pd
import matplotlib.pyplot as plt
from pyspark.sql import SparkSession
from pyspark.sql.types import StructType, StructField, StringType, DoubleType
from pyspark.sql.functions import year, col, avg, round

# --- Mulai Script Utama ---

# 2. Inisialisasi Apache Spark Session (Wajib untuk memulai Spark)
print("Memulai Apache Spark...")
spark = SparkSession.builder \
    .appName("Proyek1_API_ke_Spark") \
    .master("local[*]") \
    .getOrCreate()

# 3. Tarik Data dari API Open-Meteo
# Mengambil data dari 2015 hingga hari ini (dikurangi 5 hari untuk archive API)
end_date_str = (datetime.date.today() - datetime.timedelta(days=5)).strftime("%Y-%m-%d")
url_api = f"https://archive-api.open-meteo.com/v1/archive?latitude=-5.1476&longitude=119.4327&start_date=2015-01-01&end_date={end_date_str}&daily=temperature_2m_max,temperature_2m_min,precipitation_sum&timezone=Asia%2FMakassar"

print(f"Menarik data API dari 2015-01-01 hingga {end_date_str}...")
response = requests.get(url_api)

if response.status_code == 200:
    data_json = response.json()
    
    # 4. Preprocessing Data Mentah
    daily = data_json['daily']
    
    # Menyusun data menjadi baris-baris (List of Tsuples)qdw
    data_list = []
    for i in range(len(daily['time'])):
         # Menangani nilai null jika ada data sensor yang kosong dari API
        t_max = float(daily['temperature_2m_max'][i]) if daily['temperature_2m_max'][i] is not None else 0.0
        t_min = float(daily['temperature_2m_min'][i]) if daily['temperature_2m_min'][i] is not None else 0.0
        precip = float(daily['precipitation_sum'][i]) if daily['precipitation_sum'][i] is not None else 0.0
        
        data_list.append((daily['time'][i], t_max, t_min, precip, "Makassar"))
        
    # 5. Definisikan Skema untuk Spark (Wajib di Big Data agar tipe data ketat)
    schema = StructType([
        StructField("time", StringType(), True),
        StructField("temperature_2m_max", DoubleType(), True),
        StructField("temperature_2m_min", DoubleType(), True),
        StructField("precipitation_sum", DoubleType(), True),
        StructField("lokasi", StringType(), True)
    ])
    
    # 6. Buat SPARK DATAFRAME
    print("Membuat Spark DataFrame...")
    spark_df = spark.createDataFrame(data_list, schema)
    
    print(f"Total baris data mentah dalam Spark: {spark_df.count()} baris")
    
    # --- PROSES AGREGASI (Menghitung Rata-rata) ---
    print("Menghitung rata-rata tahunan...")
    # Ekstrak Tahun dari kolom time
    spark_df = spark_df.withColumn("Tahun", year(col("time")))
    
    # Group By Tahun dan hitung rata-rata
    rata_rata_df = spark_df.groupBy("Tahun").agg(
        round(avg("temperature_2m_max"), 2).alias("Rata_rata_TMax"),
        round(avg("temperature_2m_min"), 2).alias("Rata_rata_TMin"),
        round(avg("precipitation_sum"), 2).alias("Rata_rata_Curah_Hujan")
    ).orderBy("Tahun")
    
    print("=== HASIL RATA-RATA TAHUNAN ===")
    rata_rata_df.show(30)
    
    # --- VISUALISASI GRAFIK ---
    print("Membuat grafik...")
    # Ubah ke Pandas khusus untuk visualisasi
    pandas_df = rata_rata_df.toPandas()
    
    # Buat figure dasar dengan subplots agar bisa pakai 2 sumbu-Y (Twin Axes)
    fig, ax1 = plt.subplots(figsize=(12, 7))
    
    # Sumbu Y Pertama (KIRI) untuk Suhu (Line Chart)
    ax1.plot(pandas_df["Tahun"], pandas_df["Rata_rata_TMax"], marker='o', color='red', label='Rata-rata Suhu Max (°C)', linewidth=2, zorder=3)
    ax1.plot(pandas_df["Tahun"], pandas_df["Rata_rata_TMin"], marker='o', color='blue', label='Rata-rata Suhu Min (°C)', linewidth=2, zorder=3)
    ax1.set_xlabel('Tahun', fontsize=12)
    ax1.set_ylabel('Suhu (°C)', fontsize=12, color='black')
    ax1.tick_params(axis='y', labelcolor='black')
    ax1.set_xticks(pandas_df["Tahun"])
    ax1.grid(True, linestyle='--', alpha=0.5, zorder=0)
    
    # Sumbu Y Kedua (KANAN) untuk Curah Hujan (Bar Chart)
    ax2 = ax1.twinx()
    ax2.bar(pandas_df["Tahun"], pandas_df["Rata_rata_Curah_Hujan"], color='teal', alpha=0.3, label='Rata-rata Curah Hujan (mm)', zorder=2)
    ax2.set_ylabel('Curah Hujan (mm)', fontsize=12, color='teal')
    ax2.tick_params(axis='y', labelcolor='teal')
    
    # Memberi ruang lega di atas bar chart agar tidak tumpang tindih dengan garis suhu
    ax2.set_ylim(0, pandas_df["Rata_rata_Curah_Hujan"].max() * 3) 
    
    # Menyatukan legend dari ax1 dan ax2
    lines_1, labels_1 = ax1.get_legend_handles_labels()
    lines_2, labels_2 = ax2.get_legend_handles_labels()
    ax1.legend(lines_1 + lines_2, labels_1 + labels_2, loc='upper left')
    
    plt.title('Rata-rata Suhu dan Curah Hujan Tahunan di Makassar (2015 - Sekarang)', fontsize=14, pad=15)
    
    # Simpan Gambar Grafik
    nama_gambar = "Grafik_Suhu_Makassar.png"
    plt.savefig(nama_gambar, dpi=300, bbox_inches='tight')
    print(f"Grafik berhasil disimpan sebagai: {nama_gambar}")
    
    # 7. Simpan output DataFrame Spark ke format CSV 
    rata_rata_df.write.csv("Dataset_RataRata_Tahunan", header=True, mode="overwrite")
    spark_df.write.csv("Dataset_Mentah_Keseluruhan", header=True, mode="overwrite")
    print("SUCCESS: Data berhasil disimpan oleh Apache Spark!")

else:
    print(f"Gagal menarik data dari API. Status Code: {response.status_code}")