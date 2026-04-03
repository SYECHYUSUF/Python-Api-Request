import yfinance as yf
import datetime
import pandas as pd
import matplotlib.pyplot as plt
from pyspark.sql import SparkSession
from pyspark.sql.types import StructType, StructField, StringType, DoubleType, LongType
from pyspark.sql.functions import col, round, avg, month, year, lag, when, format_string, format_number
from pyspark.sql.window import Window

# =====================================================================
# 1. INISIALISASI APACHE SPARK
# =====================================================================
print("=====================================================")
print("Memulai Apache Spark & Menyiapkan Environment...")
spark = SparkSession.builder \
    .appName("Proyek1_IHSG_Geopolitik_Final") \
    .master("local[*]") \
    .getOrCreate()

spark.sparkContext.setLogLevel("ERROR")

# =====================================================================
# 2. TARIK DATA DARI YAHOO FINANCE API
# =====================================================================
ticker_symbol = "^JKSE" # Kode Resmi IHSG
end_date = datetime.date.today()
start_date = "2023-01-01" # Ditarik dari awal 2023 untuk perbandingan sebelum perang

print(f"Menarik Data Riil Bursa (IHSG) dari {start_date} hingga {end_date}...")
ihsg = yf.Ticker(ticker_symbol)
df_pandas = ihsg.history(start=start_date, end=end_date)

if not df_pandas.empty:
    
    # =====================================================================
    # 3. PREPROCESSING DATA MENTAH
    # =====================================================================
    df_pandas.reset_index(inplace=True)
    df_pandas['Date'] = df_pandas['Date'].dt.strftime('%Y-%m-%d')
    df_pandas = df_pandas[['Date', 'Open', 'High', 'Low', 'Close', 'Volume']].dropna()
    data_list = list(df_pandas.itertuples(index=False, name=None))
    
    schema = StructType([
        StructField("Tanggal", StringType(), True),
        StructField("Open", DoubleType(), True),
        StructField("High", DoubleType(), True),
        StructField("Low", DoubleType(), True),
        StructField("Close", DoubleType(), True),
        StructField("Volume", LongType(), True)
    ])
    
    spark_df = spark.createDataFrame(data_list, schema)
    
    # =====================================================================
    # 4. AGREGASI BULANAN & PENGHITUNGAN TREN (MoM)
    # =====================================================================
    print("Memproses agregasi Big Data dan Analisis Tren...")
    spark_df = spark_df.withColumn("Tahun", year(col("Tanggal"))) \
                       .withColumn("Bulan", month(col("Tanggal")))
    
    # Menghitung harga tutup rata-rata per bulan
    rata_bulanan = spark_df.groupBy("Tahun", "Bulan").agg(
        round(avg("Close"), 2).alias("Harga_Tutup_Rata2")
    ).orderBy("Tahun", "Bulan")
    
    # Mencari harga bulan lalu menggunakan Window Function
    windowSpec = Window.orderBy("Tahun", "Bulan")
    rata_bulanan = rata_bulanan.withColumn("Harga_Bulan_Lalu", lag("Harga_Tutup_Rata2", 1).over(windowSpec))
    
    # Menghitung persentase pertumbuhan (%)
    rata_bulanan = rata_bulanan.withColumn(
        "Persentase_Pertumbuhan", 
        round(((col("Harga_Tutup_Rata2") - col("Harga_Bulan_Lalu")) / col("Harga_Bulan_Lalu")) * 100, 2)
    )
    
    # Melabeli Fase Geopolitik (Batas: April 2024)
    rata_bulanan = rata_bulanan.withColumn(
        "Fase_Geopolitik",
        when((col("Tahun") > 2024) | ((col("Tahun") == 2024) & (col("Bulan") >= 4)), "Era Perang (Apr 2024 - Sekarang)")
        .otherwise("Sebelum Perang")
    )
    
    # Menampilkan Tabel Utama di Terminal dengan format rapi
    tabel_display = rata_bulanan.select(
        col("Tahun"),
        col("Bulan"),
        format_number(col("Harga_Tutup_Rata2"), 2).alias("Harga_IHSG"),
        format_string("%+.2f%%", col("Persentase_Pertumbuhan")).alias("Pertumbuhan_MoM"),
        col("Fase_Geopolitik")
    )
    
    print("\n================ DATA PERTUMBUHAN IHSG BULANAN ================")
    tabel_display.show(40, truncate=False)
    
    # =====================================================================
    # 5. RINGKASAN TREN SELAMA ERA PERANG
    # =====================================================================
    print("\n=== KESIMPULAN TREN SELAMA ERA PERANG (Apr 2024 - Sekarang) ===")
    df_perang = rata_bulanan.filter(col("Fase_Geopolitik") == "Era Perang (Apr 2024 - Sekarang)")
    
    ringkasan_perang = df_perang.agg(
        format_string("%+.2f%%", round(avg("Persentase_Pertumbuhan"), 2)).alias("Rata_Pertumbuhan_Bulanan"),
        format_number(avg("Harga_Tutup_Rata2"), 2).alias("Rata_Harga_IHSG_Perang")
    )
    ringkasan_perang.show(truncate=False)
    
# =====================================================================
    # 6. PREDIKSI SEDERHANA 3 BULAN KE DEPAN (Berdasarkan Tren)
    # =====================================================================
    print("=== PREDIKSI IHSG 3 BULAN KE DEPAN (Berdasarkan Tren Era Perang) ===")
    
    # Ambil fungsi round asli milik Python agar tidak bentrok dengan Spark
    import builtins
    
    avg_growth_row = df_perang.agg(avg("Persentase_Pertumbuhan").alias("avg_growth")).collect()[0]
    avg_growth = avg_growth_row['avg_growth']
    
    last_data_row = rata_bulanan.orderBy(col("Tahun").desc(), col("Bulan").desc()).limit(1).collect()[0]
    last_price = last_data_row['Harga_Tutup_Rata2']
    last_year = last_data_row['Tahun']
    last_month = last_data_row['Bulan']

    if avg_growth is not None and last_price is not None:
        prediksi_list = []
        current_price = last_price
        
        # Gunakan pembulatan Python biasa untuk variabel angka (bukan kolom)
        avg_growth_rounded = builtins.round(float(avg_growth), 2)
        
        for i in range(1, 4): 
            next_month = last_month + i
            next_year = last_year
            
            if next_month > 12:
                next_month -= 12
                next_year += 1
                
            current_price = current_price * (1 + (avg_growth / 100))
            # Simpan hasil prediksi
            prediksi_list.append((int(next_year), int(next_month), float(current_price), f"{avg_growth_rounded}% per bulan"))

        schema_prediksi = StructType([
            StructField("Tahun_Prediksi", LongType(), True),
            StructField("Bulan_Prediksi", LongType(), True),
            StructField("Prediksi_Harga_IHSG", DoubleType(), True),
            StructField("Asumsi_Tren", StringType(), True)
        ])
        
        df_prediksi = spark.createDataFrame(prediksi_list, schema_prediksi)
        
        tabel_prediksi_display = df_prediksi.select(
            col("Tahun_Prediksi"),
            col("Bulan_Prediksi"),
            format_number(col("Prediksi_Harga_IHSG"), 2).alias("Estimasi_Harga_IHSG"),
            col("Asumsi_Tren")
        )
        tabel_prediksi_display.show(truncate=False)
    else:
        print("Data tidak cukup untuk melakukan perhitungan prediksi.")
        
    print("===============================================================\n")

    # =====================================================================
    # 7. PEMBUATAN GRAFIK VISUALISASI
    # =====================================================================
    print("Membuat grafik visualisasi...")
    plot_df = rata_bulanan.orderBy("Tahun", "Bulan").select("Tahun", "Bulan", "Harga_Tutup_Rata2").toPandas()
    
    # Gabungkan Tahun dan Bulan jadi Date untuk Sumbu X
    plot_df['Date'] = pd.to_datetime(plot_df['Tahun'].astype(str) + '-' + plot_df['Bulan'].astype(str) + '-01')
    
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(plot_df['Date'], plot_df['Harga_Tutup_Rata2'], marker='o', color='navy', linewidth=2, label='Rata-rata IHSG Bulanan')
    
    # Garis Merah Penanda Era Perang (April 2024)
    perang_start = pd.to_datetime('2024-04-01')
    ax.axvline(x=perang_start, color='red', linestyle='--', linewidth=2, label='Eskalasi Perang Dimulai (Apr 2024)')
    ax.fill_betweenx([plot_df['Harga_Tutup_Rata2'].min() - 100, plot_df['Harga_Tutup_Rata2'].max() + 100], perang_start, plot_df['Date'].max(), color='red', alpha=0.1)

    ax.set_title('Pergerakan IHSG: Fase Sebelum vs Era Perang Geopolitik', fontsize=14, pad=15)
    ax.set_xlabel('Waktu (Bulan/Tahun)', fontsize=12)
    ax.set_ylabel('Harga Penutupan IHSG (IDR)', fontsize=12)
    ax.grid(True, linestyle='--', alpha=0.5)
    ax.legend(loc='upper left')
    
    nama_gambar = "Grafik_IHSG_Geopolitik.png"
    plt.savefig(nama_gambar, dpi=300, bbox_inches='tight')
    print(f"Grafik berhasil dibuat dan disimpan sebagai: {nama_gambar}")

    # =====================================================================
    # 8. EXPORT DATASET KE CSV
    # =====================================================================
    # Drop kolom lag agar tidak ada null di baris pertama CSV
    rata_bulanan = rata_bulanan.drop("Harga_Bulan_Lalu")
    rata_bulanan.write.csv("Dataset_IHSG_Geopolitik_Final", header=True, mode="overwrite")
    
    print("\nSUCCESS: Dataset berformat CSV berhasil diekspor ke folder 'Dataset_IHSG_Geopolitik_Final'!")

else:
    print("Gagal menarik data dari Yahoo Finance API. Pastikan ada koneksi internet.")