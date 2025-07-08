# config.py
# ==============================================================================
# PUSAT KENDALI APLIKASI
# ==============================================================================
# File ini berisi semua parameter dan konfigurasi untuk aplikasi.
# Ubah nilai variabel di bawah ini untuk menyesuaikan analisis tanpa mengubah
# file .py lainnya.
# ==============================================================================


# --- 1. PENGATURAN PENGAMBILAN DATA ---

"""
TICKER
--------------------------------------------------------------------------------
Tujuan:
  Mendefinisikan kode saham (emiten) yang ingin dianalisis dan diprediksi.
  Ini adalah input utama untuk pengambilan data.

Contoh & Pilihan:
  - "BBCA.JK": Bank Central Asia Tbk.
  - "TLKM.JK": Telkom Indonesia (Persero) Tbk.
  - "ASII.JK": Astra International Tbk.
  - Anda juga bisa menggunakan kode saham internasional, misalnya "GOOGL" untuk
    Alphabet (Google) atau "MSFT" untuk Microsoft.

Catatan Penting:
  - Untuk saham yang terdaftar di Bursa Efek Indonesia (BEI), Anda wajib
    menambahkan akhiran ".JK".
  - Anda bisa mencari kode ticker yang benar di situs Yahoo Finance.
"""
TICKER = "BBCA.JK"


"""
FETCH_PERIOD
--------------------------------------------------------------------------------
Tujuan:
  Menentukan rentang waktu data historis yang akan diunduh. Model akan
  belajar dari data dalam rentang ini.

Contoh & Pilihan:
  - "1y": 1 tahun terakhir.
  - "5y": 5 tahun terakhir (nilai saat ini).
  - "10y": 10 tahun terakhir.
  - "max": Mengambil semua data historis yang tersedia sejak saham tersebut IPO.
  - Pilihan lain: "1d", "5d", "1mo", "3mo", "6mo", "ytd" (year to date).

Catatan Penting:
  - Menggunakan periode panjang (misal, "max") memberikan model lebih banyak
    data, tetapi data yang sangat lampau bisa jadi tidak relevan lagi.
  - Periode pendek (misal, "1y") lebih fokus pada perilaku pasar terkini.
  - Memulai dengan "5y" adalah pilihan yang seimbang dan umum digunakan.
"""
FETCH_PERIOD = "5y"


# --- 2. PENGATURAN PERAMALAN (FORECASTING) ---

"""
FORECAST_STEPS
--------------------------------------------------------------------------------
Tujuan:
  Menentukan berapa hari ke depan prediksi akan dibuat, dihitung dari hari
  terakhir data historis.

Contoh & Pilihan:
  - 30: Memprediksi sekitar 1 bulan ke depan.
  - 90: Memprediksi sekitar 3 bulan (satu kuartal) ke depan.
  - 365: Memprediksi 1 tahun ke depan.

Catatan Penting:
  - Semakin jauh Anda memprediksi, semakin besar ketidakpastiannya.
  - Gunakan hasil prediksi jangka panjang untuk melihat arah tren umum, bukan
    untuk mendapatkan nilai harga yang pasti.
"""
FORECAST_STEPS = 90


"""
CHANGEPOINT_PRIOR_SCALE
--------------------------------------------------------------------------------
Tujuan:
  Parameter teknis Prophet yang mengontrol fleksibilitas garis tren.

Penjelasan Sederhana:
  - Nilai Kecil (misal, 0.01): Membuat tren "kaku" dan tidak reaktif.
    Berisiko underfitting (model terlalu sederhana).
  - Nilai Besar (misal, 0.5): Membuat tren "lentur" dan sangat mengikuti
    pergerakan harga. Berisiko overfitting (model terlalu sensitif pada noise).

Contoh & Pilihan:
  - 0.05: Nilai default dari Prophet.
  - 0.1: Nilai yang kita gunakan, sedikit lebih fleksibel untuk data saham.

Catatan Penting:
  - Ubah parameter ini jika Anda merasa garis tren prediksi terlalu lurus (naikkan
    nilainya) atau terlalu bergelombang (turunkan nilainya).
"""
CHANGEPOINT_PRIOR_SCALE = 0.1


# --- 3. PENGATURAN OUTPUT ---

"""
OUTPUT_FILENAME
--------------------------------------------------------------------------------
Tujuan:
  Menentukan nama file tempat hasil prediksi akan disimpan.

Contoh & Pilihan:
  - "hasil_prediksi_bbca.csv" (nilai saat ini).
  - "forecast_tlkm_q3_2025.xlsx"

Catatan Penting:
  - Menggunakan ekstensi .csv (Comma-Separated Values) adalah praktik terbaik
    karena file ini ringan dan bisa dibuka oleh hampir semua program spreadsheet
    (Excel, Google Sheets, dll.).
"""
OUTPUT_FILENAME = "hasil_prediksi_bbca.csv"
