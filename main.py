# main.py
# File utama untuk menjalankan seluruh alur proses dan menyimpan hasilnya.

# Import konfigurasi
from config import (
    TICKER,
    FETCH_PERIOD,
    FORECAST_STEPS,
    CHANGEPOINT_PRIOR_SCALE,
    OUTPUT_FILENAME
)

# Import fungsi dari file modul lainnya
from scraping import fetch_stock_data
from forecast import forecast_with_prophet

def run_analysis():
    """Fungsi utama untuk menjalankan alur kerja analisis."""
    print("--- Memulai Proses Analisis Saham ---")
    print(f"Saham           : {TICKER}")
    print(f"Periode Data    : {FETCH_PERIOD}")
    print(f"Forecast        : {FORECAST_STEPS} hari ke depan")
    print("---------------------------------------")

    stock_data = fetch_stock_data(ticker=TICKER, period=FETCH_PERIOD)

    if not stock_data.empty:
        forecast_result = forecast_with_prophet(
            stock_df=stock_data,
            ticker=TICKER,
            forecast_days=FORECAST_STEPS,
            changepoint_scale=CHANGEPOINT_PRIOR_SCALE
        )
        
        try:
            columns_to_save = ['ds', 'yhat', 'yhat_lower', 'yhat_upper', 'trend', 'trend_lower', 'trend_upper']
            forecast_result[columns_to_save].to_csv(OUTPUT_FILENAME, index=False)
            print(f"\n✅ Hasil prediksi telah disimpan ke file: {OUTPUT_FILENAME}")

        except Exception as e:
            print(f"\n❌ Gagal menyimpan file. Error: {e}")

        print(f"\n--- Cuplikan Hasil Prediksi (5 hari terakhir) ---")
        print(forecast_result[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail())
        print("\n--- Proses Analisis Selesai ---")
    else:
        print("\nProses dihentikan karena gagal mendapatkan data saham.")

if __name__ == "__main__":
    run_analysis()