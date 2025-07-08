# scraping.py
# Modul ini bertanggung jawab untuk mengambil data saham dari yfinance.

import yfinance as yf
import pandas as pd
import warnings

warnings.filterwarnings("ignore", category=FutureWarning, module="yfinance.utils")

def fetch_stock_data(ticker, period):
    """
    Fungsi untuk mengambil data historis dari SATU saham.
    """
    print(f"Mengunduh data untuk: {ticker} (periode: {period})...")
    try:
        stock_data = yf.download(ticker, period=period, auto_adjust=True, progress=False)

        if stock_data.empty:
            print(f"Gagal mengunduh data atau kode saham '{ticker}' tidak valid.")
            return pd.DataFrame()

        print("Pengunduhan data selesai.")
        return stock_data

    except Exception as e:
        print(f"Terjadi kesalahan saat mengunduh data: {e}")
        return pd.DataFrame()