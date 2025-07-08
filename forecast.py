# forecast.py
# Modul ini hanya berisi fungsi untuk melakukan peramalan menggunakan Prophet.

import pandas as pd
from prophet import Prophet

def forecast_with_prophet(stock_df, ticker, forecast_days, changepoint_scale):
    """
    Melakukan forecasting pada harga saham menggunakan Prophet dan mengembalikan hasilnya.
    """
    # a. Persiapan Data untuk Prophet ('ds' dan 'y')
    prophet_df = pd.DataFrame({
        'ds': stock_df.index,
        'y': stock_df['Close'].values.flatten()
    })

    # b. Inisialisasi Model Prophet
    model = Prophet(
        weekly_seasonality=False,
        yearly_seasonality=False,
        daily_seasonality=False,
        changepoint_prior_scale=changepoint_scale
    )
    model.add_country_holidays(country_name='ID')

    # c. Training Model
    print(f"\nMelakukan training model Prophet untuk {ticker}...")
    model.fit(prophet_df)
    print("Training selesai.")

    # d. Membuat DataFrame Masa Depan dan Melakukan Prediksi
    future_df = model.make_future_dataframe(periods=forecast_days, freq='D')
    forecast_df = model.predict(future_df)
    print(f"Prediksi untuk {forecast_days} hari ke depan selesai.")

    return forecast_df
