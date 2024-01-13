import math
import numpy as np 
import pandas as pd
import yfinance as yf
from pandas_datareader import data as pdr
import plotly.express as px
import matplotlib.pyplot as plt
import plotly.express as px
import datetime
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import load_model

from tensorflow import keras
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
import tensorflow as tf
from keras.models import load_model
import streamlit as st
import plotly.graph_objects as go
import pandas_ta as ta

st.set_page_config( page_title="NeuroStock", page_icon="chart_with_upwards_trend",layout="wide")
st.title('NeuroStock: Neural Network Enhanced Stock Price Prediction')

default_ticker = "GOOGL"
ticker=st.sidebar.text_input(f"Ticker (default: {default_ticker}):") or default_ticker
start_date = st.sidebar.date_input("Start date", datetime.date(2012, 1, 1))
end_date = st.sidebar.date_input("End date", datetime.date(2023, 9, 30))


st.sidebar.write("""
# Popular Stocks
### Alphabet Inc. (GOOG)
### Microsoft Corporation (MSFT)
### Apple Inc. (AAPL)
### Tesla, Inc. (TSLA)
### NVIDIA Corporation Common Stock (NVDA)
### JP Morgan Chase & Co. Common Stock (JPM)
### Coca-Cola Company (The) Common Stock (KO)
### Reliance Industries Limited (RELIANCE.NS)
### Netflix, Inc. (NFLX)
### Vanguard Intermediate-Term Bond ETF (BIV)
### BlackRock Global Dividend Ptf Investor A (BABDX)
### Emerson Electric Company Common Stock (EMR)
### Meta Platforms, Inc. Class A Common Stock (META)
### Walmart Inc. Common Stock (WMT)
### Tata Motors Limited (TATAMOTORS.NS)
### Tata Steel Limited (TATASTEEL.NS)
### Alphabet Inc. (GOOG)
### Amazon (AMZN)
### International Business Machines Corporation (IBM)
### Infosys Limited (INFY)
### Tata Consultancy Services Limited (TCS.NS)
""")
data1=yf.download(ticker, start=start_date, end=end_date)
data3=yf.Ticker(ticker)
df = data3.history(period='1d', start=start_date, end=end_date).reset_index()


# try:
   
#     string_logo = '<img src=%s>' % data3.info['logo_url']
#     st.markdown(string_logo, unsafe_allow_html=True)

#     string_name = data3.info['longName']
#     st.header('**%s**' % string_name)

#     string_summary = data3.info['longBusinessSummary']
#     st.info(string_summary)

# except Exception as e:
#     st.error(f"An error occurred while fetching data for {ticker}: {str(e)}")

fig1=go.Figure(data=[go.Candlestick(x=df['Date'],
                                        open=df['Open'],
                                        high=df['High'],
                                        low=df['Low'],
                                        close=df['Close'])])
fig1.update_layout(
        title=(f'CandleStick Chart of {ticker}'),
        yaxis_title='Price($)',
        xaxis_rangeslider_visible=False)
st.plotly_chart(fig1)


pricing_data,models= st.tabs(["Pricing Data","Models"])

with pricing_data:
    st.header('Price Movements')
    data2=data1
    data2["% Change"] = data1["Close"].pct_change()*100
    data2.dropna(inplace=True)
    st.write(data2)
    annual_return = data2['% Change'].mean()*252
    st.write('Annual Return :', annual_return,'%')
    sd=np.std(data2['% Change'])*np.sqrt(252)
    st.write('Standard Deviation :',sd,'%')
    st.write('Risk Adj. Return :',annual_return/(sd*100))
    


with models:
    col1,col2=st.columns((2))

    with col1:
        st.subheader("LSTM Model")

        series = df['Close']

        model1 = keras.models.load_model("my_checkpoint1.h5")

        def plot_series(time, series, format="-", start=0, end=None, label=None ,color=None):
            plt.plot(time[start:end], series[start:end], format, label=label,color=color)
            plt.xlabel("Time")
            plt.ylabel("Value")
            if label:
                plt.legend(fontsize=14)
            plt.grid(True)

        window_size = 30
        test_split_date = '2019-12-31'
        x_test = df.loc[df['Date'] >= test_split_date]['Close']

        x_test_values = x_test.values.reshape(-1, 1)
        x_train_scaler = MinMaxScaler(feature_range=(0, 1))
        x_train_scaler.fit(series.values.reshape(-1, 1))
        normalized_x_test = x_train_scaler.transform(x_test_values)

        rnn_forecast = model1.predict(normalized_x_test[np.newaxis,:])
        rnn_forecast = rnn_forecast.flatten()
        rnn_unscaled_forecast = x_train_scaler.inverse_transform(rnn_forecast.reshape(-1,1)).flatten()

        fig2=plt.figure(figsize=(7,4))

        plt.ylabel('Dollars $',color='white')
        plt.xlabel('Timestep in Days',color='white')
        plt.title(f'LSTM Forecast vs Actual',color='white')
        plot_series(x_test.index, x_test, label="Actual")
        plot_series(x_test.index, rnn_unscaled_forecast, label="Forecast")
        st.plotly_chart(fig2,use_container_width=True)

        lstm_mea=keras.metrics.mean_absolute_error(x_test, rnn_unscaled_forecast).numpy()
        st.write('LSTM Mean Absolute Error :',lstm_mea)
        lstm_rmse = np.sqrt(mean_squared_error(x_test,rnn_unscaled_forecast))
        st.write('LSTM Root Mean Square Error :',lstm_rmse)

    
    with col2:
        st.subheader("NAIVE BAYES Model")

        series = df['Close']

        test_split_date = '2021-12-01'
        test_split_index = np.where(df.Date == test_split_date)[0][0]
        x_test = df.loc[df['Date'] >= test_split_date]['Close']

        naive_forecast = series[test_split_index-1 :-1]

        fig3=plt.figure(figsize=(7,4))
        
        plot_series(x_test.index, x_test, label="Actual")
        plot_series(x_test.index, naive_forecast, label="Forecast")
        plt.ylabel('Dollars $',color='white')
        plt.xlabel('Timestep in Days',color='white')
        plt.title('Naive Forecast vs Actual',color='white')
        st.plotly_chart(fig3,use_container_width=True)

        naive_forecast_mae = keras.metrics.mean_absolute_error(x_test, naive_forecast).numpy()
        st.write('Naive Bayes Mean Absolute Error :',naive_forecast_mae)
        naive_forecast_rmse = np.sqrt(mean_squared_error(x_test,naive_forecast))
        st.write('Naive Bayes Root Mean Square Error :',naive_forecast_rmse)
        


    col1,col2=st.columns((2))

    with col1:
        st.subheader("CNN Model")

        window_size = 20
        model2 = keras.models.load_model("my_checkpoint.h5")

        def model_forecast(model, series, window_size):
            ds = tf.data.Dataset.from_tensor_slices(series)
            ds = ds.window(window_size, shift=1, drop_remainder=True)
            ds = ds.flat_map(lambda w: w.batch(window_size))
            ds = ds.batch(32).prefetch(1)
            forecast = model.predict(ds)
            return forecast

        x_train_scaler = MinMaxScaler(feature_range=(0, 1))
        x_train_scaler.fit(series.values.reshape(-1, 1)) 
        spy_normalized_to_traindata = x_train_scaler.transform(series.values.reshape(-1, 1))

        cnn_forecast = model_forecast(model2, spy_normalized_to_traindata[:,  np.newaxis], window_size)
        cnn_forecast = cnn_forecast[x_test.index.min() - window_size:-1,-1,0]
        cnn_unscaled_forecast = x_train_scaler.inverse_transform(cnn_forecast.reshape(-1,1)).flatten()

        fig4=plt.figure(figsize=(7,4))
        plt.ylabel('Dollars $',color='white')
        plt.xlabel('Timestep in Days',color='white')
        plt.title(f'Full CNN Forecast vs Actual',color='white')
        plot_series(x_test.index, x_test,label="Actual")
        plot_series(x_test.index, cnn_unscaled_forecast,label="Forecast")
        st.plotly_chart(fig4,use_container_width=True)

        cnn_mae=keras.metrics.mean_absolute_error(x_test, cnn_unscaled_forecast).numpy()
        st.write('CNN Mean Absolute Error :',cnn_mae)
        cnn_rmse = np.sqrt(mean_squared_error(x_test,cnn_unscaled_forecast))
        st.write('CNN Root Mean Square Error :',cnn_rmse)


    with col2:
        st.subheader("Linear Model")

        model4 = keras.models.load_model("my_checkpoint.h5")
        window_size = 30

        
        lin_forecast = model_forecast(model4, spy_normalized_to_traindata.flatten()[x_test.index.min() - window_size:-1], window_size)[:, 0]
        lin_forecast = x_train_scaler.inverse_transform(lin_forecast.reshape(-1,1)).flatten()

        fig5=plt.figure(figsize=(7,4))
        plt.title('Linear Forecast',color='white')
        plt.ylabel('Dollars $',color='white')
        plt.xlabel('Timestep in Days',color='white')
        plot_series(x_test.index, x_test,label="Actual")
        plot_series(x_test.index, lin_forecast,label="Forecast")
        st.plotly_chart(fig5,use_container_width=True)


        linear_mea = keras.metrics.mean_absolute_error(x_test, lin_forecast).numpy()
        st.write('LINEAR Mean Absolute Error :',linear_mea)
        linear_rmse = np.sqrt(mean_squared_error(x_test,lin_forecast))
        st.write('LINEAR Root Mean Square Error :',linear_rmse)



