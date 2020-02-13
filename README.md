# Algo-Trading-Math-Models
## Math Techniques viz. ARIMA, Frequency Decomposition, Fourier Filtering, Linear Regression &amp;  Bi-directional LSTMs on Feature Engineered Stock Market Data.


**Quant Trading**
Quant strategies follow a data-driven approach to pick stocks. This approach  which seeks to reduce the role of human bias conceptually fall in between active and passive trading. The stock data is a classic example of " time series" where the prices are sampled at regular intervals.


**Fourier Filtering**

Fourier Filtering helps to de-noise the signal in order to find out the significant curve. This technique can be used before feeding the prediction model or even to smooth the model output.
![alt text](images/fourier_filtering.png)


**ARIMA Model on TCS Stock Data**

![alt text](images/ARIMA_Prediction.png)
![alt text](images/ARIMA_Prediction_zoomed.png)

**Linear Regression**<br>
![alt text](images/LR.png)

**Frequency Decomposition using Power Spectral Density Curve**
![alt text](images/freq_decompose.png)
![alt text](images/psd.png)
![alt text](images/freq_decomposed.png)
![alt text](images/freq_decomposed_sum.png)

**LSTM on S&P500 Time Series with Fourier Filering**
<br>
![alt text](images/prediction_snp_lstm.png)
![alt text](images/lstm_fourier_filtered.png)

We tried to model stock market behaviour using supervised learning approaches viz. Linear Regression or LSTM. But Reinforcement Learning is more robust to account for various environmental factors that affects stock market, as it aims to maximise reward in a given situation.

Fourier analysis works best with waves or wavelets that are regular and predictable, for which stock market is an antithesis. Hence it is beneficial to look into spectral analysis and signal extraction also.
