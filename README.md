# Algo-Trading-Math-Models
## Math Techniques viz. ARIMA, Frequency Decomposition, Fourier Filtering, Linear Regression &amp;  Bi-directional LSTMs on Feature Engineered Stock Market Data.


**Quant Trading**
Quant strategies follow a data-driven approach to pick stocks. **This approach  which seeks to reduce the role of human bias conceptually fall in between active and passive trading.** The stock data is a classic example of " time series" where the prices are sampled at regular intervals.


**Fourier Filtering**

Fourier Filtering helps to **de-noise the signal** in order to find out the significant curve. This technique can be used before feeding the prediction model **or even to smooth the model output.**
![alt text](images/fourier_filtering.png)


**ARIMA Model on TCS Stock Data**

aka. Box-Jenkins model, ARIMA was created in 1976. It is conceptually similar to a linear regression model applied on time series data. <br><br>

**ARIMA has 3 parts:**
**- Auto Regression**
**- Integration (Differencing)**
**- Moving Average**

![alt text](images/ARIMA_Prediction.png)
![alt text](images/ARIMA_Prediction_zoomed.png)

**Linear Regression**<br>

We were using **only 'p' previous values and 'q' errors to predict.** But we can use other features such as day of week, time of day, holidays etc. This technique, known as Feature Engineering is more of an art than science.<br>

Here we take a CSV file containing daily record of the price of the S&P500 Index from 1950 to 2015. Lets try to predict response variable, i.e. closing price, prior to a day. We can use features the features below: <br>
1) **Average Price** of past 365 days.
2) Ratio of average price for the past 5 days & past 365 days.
3) **Mean and Standard Deviation** of previous 365 days.

![alt text](images/LR.png)

**Frequency Decomposition using Power Spectral Density Curve** <br>
Auto correlation is one way to compute periodicity. But a more scientific way to find periodicity is Fourier Transform. This technique can be used to find out the **most significant periodic changes in historical data,** which gives a dependable hint about future.<br>

![alt text](images/freq_decompose.png)
![alt text](images/psd.png)
![alt text](images/freq_decomposed.png)
![alt text](images/freq_decomposed_sum.png)

**LSTM on S&P500 Time Series with Fourier Filering**
<br>
Long Short-Term Memory networks, can be used to learn from the series of past observations to predict the next value in the sequence.<br><br>
A vanilla LSTM model has a single hidden layer of LSTM units, and an output layer used to make prediction. Here we are working with a uni-variate series, so the number of features is one.<br><br>
**We apply LSTM on the same S&P 500 data taken for Fourier filtering.** First we draw the auto correlation graph to estimate the lag. <br><br>
![alt text](images/prediction_snp_lstm.png)
![alt text](images/lstm_fourier_filtered.png)

We tried to model stock market behaviour using supervised learning approaches viz. Linear Regression or LSTM. But Reinforcement Learning is more robust to account for various environmental factors that affects stock market, as it aims to maximise reward in a given situation.

Fourier analysis works best with waves or wavelets that are regular and predictable, for which stock market is an antithesis. Hence it is beneficial to look into spectral analysis and signal extraction also.
