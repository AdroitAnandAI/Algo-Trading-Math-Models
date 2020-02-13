# Algo-Trading-Math-Models
## Math Techniques viz. ARIMA, Frequency Decomposition, Fourier Filtering, Linear Regression &amp;  Bi-directional LSTMs on Feature Engineered Stock Market Data.


**Quant Trading**
Quant strategies follow a data-driven approach to pick stocks. This approach  which seeks to reduce the role of human bias conceptually fall in between active and passive trading. The stock data is a classic example of " time series" where the prices are sampled at regular intervals.


**Fourier Filtering**

Fourier Filtering helps to de-noise the signal in order to find out the significant curve. This technique can be used before feeding the prediction model or even to smooth the model output.
![alt text](images/fourier_filtering.png)

**ARIMA Model on TCS Stock Data**
![alt text](images/autoCorrelation_tcs.png)
![alt text](images/ARIMA_Prediction.png)
![alt text](images/ARIMA_Prediction_zoomed.png)

**Linear Regression**
![alt text](images/LR.png)

**Frequency Decomposition using Power Spectral Density Curve**
![alt text](images/freq_decompose.png)
![alt text](images/psd.png)
![alt text](images/freq_decomposed.png)
![alt text](images/freq_decomposed_sum.png)

**LSTM on S&P500 Time Series with Fourier Filering**
![alt text](images/Autocorr_snp.png)
![alt text](images/prediction_snp_lstm.png)
![alt text](images/lstm_fourier_filtered.png)


**References**
<br>
<br> [1] *Inversion Detection in Text Document Images. Hamid Pilevar, A. G. Ramakrishnan, Medical Intelligence and Language Engineering Lab, Department of Electrical Engineering, Indian Institute of Science, Bangalore (JCIS 2006)*<br>
<br> [2] *Shape Context: A new descriptor for shape matching and object recognition. Serge Belongie, Jitendra Malik and Jan Puzicha. Department of Electrical Engineering and Computer Sciences, University of California at Berkeley (NIPS 2000)*<br>
<br> [3] *Shape Matching and Object Recognition Using Shape Contexts. Serge Belongie, Jitendra Malik and Jan Puzicha. Computer Science Division, University of California at Berkeley (PAMI 2002)*
