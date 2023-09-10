# Model Card

## Model Description

**Input:** The inputs of the model included a sequence of daily prices for the MSFT stock for each close price where the sequence had short-term (7 days) historical prices for each input

**Output:** The output was a single prediction of the price for the given day using the short term historical prices

**Model Architecture:** The model architecture used for stock price prediction with LSTM (Long Short-Term Memory) involved a single LSTM layer with a hidden size of 64 followed by 2 fully connected layers and ReLU activation in between them eventually generating a single output.

## Performance

The model performed very well and identified sequences and temporal patterns of the time series data with great relative accuracy. As the prediction was for stock prices, I used the Mean Squared Error as my main metric to evaluate the performance of the model. When testing, the MSE was 0.0031643306928042943 which is very low, showing that it worked well.

## Limitations

The model only considers closing prices for x number of previous days currently. It does not take into account other factors that may affect the forecasting of the stock prices and how they could affect the movement of the price.

## Trade-offs

The performance of the model requires a large amount of historical data to perform well. Therefore a large amount of training data was used to train. This may sometimes cause a risk of overfitting if not handled properly though.
