# Datasheet for Stock Price of MSFT


## Motivation

- The dataset was created for general use. It is historical pricing data for the stock: MSFT. It is a valuable resource for financial analysts, researchers, and data scientists. It enables users to analyse and model the historical performance of Microsoft's stock, which is one of the world's leading technology companies.
- The dataset is generated and made available by Yahoo Finance as part of its financial data services. The funding for the creation and maintenance of Yahoo Finance, including the provision of such financial data comes from Yahoo Finance's parent companny Yahoo Inc.

 
## Composition

- The dataset represents historical stock price data for Microsoft Corporation (MSFT). Each instance in the dataset corresponds to a specific date, and the data associated with each instance includes various attributes related to MSFT stock prices and trading indicators for that particular day.

- The number of instances depends on the range of data you would like to pull as it is time series data. It would be equal to the number of business days within the time range that you would select when pulling the data.

- There was no missing data within the dataset. All values had been reviewed by Yahoo Finance before being made available to users.

- The dataset does not contain data that might be considered confidential or include any information protected by legal privilege. It consists of publicly available historical stock price and trading indicator data for Microsoft Corporation (MSFT). This data is not related to individuals' non-public communications or sensitive personal information.

## Collection process

- The data for historical MSFT (Microsoft Corporation) stock prices was acquired from financial data providers, stock exchanges, or publicly available sources.
- The data can be retrieved in smaller samples by filtering the start and end date parameters when pulling the time series data. However, for this model we decided to use all of the data available.
- The dataset was collected daily over many years ranging from 1986 to present day.

## Preprocessing/cleaning/labelling

- There was no preprocessing or cleaning of the data needed. It was already in a clean form and ready to be processed. There was some further target variable generation to work with LSTMs for our model to work but other than that, the data was of high quality when it was pulled.
 
## Uses

- It could be used for various financial analysis tasks beyond stock price forecasting. Some potential applications include volatility prediction, trading strategy development, risk assessment, and portfolio optimization. Researchers and analysts in the finance domain may find this dataset valuable for exploring different aspects of stock market data.

- Financial datasets inherently involve sensitive information that, if misused, could result in unfair treatment, stereotyping, or financial losses. Dataset consumers should be cautious about making decisions solely based on model predictions without human judgment. Additionally, it's essential to be aware of potential biases in the data, such as historical biases or data collection practices that may not be representative of future market conditions. To mitigate risks, consumers can use this dataset as a tool for analysis and decision support rather than as the sole basis for critical financial decisions.

- The dataset should not be used for making high-stakes financial decisions without thorough validation and risk assessment. It is not a crystal ball for predicting stock prices with absolute certainty. Additionally, it should not be used for making judgments about individual investors or companies beyond the scope of financial analysis

## Distribution

- The dataset and it's smaller samples are currently distributed via Yahoo Finance's website under the MSFT ticker at https://finance.yahoo.com/quote/MSFT/history.

- The data is publicly available and available for downloading. There is nothing shown from Yahoo Finance to suggest there is any issue of redistribution or the dataset being under Copyright.

## Maintenance

- The dataset is primarily created and maintained by Yahoo Finance itself.