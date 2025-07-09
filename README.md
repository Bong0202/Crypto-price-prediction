# Crypto-price-prediction

# Bitcoin Price Forecasting using Ridge Regression 

This project focuses on applying machine learning to forecast the price of Bitcoin (BTC-USD) using historical time-series data and engineered features. This project uses Ridge Regression as a baseline forecasting model and prepares the foundation for more advanced models like Random Forest and XGBoost.

---

## Objective

The goal of this project is to answer the question:

> *"Can we forecast the next-day price of Bitcoin based on historical patterns such as lagged prices, returns, and volatility?"*

This involves:
- Acquiring historical Bitcoin price data
- Engineering time-series features
- Training a Ridge Regression model for 1-day ahead price prediction
- Evaluating model accuracy using RMSE and visualizations
- Laying the groundwork for multi-horizon and multi-asset forecasting

---

## Data Source

- **Ticker**: BTC-USD (Bitcoin to USD)
- **Source**: `yfinance` (Yahoo Finance)
- **Date Range**: January 2018 – June 2025
- **Granularity**: Daily price data

---

##  Methodology

###  1. Feature Engineering

Time-series forecasting relies heavily on past trends. These features were created:

| Feature Name       | Description                                      |
|--------------------|--------------------------------------------------|
| `price_lag_1`      | Price 1 day ago                                  |
| `price_lag_7`      | Price 7 days ago                                 |
| `price_return_1d`  | 1-day return (percent change)                    |
| `price_return_7d`  | 7-day return                                     |
| `rolling_mean_7`   | 7-day moving average of price                    |
| `rolling_std_7`    | 7-day rolling standard deviation (volatility)   |

Target variable:
- `target_1d`: the price of Bitcoin 1 day in the future (`price.shift(-1)`)

---

###  2. Ridge Regression

Ridge Regression is a regularized form of linear regression that applies an L2 penalty to reduce overfitting and improve generalization. It assumes a linear relationship between features and the target, which makes it suitable as a **simple, interpretable baseline** model.

Model training steps:
- Data split: 80% training, 20% testing
- No shuffling (to preserve time order)
- Model evaluation using RMSE
- Visual comparison between predicted and actual prices

---

##  Results

| Metric     | Value        |
|------------|--------------|
| RMSE       | ~$2,445 USD  |
| Time Horizon | 1 Day Ahead |
| Model      | Ridge Regression |

###  Visualization

The chart below shows how closely the Ridge model predicted next-day prices over the test set period:

![image](https://github.com/user-attachments/assets/4ea724b8-7d93-4714-bed3-e442d4253797)


---

## Technologies Used

- **Python 3.11** via Jupyter Notebook
- `yfinance` – Data acquisition
- `pandas`, `numpy` – Data cleaning & feature engineering
- `scikit-learn` – Ridge Regression, train/test split, evaluation
- `matplotlib` – Data visualization


