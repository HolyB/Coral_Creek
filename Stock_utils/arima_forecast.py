import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
import numpy as np
import statsmodels.api as sm
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.stats.diagnostic import acorr_ljungbox
from sklearn import metrics

# Define ARIMA analysis class
class ARIMAStockForecast:
    def __init__(self, df, ticker: str):
        self.df = df
        self.ticker = ticker
        self.training_set = None
        self.test_set = None

    def preprocess_data(self, train_end_date='2021-06-21', test_start_date='2021-06-22'):
        # Convert date column to datetime if not already done
        self.df.index = pd.to_datetime(self.df.index, format='%Y-%m-%d')
        # Split data into training and test sets based on provided dates
        self.training_set = self.df.loc[:train_end_date, :]
        self.test_set = self.df.loc[test_start_date:, :]
        
        plt.figure(figsize=(10, 6))
        plt.plot(self.training_set['close'], label='Training Set')
        plt.plot(self.test_set['close'], label='Test Set')
        plt.title('Close Price')
        plt.xlabel('Time', fontsize=12, verticalalignment='top')
        plt.ylabel('Close Price', fontsize=14, horizontalalignment='center')
        plt.legend()
        plt.show()

    def perform_diff(self):
        # First-order diff
        self.df['diff_1'] = self.df['close'].diff(1)
        plt.figure(figsize=(10, 6))
        self.df['diff_1'].plot()
        plt.title('First-order Difference')
        plt.xlabel('Time', fontsize=12, verticalalignment='top')
        plt.ylabel('Difference', fontsize=14, horizontalalignment='center')
        plt.show()

        # Second-order diff
        self.df['diff_2'] = self.df['diff_1'].diff(1)
        plt.figure(figsize=(10, 6))
        self.df['diff_2'].plot()
        plt.title('Second-order Difference')
        plt.xlabel('Time', fontsize=12, verticalalignment='top')
        plt.ylabel('Difference', fontsize=14, horizontalalignment='center')
        plt.show()

    def white_noise_test(self):
        # White noise test
        temp2 = np.diff(self.df['close'], n=1)
        test_results = acorr_ljungbox(temp2, lags=2, boxpierce=True)
        print("White Noise Test Result:", test_results)
        return test_results

    def fit_arima_model(self, order=(2, 1, 0)):
        # Fit ARIMA model to training data
        model = ARIMA(self.training_set['close'], order=order)
        model_fit = model.fit()
        print(model_fit.summary())
        return model_fit

    def forecast(self, model_fit):
        history = list(self.training_set['close'])
        predictions = []
        
        for t in range(len(self.test_set)):
            # Use the history to fit ARIMA and forecast next step
            model = ARIMA(history, order=(2, 1, 0))
            model_fit = model.fit()
            yhat = model_fit.forecast()[0]
            predictions.append(yhat)
            obs = self.test_set['close'].iloc[t]
            history.append(obs)
        
        predictions_df = pd.DataFrame({'trade_date': self.test_set.index, 'close': predictions})
        predictions_df.set_index('trade_date', drop=True, inplace=True)
        
        plt.figure(figsize=(10, 6))
        plt.plot(self.test_set['close'], label='Stock Price')
        plt.plot(predictions_df['close'], label='Predicted Stock Price')
        plt.title(f'ARIMA: Stock Price Prediction for {self.ticker}')
        plt.xlabel('Time', fontsize=12, verticalalignment='top')
        plt.ylabel('Close Price', fontsize=14, horizontalalignment='center')
        plt.legend()
        plt.show()
        
        return predictions_df

    def residual_analysis(self, model_fit):
        residuals = pd.DataFrame(model_fit.resid)
        fig, ax = plt.subplots(1, 2, figsize=(14, 6))
        residuals.plot(title="Residuals", ax=ax[0])
        residuals.plot(kind='kde', title='Density', ax=ax[1])
        plt.show()

    def evaluate_model(self, test_set, predictions):
        mae = metrics.mean_absolute_error(test_set, predictions)
        mse = metrics.mean_squared_error(test_set, predictions)
        rmse = np.sqrt(mse)
        print(f'MAE: {mae:.2f}, MSE: {mse:.2f}, RMSE: {rmse:.2f}')

# Example usage
if __name__ == "__main__":
    # Sample DataFrame (you should provide your own DataFrame here)
    data = {
        'close': [100, 102, 101, 105, 110, 115, 120, 125, 130, 128, 134, 138],
    }
    df = pd.DataFrame(data)
    df.index = pd.date_range(start='2023-01-01', periods=len(df))

    # Initialize ARIMAStockForecast
    arima_forecast = ARIMAStockForecast(df, ticker='AAPL')
    arima_forecast.preprocess_data(train_end_date='2023-01-08', test_start_date='2023-01-09')
    arima_forecast.perform_diff()
    model_fit = arima_forecast.fit_arima_model()
    predictions = arima_forecast.forecast(model_fit)
    arima_forecast.residual_analysis(model_fit)
    arima_forecast.evaluate_model(arima_forecast.test_set['close'], predictions['close'])