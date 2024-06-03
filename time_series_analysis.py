import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.arima.model import ARIMA

# Load the dataset
file_path = r'C:\Users\nisch\Downloads\New folder\portfolio_data.csv'
data = pd.read_csv(file_path)

# Display the first few rows and check for missing values
print(data.head())
print(data.isnull().sum())

# Parse dates and set the index
data['Date'] = pd.to_datetime(data['Date'])
data.set_index('Date', inplace=True)

# Display summary statistics for AMZN
print(data['AMZN'].describe())

# Plot AMZN stock price over time
plt.figure(figsize=(10, 6))
plt.plot(data['AMZN'])
plt.title('AMZN Stock Prices Over Time')
plt.xlabel('Date')
plt.ylabel('AMZN Closing Price')
plt.show()

# Perform seasonal decomposition for AMZN
decomposition = seasonal_decompose(data['AMZN'], model='additive', period=365)
fig = decomposition.plot()
fig.set_size_inches(14, 7)
plt.show()

# Fit an ARIMA model
model = ARIMA(data['AMZN'], order=(5, 1, 0))
model_fit = model.fit()

# Display the model summary
print(model_fit.summary())

# Forecast the next 30 days
forecast = model_fit.forecast(steps=30)
print(forecast)

# Create a new date range for the forecast
forecast_index = pd.date_range(start=data.index[-1] + pd.Timedelta(days=1), periods=30, freq='D')

# Plot the forecast
plt.figure(figsize=(10, 6))
plt.plot(data['AMZN'], label='Historical')
plt.plot(forecast_index, forecast, label='Forecast', color='red')
plt.title('AMZN Stock Price Forecast')
plt.xlabel('Date')
plt.ylabel('AMZN Closing Price')
plt.legend()
plt.show()
