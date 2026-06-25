import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# Load stock data
data = yf.download('AAPL', start='2020-01-01', end='2023-01-01')

# Create prediction column
data['Prediction'] = data['Close'].shift(-30)

# Prepare data
X = data[['Close']][:-30]
y = data['Prediction'][:-30]

# Train model
model = LinearRegression()
model.fit(X, y)

# Predict future
future_price = model.predict([[150]])
print("Predicted Price:", future_price)

# Plot
plt.plot(data['Close'])
plt.title("Stock Price")
plt.savefig("output.png")
plt.show()