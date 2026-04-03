import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error

df = pd.read_csv(r"C:\Users\yoeshwar\OneDrive\Pictures\Desktop\internships\cleaned_superstore.csv", encoding='latin1')

df['Order Date'] = pd.to_datetime(df['Order Date'])

sales_data = df.groupby('Order Date')['Sales'].sum().reset_index()

sales_data = sales_data.sort_values('Order Date')

sales_data['Days'] = (sales_data['Order Date'] - sales_data['Order Date'].min()).dt.days

X = sales_data[['Days']]
y = sales_data['Sales']

split = int(len(sales_data) * 0.8)

X_train = X[:split]
X_test = X[split:]
y_train = y[:split]
y_test = y[split:]

model = LinearRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)

print("MAE:", mae)
print("MSE:", mse)

plt.figure()
plt.plot(sales_data['Order Date'], sales_data['Sales'])
plt.title("Historical Sales")
plt.show()

plt.figure()
plt.plot(sales_data['Order Date'][split:], y_test, label='Actual')
plt.plot(sales_data['Order Date'][split:], y_pred, label='Predicted')
plt.legend()
plt.title("Actual vs Predicted Sales")
plt.show()

future_days = np.arange(X['Days'].max()+1, X['Days'].max()+31).reshape(-1,1)

future_predictions = model.predict(future_days)

future_dates = pd.date_range(start=sales_data['Order Date'].max()+pd.Timedelta(days=1), periods=30)

future_df = pd.DataFrame({
    'Date': future_dates,
    'Predicted Sales': future_predictions
})

print(future_df)

plt.figure()
plt.plot(sales_data['Order Date'], sales_data['Sales'], label='Historical')
plt.plot(future_dates, future_predictions, label='Forecast')
plt.legend()
plt.title("Sales Forecast (Next 30 Days)")
plt.show()

future_df.to_csv(r"C:\Users\yoeshwar\OneDrive\Pictures\Desktop\internships\Sales_Forecast_Output.csv", index=False)