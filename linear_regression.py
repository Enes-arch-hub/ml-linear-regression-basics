import pandas as pd  # For reading and handling data
import matplotlib.pyplot as plt  # For plotting charts
# Replace the path if your folder is different
data = pd.read_csv("house_prices.csv")  

# Check the first few rows
print(data.head())
plt.scatter(data['Size'], data['Price'])
plt.title("House Size vs Price")
plt.xlabel("Size")
plt.ylabel("Price")
plt.show()

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

# Feature (independent variable)
X = data[['Size']]

# Target (dependent variable)
y = data['Price']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = LinearRegression()

model.fit(X_train, y_train)

prediction = model.predict([[115]])
print("Predicted price:", prediction[0])

plt.scatter(X, y)  # original points
plt.plot(X, model.predict(X), color='red')  # prediction line
plt.title("Linear Regression Fit")
plt.xlabel("Size")
plt.ylabel("Price")
plt.show()

