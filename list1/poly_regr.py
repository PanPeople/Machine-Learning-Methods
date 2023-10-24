import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline
from sklearn.metrics import mean_squared_error

# 1. Generate the data points
X = 0.1 * np.linspace(-10, 10, 500).reshape(500, 1)
y = 3 * X**3 + 0.5 * X**2 + X + 2 + np.random.randn(500, 1)

# 2. Train polynomial regression with different degrees of polynomial
degrees = list(range(1, 11))
models = {}

for degree in degrees:
    model = make_pipeline(PolynomialFeatures(degree), LinearRegression())
    model.fit(X, y)
    models[degree] = model

# 3. Calculate mean squared error (MSE) for each model
mse_values = {}

for degree, model in models.items():
    y_pred = model.predict(X)
    mse = mean_squared_error(y, y_pred)
    mse_values[degree] = mse

# 4. Select the best model based on MSE
best_degree = min(mse_values, key=mse_values.get)
best_model = models[best_degree]

# 5. Write down the polynomial regression prediction equation
polynomial_features = best_model.named_steps["polynomialfeatures"]
linear_regression = best_model.named_steps["linearregression"]

coefficients = linear_regression.coef_
intercept = linear_regression.intercept_

equation = "ŷ = {:.2f}".format(intercept[0])
for i, coef in enumerate(coefficients[0][1:]):
    equation += " + {:.2f}x^{}".format(coef, i + 1)

print(equation)

# 6. Plot the original data and the polynomial regression curves for each degree
plt.figure(figsize=(15, 10))
plt.scatter(X, y, s=10, color='blue', label="Original Data")
colors = ['red', 'green', 'orange', 'purple', 'cyan', 'magenta', 'yellow', 'black', 'pink', 'brown']

for degree, color in zip(degrees, colors):
    X_sorted = np.sort(X, axis=0)
    plt.plot(X_sorted, models[degree].predict(X_sorted), color=color, linewidth=2, label="Degree {}".format(degree))

plt.title("Polynomial Regression for Different Degrees")
plt.xlabel("X")
plt.ylabel("y")
plt.legend()
plt.show()
