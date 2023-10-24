import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression, SGDRegressor
import time

x = 0.4 * np.linspace(-3, 3, 500).reshape(500, 1)
y = 6 + 4 * x + np.random.randn(500, 1)

def linReg(x,y): 
    # returns [y_approx, a, b] 
    lin_reg = LinearRegression()
    lin_reg.fit(x, y)
    y_approx = lin_reg.predict(x)
    return [y_approx, lin_reg.coef_[0][0], lin_reg.intercept_[0]]

def sgdReg(x,y): 
    # returns [y_approx, a, b] 
    sgd_reg = SGDRegressor()
    sgd_reg.fit(x, y.ravel())
    y_approx = sgd_reg.predict(x)
    return [y_approx, sgd_reg.coef_[0], sgd_reg.intercept_[0]]

def sgdReg2(x,y): 
    # returns [y_approx, a, b] 
    sgd_reg = SGDRegressor(eta0=0.1)
    sgd_reg.fit(x, y.ravel())
    y_approx = sgd_reg.predict(x)
    return [y_approx, sgd_reg.coef_[0], sgd_reg.intercept_[0]]

def sgdReg3(x,y): 
    # returns [y_approx, a, b] 
    sgd_reg = SGDRegressor(eta0=0.1, penalty="elasticnet")
    sgd_reg.fit(x, y.ravel())
    y_approx = sgd_reg.predict(x)
    return [y_approx, sgd_reg.coef_[0], sgd_reg.intercept_[0]]

def plot_print(x,y,f, label):
    start_time = time.time()
    for i in range(1000):
        [y_approx, a, b] = f(x,y)
    end_time = time.time()

    plt.plot(x, y_approx, label=label)
    print(f"{label}:\n" +
        f"  a = {a}\n" +
        f"  b = {b}\n" +
        f"  time = {end_time - start_time}/1000")

plot_print(x,y,sgdReg,"SGD regr.")
plot_print(x,y,sgdReg2,"SGD regr. 2")
plot_print(x,y,sgdReg3,"SGD regr. 3")
plot_print(x,y,linReg,"Linear regr.")
plt.scatter(x, y, alpha=0.5)
plt.legend()
plt.show()
