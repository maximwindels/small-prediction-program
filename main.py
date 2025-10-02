import numpy as np
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
from sklearn.datasets import make_regression

x, y = make_regression(n_samples=200, n_features=1, noise=12, random_state=2)
print("The first five values of x are: \n", x[0:10])
print("The first five values of y are: \n", y[0:10])

model = LinearRegression()

model.fit(x, y)

y_pred = model.predict(x)

plt.scatter(x, y, color='blue', label='Actual data')
plt.plot(x, y_pred, color='red', linewidth=2, label='Regression line')
equation = f"y = {model.intercept_:.4f} + {model.coef_[0]:.4f}x"
plt.title('Linear Regression Example')
plt.text(min(x), max(y), equation, fontsize=12, color="g", bbox=dict(facecolor='white', alpha=0.5))
plt.xlabel('X')
plt.ylabel('Y')
plt.show()