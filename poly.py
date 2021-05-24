# Importing modules
import numpy as np
import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

# Training set
x_train = [[2], [4], [6], [8], [16]] # amount of ram in a pc
y_train = [[3000], [5000], [7000], [8000], [10000]] #prices of pc

# Testing set
x_test = [[2], [6], [8], [16]] # amount of ram in a pc
y_test = [[3500], [5000], [7000], [9000]] #prices of pizzas

# Training the Linear Regression model and plotting a prediction
regressor = LinearRegression()
regressor.fit(x_train, y_train)
xx = np.linspace(0, 26, 100)
yy = regressor.predict(xx.reshape(xx.shape[0], 1))
plt.plot(xx, yy)

# Setting the degree of the Polynomial Regression model
quadratic_featurizer = PolynomialFeatures(degree=2)

# Transforming an input data matrix into a new data matrix of a given degree
x_train_quadratic = quadratic_featurizer.fit_transform(x_train)
x_test_quadratic = quadratic_featurizer.transform(x_test)

# Train and test the regressor_quadratic model
regressor_quadratic = LinearRegression()
regressor_quadratic.fit(x_train_quadratic, y_train)
xx_quadratic = quadratic_featurizer.transform(xx.reshape(xx.shape[0], 1))

# Plotting the graph
plt.plot(xx, regressor_quadratic.predict(xx_quadratic), c='r', linestyle='--')
plt.title('PC price regressed on amount of RAM')
plt.xlabel('Amount of RAM in GB')
plt.ylabel('Price in dollars')
plt.axis([0, 20, 0, 20000])
plt.grid(True)
plt.scatter(x_train, y_train)
plt.show()
print(x_train)
print(x_train_quadratic)
print(x_test)
print(x_test_quadratic)