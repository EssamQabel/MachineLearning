import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


class LinearRegressionWithMultipleVariable:
    def __init__(self):
        self.theta = None
        self.new_theta = None
        self.cost = None

    # Compute the cost function
    def compute_cost_function(self, X, y, theta):
        return np.sum(np.power(((X * theta.T) - y), 2)) / (2 * len(X))

    # Gradient Descent algorithm to minimize the cost function
    def gradient_descent(self, X, y, theta, alpha, iters):
        # The randomly chosen theta parameters, and the goal is to minimize
        # the cost function made by them
        temp = np.matrix(np.zeros(theta.shape))  # [0, 0]
        parameters = int(theta.ravel().shape[1])  # 2 the number of thetas
        cost = np.zeros(iters)  # An array that holds the cost function values

        for i in range(iters):  # iterate the number of epochs
            error = (X * theta.T) - y

            for j in range(parameters):  # 0, 1
                term = np.multiply(error, X[:, j])
                temp[0, j] = theta[0, j] - (alpha / len(X)) * np.sum(term)

            theta = temp
            cost[i] = self.compute_cost_function(X, y, theta)

        return theta, cost

    def fit(self, X, y, alpha=0.01, epochs=1000):
        X.insert(0, 'Ones', 1)
        self.theta = np.matrix(np.zeros(X.shape[1]))
        self.new_theta, self.cost = self.gradient_descent(np.matrix(X.values),
                                                         np.matrix(y.values),
                                                         self.theta, alpha, epochs)

    def predict(self, x):
        np.insert(x, 0, 1)
        return self.new_theta.T * x


# get the data
data = pd.read_csv('ex1data2.csv', header=None, names=['Size', 'Bedrooms', 'Price'])
# rescaling the data "normalization"
data = (data - data.mean()) / data.std()
# separate X from y
cols = data.shape[1]
X = data.iloc[:, 0:cols - 1]
y = data.iloc[:, cols - 1:cols]

alpha = 0.1
epochs = 100
model = LinearRegressionWithMultipleVariable()
model.fit(X, y, alpha, epochs)
print(model.predict(np.array([2000, 3])))
