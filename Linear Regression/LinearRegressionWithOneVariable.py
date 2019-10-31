import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

class LinearRegressionWithOneVariable:
    def __init__(self):
        pass
        
    # Compute the cost function
    def computeCostFunction(self, X, y, theta):
        return np.sum(np.power(((X * theta.T) - y), 2)) / (2 * len(X))
    
    # Gradient Descent algorithm to minimize the cost function
    def gradientDescent(self, X, y, theta, alpha, iters):
        # The randomly choosed theta parametes, and the goal is to minimize 
        # the cost function made by them
        temp = np.matrix(np.zeros(theta.shape)) # [0, 0]
        parameters = int(theta.ravel().shape[1]) # 2 the number of thetas
        cost = np.zeros(iters) # An array that holds the cost function values
        
        for i in range(iters): # iterate the number of epochs
            error = (X * theta.T) - y
            
            for j in range(parameters): # 0, 1
                term = np.multiply(error, X[:, j])
                temp[0, j] = theta[0, j] - (alpha / len(X)) * np.sum(term)
            
            theta = temp
            cost[i] = self.computeCostFunction(X, y, theta)
        
        return theta, cost
    
    def fit(self, X, y, alpha=0.01, epochs=1000):
        X.insert(0, 'Ones', 1)
        self.theta = np.matrix(np.array([0, 0]))
        self.new_theta, self.cost = self.gradientDescent(np.matrix(X.values),
                                                         np.matrix(y.values), 
                                                         self.theta, alpha,epochs)
    
    def predict(self, x):
        return self.new_theta[0, 0] + self.new_theta[0, 1] * x
    
    
""" IN THE PROGRAM """

# Help displaying information
def printCustom(message, item):
    print(message + ' = \n')
    print(item)
    print('**********************************')

# Importing data 
data = pd.read_csv('ex1data1.csv', header=None, names=['Population', 'Profit'])

# Plotting the imported data
data.plot(kind='scatter', x='Population', y='Profit', figsize=(5, 5))
plt.show()

# Seperate the features from the target column
cols = data.shape[1]
X = data.iloc[:, 0:cols-1]
y = data.iloc[:, cols-1:]
        
model = LinearRegressionWithOneVariable()
model.fit(X, y, 0.04, 10000)
print(model.predict(20.0))
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        