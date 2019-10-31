import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Help displaying information
def printCustom(message, item):
    print(message + ' = \n')
    print(item)
    print('**********************************')

# Importing data 
data = pd.read_csv('ex1data1.csv', header=None, names=['Population', 'Profit'])
printCustom('data', data.head(10))
printCustom('data.describe', data.describe())

# Plotting the imported data
data.plot(kind='scatter', x='Population', y='Profit', figsize=(5, 5))
plt.show()

# Add the x0 column for mathematical reasons
data.insert(0, 'Ones', 1)
printCustom('new data', data.head(10))

# Seperate the features from the target column
cols = data.shape[1]
X = data.iloc[:, 0:cols-1]
y = data.iloc[:, cols-1:]

# Convert from data frames to numpy matrices
X = np.matrix(X.values)
y = np.matrix(y.values)
theta = np.matrix(np.array([0, 0]))

"""
printCustom('X matrix', X)
printCustom('X.shape', X.shape)
printCustom('y matrix', y)
printCustom('y.shape', y.shape)
printCustom('theta matrix', theta)
printCustom('theta.shape', theta.shape)
"""

# Compute the cost function
def computeCostFunction(X, y, theta):
    return np.sum(np.power(((X * theta.T) - y), 2)) / (2 * len(X))

printCustom('first cost function', computeCostFunction(X, y, theta))

# Gradient Descent algorithm to minimize the cost function
def gradientDescent(X, y, theta, alpha, iters):
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
        cost[i] = computeCostFunction(X, y, theta)
    
    return theta, cost

alpha = 0.01
iters = 1700

new_theta, cost = gradientDescent(X, y, theta, alpha, iters)
printCustom('cost after first iteration', cost[0])
printCustom('minimue cost', computeCostFunction(X, y, new_theta))

# Get the best fitting line
population = np.linspace(data['Population'].min(), data['Population'].max(), 100)
function_of_population = new_theta[0, 0] + new_theta[0, 1] * population

# Plot the data, and the prediction (best fit line)
fig, ax = plt.subplots(figsize=(5, 5))
ax.plot(population, function_of_population, 'r', label='Prediction')
ax.scatter(data.Population, data.Profit, label='Training Data')
ax.legend(loc=2)
ax.set_xlabel('Population')
ax.set_ylabel('Profit')
ax.set_title('Predicted Profit vs. Population Size')

# Draw error graph
fig, ax = plt.subplots(figsize=(5, 5))
ax.plot(np.arange(iters), cost, 'r')
ax.set_xlabel('Iterations')
ax.set_ylabel('Cost')
ax.set_title('Error vs. Training Epoch')















