import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.optimize as opt

class LogisticRegression:
    
    def __init__(self):
        pass

    def _sigmoid(self, z):
        return 1 / (1+np.exp(-z))
    
    def _cost(self, thetav, Xv, yv):
        thetav = np.matrix(thetav)
        Xv = np.matrix(Xv)
        yv = np.matrix(yv)
        first = np.multiply(-yv, np.log(self._sigmoid(Xv * thetav.T)))
        second = np.multiply((1-yv), np.log(1-self._sigmoid(Xv * thetav.T)))
        return np.sum(first - second) / len(Xv)
    
    def _gradient(self, thetav, Xv, yv):
        thetav = np.matrix(thetav)
        Xv = np.matrix(Xv)
        yv = np.matrix(yv)
        
        parameters = int(thetav.ravel().shape[1])
        grad = np.zeros(parameters)
        
        error = self._sigmoid(Xv * thetav.T) - yv
        
        for i in range(parameters):
            term = np.multiply(error, Xv[:, i])
            grad[i] = np.sum(term) / len(Xv)
        
        return grad
    
    def fit(self, X, y):
        theta = np.zeros(X.shape[1])
        self.result = opt.fmin_tnc(func=self._cost, x0=theta, fprime=self._gradient, args=(X, y), disp=False)
    
    def predict(self, X):
        theta = np.matrix(self.result[0])
        probability = self._sigmoid(X * theta.T)
        return [1 if x >= 0.5 else 0 for x in probability]

# Get the data
data = pd.read_csv('ex2data1.csv', header=None, names=["Exam 1", "Exam 2", 'Admitted'])

# Split the data to success and failure samples
positive = data[ data.Admitted == 1 ]
negative = data[ data.Admitted == 0 ]

# Plot the data
fig, ax = plt.subplots(figsize=(5, 5))
ax.scatter(positive['Exam 1'], positive['Exam 2'], s=50, marker='o', label='Admitted')
ax.scatter(negative['Exam 1'], negative['Exam 2'], s=50, marker='o', color='red', label='Not Admitted')
ax.legend(loc=3)
ax.set_xlabel("Exam 1 Score")
ax.set_ylabel("Exam 2 Score")
plt.show()

# Adding 1s to the data, such that x0 = 1
data.insert(0, 'Ones', 1)

# Split the data to samples and their targets
cols = data.shape[1]
X = data.iloc[:, 0:cols-1].values
y = data.iloc[:, cols-1:].values

# Train the model, then predict
logistic_regression = LogisticRegression()
logistic_regression.fit(X, y)
print("We predicted: " + str(logistic_regression.predict(np.array([1, 34, 78]))))