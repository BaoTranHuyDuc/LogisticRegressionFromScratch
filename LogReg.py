#import required libraries

import numpy as np
import pandas as pd
import math
import matplotlib.pyplot as plt

#read_data
data = pd.read_csv("https://raw.githubusercontent.com/BaoTranHuyDuc/LogisticRegressionFromScratch/main/diabetes2.csv")

#separate the data into X(the input) and y(the output)
X = data.drop(columns = ["Outcome"], axis = 1)
y = data.Outcome

#apply standard scaling using the formula x' = (x - mean) / standard deviation
X_scaled = (X - np.mean(X))/np.std(X)

#split to train and test sets
X_train = X_scaled.iloc[:612]
X_test = X_scaled.iloc[612:768]
y_train = y.iloc[:612]
y_test = y.iloc[612:768]

# define the sigmoid function
def sigmoid(z):
  result = 1/ (1 + math.e ** -(z))
  return result
  
def fit(X, y, iterations, learning_rate, regularization_term):
  #m will be the number of observations, and n will be the number of features
  m, n = X.shape

  #initialize w and b to be 0 at first (it can be any starting value and I choose 0)

  #each feature will have its own unique weight (w) and so we will initialize w as an array
  w = np.zeros(n)
  #b is a constant so it only needs to be one nuymber
  b = 0

  #use a loop so that we can run gradient descend as many times as we want
  for i in range(iterations):
    # the np.dot function allows us to multiply each feature from each observation with its respective weight
    #without the dot function, we will have to use a for loop
    predicted_y = sigmoid(np.dot(X, w) + b)

    #this is the current cost (how much the predicted value strays from the true value)
    #I added the regularization term to avoid overfitting
    cost = (1/m)*np.sum((-y) * np.log(predicted_y) - (1-y)*np.log(1-predicted_y)) + (regularization_term / m) * np.sum(w ** 2)

    #this is the gradient descent part
    #I continously update w and b until the cost is at a minimal
    #we use the dot function at dw because we want to multiply the predicted_y - y by the all values of a feature in each iteration
    #we have to transpose the X because otherwise, predicted_y - y will be multiplied by all features of a specific observation in each iteration instead
    dw = (1 / m) * np.dot(X.T, (predicted_y-y)) 
    db = (1 / m) * np.sum(predicted_y - y)

    #here we update the weights and bias
    #I added the regulization calculation to w to do regulization
    w = w - learning_rate * dw - (regularization_term / m) * w
    b = b - learning_rate * db
    
    #this prints the cost every 100 cycles so we can see if the gradient descent is working
    if i % 100 == 0:
      print(cost)
  return w, b
  
def predict(X, y, w, b):
  #this calculates the predicted y values
  predicted_y = sigmoid(np.dot(X, w) +b)
  
  #this segment takes the y value and then classify it to either class 1 (Yes) or class 0 (No)
  predictions = []
  results = []
  for i in predicted_y:
    if i > 0.5:
      predictions.append(1)
    if i < 0.5:
      predictions.append(0)
  
  #this compares the predicted y classes with the actual y classes
  #this outputs 1 if it matches and 0 if it doesn't match
  for i in range(len(y_test)):
    if y_test.values[i] == predictions[i]:
      results.append(1)
    else:
      results.append(0)

  #this calculates the accuracy by taking the mean of the results list
  print("Accuracy is: " + str(sum(results)/len(results)))
  
  
print("The cost after every 100 iterations: ")
w, b = fit(X_train, y_train, 1000, 0.001, 0.5)
print("\n")
print("The weights are: " + str(w) + "\n")
print("The bias is: " + str(b)+ "\n")
predict(X_test, y_test, w, b)
