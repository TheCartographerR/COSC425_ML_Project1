#   Ryan Pauly
#   CS 425 - Intro to Machine Learning
#   Project 1
#
#   Project Description:
#
#       Use a Multi-variate Linear Regression to predict mpg (miles per gallon)
#       given the other 7 numeric parameters.
#
########################################################################################################################

#   Imports
from typing import List
from pandas import DataFrame
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import math

myData = pd.read_csv("shuffledData.csv")

print(myData)   #   Showing that the data was normalized.

#   Next we need to establish the matrices.
#   We'll create a matrix X which will have an all 1 column for the value we're trying to estimate (mpg).
#   The rest of the columns will be the other parameters provided in our .data
#   We'll assign the miles per gallon column to matrix Y

########################################################################################################################
#
#   TRAINING DATA
#

trainX = myData.iloc[:318, 1:8]                             #   Create a training matrix of the independent variables

oneColumn = np.ones([trainX.shape[0], 1])                   #   Create a matrix of all 1's with respect to trainX
trainX = np.concatenate((oneColumn, trainX), axis=1)        #   Will concatenate a column of all 1's into trainX.

print("Training X matrix: \n", trainX)

#   Next we'll put the miles per gallon column into a matrix Y
trainY = myData.iloc[:318, 0:1]                              #   rows of MPG (column 0 in our .data)
print("\n Training Y matrix: \n", trainY)

#   Calculate coefficients

tempC = np.dot(trainX.T, trainX)
matrix_C = np.dot(np.linalg.inv(tempC), np.dot(trainX.T, trainY))

print("Matrix_C = \n", matrix_C)

########################################################################################################################
#
#   TEST AND PLOT
#

test_X = myData.iloc[318:, 1:8]

oneColumn = np.ones([test_X.shape[0], 1])
test_X = np.concatenate((oneColumn, test_X), axis=1)

real_Y = myData.iloc[318:, 0:1]                             #   Grabbing the actual Y values from the shuffled rows
test_Y = np.dot(test_X, matrix_C)                           #   Calculating our estimate MPG [Y] from coefficients

#   Calculating the mean squared error and the root mean squared error:
mse = (np.square(real_Y - test_Y)).mean(axis=1)
rms = np.sqrt(mse)
print("\n RMSE = \n", rms)
avg_rms = np.average(rms)
print("Non-Standardized: Avg_rms = ", avg_rms)

#print("\n TEST_X = \n")
#print(test_X)
#   Avg Percent Difference:
#
#percentDifference = ((abs(np.average(real_Y) - np.average(test_Y))) / (np.average(real_Y) + np.average(test_Y)) / 2) * 100
#print("\n Average Percent Difference between actual and estimate: ", percentDifference)
#
#   Avg percent Error
#
#percentError = ((abs(np.average(real_Y) - np.average(test_Y))) / (np.average(real_Y))) * 100
#print("\n Average Percent Error between actual and estimate: ", percentError)

#   Plotting our estimate and real vs each independent variable.
axis_X_list = ["Cylinders", "Displacement", "Horsepower", "Weight", "Acceleration", "Model Year", "Origin", "Car Name"]

for graphing in range(7):
    plt.scatter(test_X[:, graphing+1], test_Y, c='blue', marker='o', label="Estimated MPG vs " + axis_X_list[graphing])
    plt.scatter(test_X[:, graphing+1], real_Y, c='grey', marker='x', label="Actual MPG vs " + axis_X_list[graphing])
    plt.xlabel(axis_X_list[graphing], fontsize=16)
    plt.ylabel('Miles Per Gallon', fontsize=16)
    plt.title("Miles Per Gallon vs " + axis_X_list[graphing], fontsize=20)
    plt.legend()
    plt.show()

#   Plotting Root Mean Squared Error
for graphing in range(7):
    plt.scatter(test_X[:, graphing+1], rms, c='red', marker='o', label="RMSE MPG vs" + axis_X_list[graphing])
    plt.xlabel(axis_X_list[graphing], fontsize=16)
    plt.ylabel('RMSE MPG ', fontsize=16)
    plt.title("RMSE MPG vs " + axis_X_list[graphing], fontsize=20)
    plt.legend()
    plt.show()
