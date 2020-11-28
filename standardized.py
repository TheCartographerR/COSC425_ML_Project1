#   Ryan Pauly
#   CS 425 - Intro to Machine Learning
#   Standardized
#
#   A standardized version of the original project1.
#
########################################################################################################################
#   Imports
from typing import List
from pandas import DataFrame
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import math
from math import sqrt

myData = pd.read_csv("shuffledData.csv")

print(myData)

#   Next we need to establish the matrices.
#   We'll create a matrix X which will have an all 1 column for the value we're trying to estimate (mpg).
#   The rest of the columns will be the other parameters provided in our .data
#   We'll assign the miles per gallon column to matrix Y

########################################################################################################################
#
#   TRAINING DATA
#


#   Standardizes the independent variables.
#   myData.iloc[:, 1:8] = (myData.iloc[:, 1:8] - myData.iloc[:, 1:8].mean()) / myData.iloc[:, 1:8].std()

trainX = myData.iloc[:318, 1:8]                             #   Create a training matrix of the independent variables

standard_trainX = (trainX - np.mean(trainX)) / np.std(trainX)

#   Create a matrix of all 1's with respect to standard_trainX
#   Will concatenate a column of all 1's into standard_trainX.
oneColumn = np.ones([standard_trainX.shape[0], 1])
standard_trainX = np.concatenate((oneColumn, standard_trainX), axis=1)

print("Training X matrix: \n")
print(standard_trainX)

#   Next we'll put the miles per gallon column into a matrix Y
trainY = myData.iloc[:318, 0:1]                             #   rows of MPG (column 0 in our .data)
print("\n Training Y matrix: \n")
print(trainY)


#   Calculate coefficients

tempC = np.dot(standard_trainX.T, standard_trainX)
matrix_C = np.dot(np.linalg.inv(tempC), np.dot(standard_trainX.T, trainY))

print("Matrix_C = \n")
print(matrix_C)

########################################################################################################################
#
#   TEST AND PLOT
#

test_X = myData.iloc[318:, 1:8]
print("\n test_X = \n", test_X)

standard_testX = (test_X - np.mean(test_X)) / np.std(test_X)

oneColumn = np.ones([standard_testX.shape[0], 1])
standard_testX = np.concatenate((oneColumn, standard_testX), axis=1)

real_Y = myData.iloc[318:, 0:1]                             #   Grabbing the actual Y values from the shuffled rows
test_Y = np.dot(standard_testX, matrix_C)                           #   Calculating our estimate MPG [Y] from coefficients

#   Calculating the mean squared error and root mean squared error:
mse = (np.square(real_Y - test_Y)).mean(axis=1)
rms = np.sqrt(mse)

avg_rms = np.average(rms)
print("Standardized: Avg_rms = ", avg_rms)

#   Plotting:
#   NOTE: NEED TO FOR-LOOP TO MAKE GRAPHS FOR EACH INDEPENDENT VARIABLE X

axis_X_list = ["Cylinders", "Displacement", "Horsepower", "Weight", "Acceleration", "Model Year", "Origin", "Car Name"]

for graphing in range(7):
    plt.scatter(test_X.iloc[:, graphing], test_Y, c='blue', marker='o', label="Estimated MPG vs " + axis_X_list[graphing])
    plt.scatter(test_X.iloc[:, graphing], real_Y, c='grey', marker='x', label="Actual MPG vs " + axis_X_list[graphing])
    plt.xlabel("Standardized " + axis_X_list[graphing], fontsize=16)
    plt.ylabel('Miles Per Gallon', fontsize=16)
    plt.title("Standardized: \n Miles Per Gallon vs " + axis_X_list[graphing], fontsize=18)
    plt.legend()
    plt.show()

#   Plotting Root Mean Squared Error
for graphing in range(7):
    plt.scatter(test_X.iloc[:, graphing], rms, c='red', marker='o', label="RMSE MPG vs" + axis_X_list[graphing])
    plt.xlabel("Standardized " + axis_X_list[graphing], fontsize=16)
    plt.ylabel('RMSE MPG ', fontsize=16)
    plt.title("Standardized: \n RMSE MPG vs " + axis_X_list[graphing], fontsize=20)
    plt.legend()
    plt.show()
