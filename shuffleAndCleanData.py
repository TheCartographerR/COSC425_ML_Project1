#   Ryan Pauly
#   CS 425 - Intro to Machine Learning
#   ShuffleAndCleanData
#
#   File I/O and shuffling the rows around for training and testing to have a more fairly distributed and diverse
#       of rows and row content.
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

file = open("auto-mpg.data", 'r')

data = []                   #   Our Data array of parsed rows from the .data file
horsePowerSum = 0.0         #   Summing variable of all valid horse power values from .data
totalRows = 0               #   Total rows with valid horse power values.
avgHorsePower = 0.0         #   The variable for average horse power we'll use to replace invalid horse power cells.

for row in file:
    row = row.replace('\n', '')
    rowData = row.split(None, 8)

    #   We will replace '?' values with the average of the entire column (horse power).
    #   To do that we need to compute the sum of all the horse power provided.
    if rowData[3] != '?':
        totalRows += 1                                      #   Increment totalRows, we need this for the average
        horsePowerSum = horsePowerSum + float(rowData[3])   #   Summing all available horse power values
    del rowData[8]

    data.append(rowData)
file.close()

#   DATA CLEAN-UP

#   Next we'll calculate the average horse power from the collected sum and counted valid rows.

avgHorsePower = round(horsePowerSum / totalRows, 2)         #   avgHorsePower is about 104.47

for rows in data:
    if rows[3] == '?':
        rows[3] = avgHorsePower     #   Replacing the invalid horse power cells with the computed avg from valid data

    #   If its a string we want to make it a float type for data prep
    length = len(rows)
    for items in range(length):
        rows[items] = float(rows[items])

#   First we'll use pandas to put our data into a DataFrame for data computation
myData = DataFrame.from_records(data)

#   SHUFFLE DATA

myData = myData.sample(frac=1).reset_index(drop=True)       #   Shuffles the rows of the dataframe.
np.random.shuffle(myData.values)

#   SAVE .CSV Data for consistent results:
np.savetxt("shuffledData.csv", myData, delimiter=",")

print(myData)
