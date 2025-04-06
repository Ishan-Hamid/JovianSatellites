import numpy as np
import pandas as pd

### data ###

filename = 'C:/Users/ishan/Desktop/Uni Files/Year 3/Labs/Jupiter Data 24 01 2025 (14 03 25).xlsx'

#Comments on side correspond to index in compiled data file at the bottom#

JupX = pd.read_excel(filename, sheet_name='Sheet3', usecols = 'B', dtype=float).to_numpy().flatten() #0
JupY = pd.read_excel(filename, sheet_name='Sheet3', usecols = 'C', dtype=float).to_numpy().flatten()
IoX = pd.read_excel(filename, sheet_name='Sheet3', usecols = 'D', dtype=float).to_numpy().flatten() #1
IoY = pd.read_excel(filename, sheet_name='Sheet3', usecols = 'E', dtype=float).to_numpy().flatten()
EuroX = pd.read_excel(filename, sheet_name='Sheet3', usecols = 'F', dtype=float).to_numpy().flatten() #2
EuroY = pd.read_excel(filename, sheet_name='Sheet3', usecols = 'G', dtype=float).to_numpy().flatten()
GanyX = pd.read_excel(filename, sheet_name='Sheet3', usecols = 'J', dtype=float).to_numpy().flatten() #3
GanyY = pd.read_excel(filename, sheet_name='Sheet3', usecols = 'K', dtype=float).to_numpy().flatten()
CalliX = pd.read_excel(filename, sheet_name='Sheet3', usecols = 'H', dtype=float).to_numpy().flatten() #4
CalliY = pd.read_excel(filename, sheet_name='Sheet3', usecols = 'I', dtype=float).to_numpy().flatten()
theta = pd.read_excel(filename, sheet_name='Sheet3', usecols = 'L', dtype=float).to_numpy().flatten() #5
scale = pd.read_excel(filename, sheet_name='Sheet3', usecols = 'm', dtype=float).to_numpy().flatten() #6
flip = pd.read_excel(filename, sheet_name='Sheet3', usecols = 'n', dtype=bool).to_numpy().flatten()  #7
JupErrX = pd.read_excel(filename, sheet_name='Sheet3', usecols = 'O', dtype=float).to_numpy().flatten()
JupErrY = pd.read_excel(filename, sheet_name='Sheet3', usecols = 'P', dtype=float).to_numpy().flatten()
Time_mins = pd.read_excel(filename, sheet_name='Sheet3', usecols = 'X', dtype=float).to_numpy().flatten() #8

compile = [[JupX, JupY], [IoX, IoY], [EuroX, EuroY], [GanyX, GanyY], [CalliX, CalliY], theta, scale, flip, Time_mins]

if __name__ == "__main__":
    from data_processing.rotate_data import Rotate_Data
    import matplotlib.pyplot as plt

    X = Rotate_Data(JupX, JupY, IoX, IoY, theta, scale, flip)[0]
    Y = Rotate_Data(JupX, JupY, IoX, IoY, theta, scale, flip)[1]

    plt.scatter(X,Y)
    plt.show()

    print(compile[8])
    pass