import pandas as pd
import numpy as np

from data_processing.rotate_data import Rotate_Data

filename = 'C:/Users/ishan/Desktop/Uni Files/Year 3/Labs/Jupiter Data 24 01 2025 (14 03 25).xlsx'

JupXI = pd.read_excel(filename, sheet_name='IoData', usecols='A', dtype=float).to_numpy().flatten()
JupYI = pd.read_excel(filename, sheet_name='IoData', usecols='B', dtype=float).to_numpy().flatten()
IoX = pd.read_excel(filename, sheet_name='IoData', usecols='C', dtype=float).to_numpy().flatten()
IoY = pd.read_excel(filename, sheet_name='IoData', usecols='D', dtype=float).to_numpy().flatten()
thetaI = pd.read_excel(filename, sheet_name='IoData', usecols='E', dtype=float).to_numpy().flatten()
scaleI = pd.read_excel(filename, sheet_name='IoData', usecols='F', dtype=float).to_numpy().flatten()
flipI = pd.read_excel(filename, sheet_name='IoData', usecols='G', dtype=bool).to_numpy().flatten()
JupErrXI = pd.read_excel(filename, sheet_name='IoData', usecols='H', dtype=float).to_numpy().flatten()
JupErrYI = pd.read_excel(filename, sheet_name='IoData', usecols='I', dtype=float).to_numpy().flatten()
Time_minsI = pd.read_excel(filename, sheet_name='IoData', usecols='N', dtype=float).to_numpy().flatten()

JupXE = pd.read_excel(filename, sheet_name='EuroData', usecols='A', dtype=float).to_numpy().flatten()
JupYE = pd.read_excel(filename, sheet_name='EuroData', usecols='B', dtype=float).to_numpy().flatten()
EuroX = pd.read_excel(filename, sheet_name='EuroData', usecols='C', dtype=float).to_numpy().flatten()
EuroY = pd.read_excel(filename, sheet_name='EuroData', usecols='D', dtype=float).to_numpy().flatten()
thetaE = pd.read_excel(filename, sheet_name='EuroData', usecols='E', dtype=float).to_numpy().flatten()
scaleE = pd.read_excel(filename, sheet_name='EuroData', usecols='F', dtype=float).to_numpy().flatten()
flipE = pd.read_excel(filename, sheet_name='EuroData', usecols='G', dtype=bool).to_numpy().flatten()
JupErrXE = pd.read_excel(filename, sheet_name='EuroData', usecols='H', dtype=float).to_numpy().flatten()
JupErrYE = pd.read_excel(filename, sheet_name='EuroData', usecols='I', dtype=float).to_numpy().flatten()
Time_minsE = pd.read_excel(filename, sheet_name='EuroData', usecols='N', dtype=float).to_numpy().flatten()

JupXG = pd.read_excel(filename, sheet_name='GanyData', usecols='A', dtype=float).to_numpy().flatten()
JupYG = pd.read_excel(filename, sheet_name='GanyData', usecols='B', dtype=float).to_numpy().flatten()
GanyX = pd.read_excel(filename, sheet_name='GanyData', usecols='C', dtype=float).to_numpy().flatten()
GanyY = pd.read_excel(filename, sheet_name='GanyData', usecols='D', dtype=float).to_numpy().flatten()
thetaG = pd.read_excel(filename, sheet_name='GanyData', usecols='E', dtype=float).to_numpy().flatten()
scaleG = pd.read_excel(filename, sheet_name='GanyData', usecols='F', dtype=float).to_numpy().flatten()
flipG = pd.read_excel(filename, sheet_name='GanyData', usecols='G', dtype=bool).to_numpy().flatten()
JupErrXG = pd.read_excel(filename, sheet_name='GanyData', usecols='H', dtype=float).to_numpy().flatten()
JupErrYG = pd.read_excel(filename, sheet_name='GanyData', usecols='I', dtype=float).to_numpy().flatten()
Time_minsG = pd.read_excel(filename, sheet_name='GanyData', usecols='N', dtype=float).to_numpy().flatten()

JupXC = pd.read_excel(filename, sheet_name='CalliData', usecols='A', dtype=float).to_numpy().flatten()
JupYC = pd.read_excel(filename, sheet_name='CalliData', usecols='B', dtype=float).to_numpy().flatten()
CalliX = pd.read_excel(filename, sheet_name='CalliData', usecols='C', dtype=float).to_numpy().flatten()
CalliY = pd.read_excel(filename, sheet_name='CalliData', usecols='D', dtype=float).to_numpy().flatten()
thetaC = pd.read_excel(filename, sheet_name='CalliData', usecols='E', dtype=float).to_numpy().flatten()
scaleC = pd.read_excel(filename, sheet_name='CalliData', usecols='F', dtype=float).to_numpy().flatten()
flipC = pd.read_excel(filename, sheet_name='CalliData', usecols='G', dtype=bool).to_numpy().flatten()
JupErrXC = pd.read_excel(filename, sheet_name='CalliData', usecols='H', dtype=float).to_numpy().flatten()
JupErrYC = pd.read_excel(filename, sheet_name='CalliData', usecols='I', dtype=float).to_numpy().flatten()
Time_minsC = pd.read_excel(filename, sheet_name='CalliData', usecols='N', dtype=float).to_numpy().flatten()

Data_stackI = np.column_stack([JupXI, JupYI, IoX, IoY, Time_minsI])
Data_stackE = np.column_stack([JupXE, JupYE, EuroX, EuroY, Time_minsE])
Data_stackG = np.column_stack([JupXG, JupYG, GanyX, GanyY, Time_minsG])
Data_stackC = np.column_stack([JupXC, JupYC, CalliX, CalliY, Time_minsC])

# Process Data
x_dataI, y_dataI = Rotate_Data(JupXI, JupYI, IoX, IoY, thetaI, scaleI, flipI)
x_dataE, y_dataE = Rotate_Data(JupXE, JupYE, EuroX, EuroY, thetaE, scaleE, flipE)
x_dataG, y_dataG = Rotate_Data(JupXG, JupYG, GanyX, GanyY, thetaG, scaleG, flipG)
x_dataC, y_dataC = Rotate_Data(JupXC, JupYC, CalliX, CalliY, thetaC, scaleC, flipC)

x_dataI = x_dataI.flatten()
y_dataI = y_dataI.flatten()
Time_minsI = Time_minsI.flatten()

x_dataE = x_dataE.flatten()
y_dataE = y_dataE.flatten()
Time_minsE = Time_minsE.flatten()

x_dataG = x_dataG.flatten()
y_dataG = y_dataG.flatten()
Time_minsG = Time_minsG.flatten()

x_dataC = x_dataC.flatten()
y_dataC = y_dataC.flatten()
Time_minsC = Time_minsC.flatten()

initial_valuesIx = np.array([130, 1.769, 0])
initial_valuesEx = np.array([200, 3.551, 0])
initial_valuesGx = np.array([300, 7.154, 0])
initial_valuesCx = np.array([500, 16.689, 0])

initial_valuesIy = np.array([20, 1.769, 0])
initial_valuesEy = np.array([25, 3.551, 0])
initial_valuesGy = np.array([40, 7.154, 0])
initial_valuesCy = np.array([50, 16.689, 0])

Io_Data = [x_dataI, y_dataI, Time_minsI, initial_valuesIx, initial_valuesIy]
Europa_data = [x_dataE, y_dataE, Time_minsE, initial_valuesEx, initial_valuesEy]
Ganymede_data = [x_dataG, y_dataG, Time_minsG, initial_valuesGx, initial_valuesGy]
Callisto_data = [x_dataC, y_dataC, Time_minsC, initial_valuesCx, initial_valuesCy]

## Index: IO = 1, Euro = 2 , Gany = 3, Calli = 4  (to be consistant with other data sheet)
compiled_data = [[],Io_Data, Europa_data, Ganymede_data, Callisto_data]