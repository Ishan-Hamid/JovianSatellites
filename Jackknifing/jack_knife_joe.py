import numpy as np
import matplotlib.pyplot as plt
from scipy import optimize as opt
import pandas as pd
import astropy.stats as astats
from matplotlib import gridspec
from scipy.special import erfinv

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

def sinusoid(t, A, T, c):
    return A * np.sin((t + c) * (2 * np.pi) / T)

def model(x, *params):
    return sinusoid(x, params[0], params[1], params[2])

# Curve Fitting Function
def CurveFit(model, x, initial_values, Time):
    popt, cov = opt.curve_fit(
        model, Time / (24 * 60), x, sigma=np.ones(len(x)) * 0.5,
        absolute_sigma=True, p0=initial_values, check_finite=True, maxfev=50000
    )
    return popt, cov

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

initial_valuesIx = [130, 1.769, 0]
initial_valuesEx = [200, 3.551, 0]
initial_valuesGx = [300, 7.154, 0]
initial_valuesCx = [500, 16.689, 0]

initial_valuesIy = [20, 1.769, 0]
initial_valuesEy = [25, 3.551, 0]
initial_valuesGy = [40, 7.154, 0]
initial_valuesCy = [50, 16.689, 0]


def JackKnife(model, x, confidence_level, Time, initialx):
    JKx, JKP = astats.jackknife_resampling(x), astats.jackknife_resampling(Time)
    n = x.shape[0]
    Radius = np.zeros(len(JKx))
    Period = np.zeros(len(JKx))
    Start_angle = np.zeros(len(JKx))

    for i in range(n):
        popt, _ = opt.curve_fit(model, JKP[i] / (24 * 60), JKx[i], sigma=np.ones(len(JKx[i])),
                                p0=initialx, absolute_sigma=True, check_finite=True, maxfev=50000)
        Radius[i] = popt[0]
        Period[i] = popt[1]
        Start_angle[i] = popt[2]

    stat1 = CurveFit(model, x, initialx, Time)[0][0]
    stat2 = CurveFit(model, x, initialx, Time)[0][1]
    stat3 = CurveFit(model, x, initialx, Time)[0][2]

    mean1 = np.mean(Radius)
    mean2 = np.mean(Period)
    mean3 = np.mean(Start_angle)

    bias1 = (n - 1) * (mean1 - stat1)
    bias2 = (n - 1) * (mean2 - stat2)
    bias3 = (n - 1) * (mean3 - stat3)

    std1 = np.sqrt((n - 1) * np.mean((Radius - mean1) * (Radius - mean1), axis=0))
    std2 = np.sqrt((n - 1) * np.mean((Period - mean2) * (Period - mean2), axis=0))
    std3 = np.sqrt((n - 1) * np.mean((Start_angle - mean3) * (Start_angle - mean3), axis=0))

    estimate1 = stat1 - bias1
    estimate2 = stat2 - bias2
    estimate3 = stat3 - bias3

    z_score = np.sqrt(2.0) * erfinv(confidence_level)

    conf_interval1 = estimate1 + z_score * np.array((-std1, std1))
    conf_interval2 = estimate2 + z_score * np.array((-std2, std2))
    conf_interval3 = estimate3 + z_score * np.array((-std3, std3))

    estimate = np.array([estimate1, estimate2, estimate3])
    std = np.array([std1, std2, std3])

    return estimate, std


##### Used for phase plots #####
smooth_phase = np.linspace(0, 1, 300)


def time_phase(time, period):
    return (time / (24 * 60)) / period % 1


###### Fucntions for residuals ########
def residuals(x, modelval):
    return x - modelval


def normal_resid(x, modelval):
    return residuals(x, modelval) / np.std(residuals(x, modelval))


########################### 1D plots for x and y of each moon #############################

# My bad its so long, couldn't figure out how to make it shorter

############# Io x plot ####################

plt.figure(figsize=(8, 6))
gs = gridspec.GridSpec(2, 1, height_ratios=[3, 1], hspace=0)

ax1 = plt.subplot(
    gs[0])  # 0.8 is the confidence level (not sure what is should be set to yet)
ax1.errorbar(time_phase(Time_minsI, JackKnife(model, x_dataI, 0.8, Time_minsI, initial_valuesIx)[0][1]),
             # *JackKnife(model, x_dataI, 0.8, Time_minsI, initial_valuesIx)[0][1],
             x_dataI, yerr=0.5, marker='o', markersize=5, capsize=3, label='Data', ls='none',
             color='black')  # ^^^including above makes the y-axis be in terms of days instead of phase

ax1.plot(smooth_phase,  # *JackKnife(model, x_dataI, 0.8, Time_minsI, initial_valuesIx)[0][1],
         model(smooth_phase * JackKnife(model, x_dataI, 0.8, Time_minsI, initial_valuesIx)[0][1],
               JackKnife(model, x_dataI, 0.8, Time_minsI, initial_valuesIx)[0][0],
               JackKnife(model, x_dataI, 0.8, Time_minsI, initial_valuesIx)[0][1],
               JackKnife(model, x_dataI, 0.8, Time_minsI, initial_valuesIx)[0][2]),
         label='Fit', linewidth=2, color='orange')
ax1.set_ylabel('X (Arcsec)', fontsize=18)
ax1.legend(fontsize=12)
ax1.set_title('Io x', fontsize=18)
ax1.tick_params(axis='both', which='major', labelsize=14, width=2, length=7, top=True, right=True)
ax1.tick_params(axis='both', which='minor', labelsize=12, width=1, length=3, top=True, right=True)
ax1.minorticks_on()

ax2 = plt.subplot(gs[1], sharex=ax1)
ax2.axhline(0, color='orange', linestyle='--', linewidth=2)  # Reference line at zeroq
ax2.errorbar(time_phase(Time_minsI, JackKnife(model, x_dataI, 0.8, Time_minsI, initial_valuesIx)[0][1]),
             # *JackKnife(model, x_dataI, 0.8, Time_minsI, initial_valuesIx)[0][1],
             normal_resid(x_dataI, model(Time_minsI / (24 * 60),
                                         # time_phase(Time_minsI,JackKnife(model, x_dataI, 0.8, Time_minsI, initial_valuesIx)[0][1]),
                                         JackKnife(model, x_dataI, 0.8, Time_minsI, initial_valuesIx)[0][0],
                                         JackKnife(model, x_dataI, 0.8, Time_minsI, initial_valuesIx)[0][1],
                                         JackKnife(model, x_dataI, 0.8, Time_minsI, initial_valuesIx)[0][2])),
             marker='o', markersize=5, capsize=3, ls='none', color='black')

ax2.set_xlabel('Phase', fontsize=18)
ax2.set_ylabel('Norm Residuals', fontsize=18)
# ax2.set_ylim(-1, 1)
ax2.tick_params(axis='both', which='major', labelsize=16, width=2, length=7, right=True)
ax2.tick_params(axis='both', which='minor', labelsize=12, width=1, length=3, right=True)
ax2.minorticks_on()
# plt.savefig('figures/Iox.jpg', dpi=300, bbox_inches='tight')
plt.tight_layout()
# plt.show()


############# Io y plot ####################

plt.figure(figsize=(8, 6))
gs = gridspec.GridSpec(2, 1, height_ratios=[3, 1], hspace=0)

ax1 = plt.subplot(gs[0])
ax1.errorbar(time_phase(Time_minsI, JackKnife(model, y_dataI, 0.8, Time_minsI, initial_valuesIy)[0][1]),
             # *JackKnife(model, y_dataI, 0.8, Time_minsI, initial_valuesIy)[0][1],
             y_dataI, yerr=0.5, marker='o', markersize=5, capsize=3, label='Data', ls='none', color='black')

ax1.plot(smooth_phase,  # *JackKnife(model, y_dataI, 0.8, Time_minsI, initial_valuesIy)[0][1],
         model(smooth_phase * JackKnife(model, y_dataI, 0.8, Time_minsI, initial_valuesIy)[0][1],
               JackKnife(model, y_dataI, 0.8, Time_minsI, initial_valuesIy)[0][0],
               JackKnife(model, y_dataI, 0.8, Time_minsI, initial_valuesIy)[0][1],
               JackKnife(model, y_dataI, 0.8, Time_minsI, initial_valuesIy)[0][2]),
         label='Fit', linewidth=2, color='orange')
ax1.set_ylabel('Y (Arcsec)', fontsize=18)
ax1.legend(fontsize=12)
ax1.set_title('Io y', fontsize=18)
ax1.tick_params(axis='both', which='major', labelsize=14, width=2, length=7, top=True, right=True)
ax1.tick_params(axis='both', which='minor', labelsize=12, width=1, length=3, top=True, right=True)
ax1.minorticks_on()

ax2 = plt.subplot(gs[1], sharex=ax1)
ax2.axhline(0, color='orange', linestyle='--', linewidth=2)  # Reference line at zeroq
ax2.errorbar(time_phase(Time_minsI, JackKnife(model, y_dataI, 0.8, Time_minsI, initial_valuesIy)[0][1]),
             # *JackKnife(model, y_dataI, 0.8, Time_minsI, initial_valuesIy)[0][1],
             normal_resid(y_dataI, model(Time_minsI / (24 * 60),
                                         # (model, time_phase(Time_minsI, JackKnife(model, y_dataI, 0.8, Time_minsI, initial_valuesIy)[0][1]),
                                         JackKnife(model, y_dataI, 0.8, Time_minsI, initial_valuesIy)[0][0],
                                         JackKnife(model, y_dataI, 0.8, Time_minsI, initial_valuesIy)[0][1],
                                         JackKnife(model, y_dataI, 0.8, Time_minsI, initial_valuesIy)[0][2])),
             marker='o', markersize=5, capsize=3, ls='none', color='black')

ax2.set_xlabel('Phase', fontsize=18)
ax2.set_ylabel('Norm Residuals', fontsize=18)
# ax2.set_ylim(-1, 1)
ax2.tick_params(axis='both', which='major', labelsize=16, width=2, length=7, right=True)
ax2.tick_params(axis='both', which='minor', labelsize=12, width=1, length=3, right=True)
ax2.minorticks_on()
# plt.savefig('figures/Iox.jpg', dpi=300, bbox_inches='tight')
plt.tight_layout()
# plt.show()


############# Europa x plot ####################

plt.figure(figsize=(8, 6))
gs = gridspec.GridSpec(2, 1, height_ratios=[3, 1], hspace=0)

ax1 = plt.subplot(gs[0])
ax1.errorbar(time_phase(Time_minsE, JackKnife(model, x_dataE, 0.8, Time_minsE, initial_valuesEx)[0][1]),
             # *JackKnife(model, x_dataE, 0.8, Time_minsE, initial_valuesEx)[0][1],
             x_dataE, yerr=0.5, marker='o', markersize=5, capsize=3, label='Data', ls='none', color='black')

ax1.plot(smooth_phase,  # *JackKnife(model, x_dataE, 0.8, Time_minsE, initial_valuesEx)[0][1],
         model(smooth_phase * JackKnife(model, x_dataE, 0.8, Time_minsE, initial_valuesEx)[0][1],
               JackKnife(model, x_dataE, 0.8, Time_minsE, initial_valuesEx)[0][0],
               JackKnife(model, x_dataE, 0.8, Time_minsE, initial_valuesEx)[0][1],
               JackKnife(model, x_dataE, 0.8, Time_minsE, initial_valuesEx)[0][2]), label='Fit', linewidth=2,
         color='purple')
ax1.set_ylabel('X (Arcsec)', fontsize=16)
ax1.legend(fontsize=12)
# ax1.set_title('Europa x', fontsize=18)
ax1.tick_params(axis='both', which='major', labelsize=14, width=2, length=7, top=True, right=True)
ax1.tick_params(axis='both', which='minor', labelsize=12, width=1, length=3, top=True, right=True)
ax1.minorticks_on()

ax2 = plt.subplot(gs[1], sharex=ax1)
ax2.axhline(0, color='purple', linestyle='--', linewidth=2)  # Reference line at zeroq
ax2.errorbar(time_phase(Time_minsE, JackKnife(model, x_dataE, 0.8, Time_minsE, initial_valuesEx)[0][1]),
             # *JackKnife(model, x_dataE, 0.8, Time_minsE, initial_valuesEx)[0][1],
             normal_resid(x_dataE, model(Time_minsE / (24 * 60),
                                         # )(model, time_phase(Time_minsE, JackKnife(model, x_dataE, 0.8, Time_minsE, initial_valuesEx)[0][1]),
                                         JackKnife(model, x_dataE, 0.8, Time_minsE, initial_valuesEx)[0][0],
                                         JackKnife(model, x_dataE, 0.8, Time_minsE, initial_valuesEx)[0][1],
                                         JackKnife(model, x_dataE, 0.8, Time_minsE, initial_valuesEx)[0][2])),
             marker='o', markersize=5, capsize=3, ls='none', color='black')
ax2.set_xlabel('Phase', fontsize=16)
ax2.set_ylabel('Norm Residuals', fontsize=16)
# ax2.set_ylim(-1, 1)
ax2.tick_params(axis='both', which='major', labelsize=14, width=2, length=7, right=True)
ax2.tick_params(axis='both', which='minor', labelsize=12, width=1, length=3, right=True)
ax2.minorticks_on()
# plt.savefig('figures/Europax.jpg', dpi=300, bbox_inches='tight')
plt.tight_layout()
# plt.show()


############# Europa y plot ####################

plt.figure(figsize=(8, 6))
gs = gridspec.GridSpec(2, 1, height_ratios=[3, 1], hspace=0)

ax1 = plt.subplot(gs[0])
ax1.errorbar(time_phase(Time_minsE, JackKnife(model, y_dataE, 0.8, Time_minsE, initial_valuesEy)[0][1]),
             # *JackKnife(model, y_dataE, 0.8, Time_minsE, initial_valuesEy)[0][1],
             y_dataE, yerr=0.5, marker='o', markersize=5, capsize=3, label='Data', ls='none', color='black')

ax1.plot(smooth_phase,  # *JackKnife(model, y_dataE, 0.8, Time_minsE, initial_valuesEy)[0][1],
         model(smooth_phase * JackKnife(model, y_dataE, 0.8, Time_minsE, initial_valuesEy)[0][1],
               JackKnife(model, y_dataE, 0.8, Time_minsE, initial_valuesEy)[0][0],
               JackKnife(model, y_dataE, 0.8, Time_minsE, initial_valuesEy)[0][1],
               JackKnife(model, y_dataE, 0.8, Time_minsE, initial_valuesEy)[0][2]), label='Fit', linewidth=2,
         color='purple')
ax1.set_ylabel('Y (Arcsec)', fontsize=18)
ax1.legend(fontsize=12)
ax1.set_title('Europa y', fontsize=18)
ax1.tick_params(axis='both', which='major', labelsize=14, width=2, length=7, top=True, right=True)
ax1.tick_params(axis='both', which='minor', labelsize=12, width=1, length=3, top=True, right=True)
ax1.minorticks_on()

ax2 = plt.subplot(gs[1], sharex=ax1)
ax2.axhline(0, color='purple', linestyle='--', linewidth=2)  # Reference line at zeroq
ax2.errorbar(time_phase(Time_minsE, JackKnife(model, y_dataE, 0.8, Time_minsE, initial_valuesEy)[0][1]),
             # *JackKnife(model, y_dataE, 0.8, Time_minsE, initial_valuesEy)[0][1],
             normal_resid(y_dataE, model(Time_minsE / (24 * 60),
                                         # (model, time_phase(Time_minsE, JackKnife(model, y_dataE, 0.8, Time_minsE, initial_valuesEy)[0][1]),
                                         JackKnife(model, y_dataE, 0.8, Time_minsE, initial_valuesEy)[0][0],
                                         JackKnife(model, y_dataE, 0.8, Time_minsE, initial_valuesEy)[0][1],
                                         JackKnife(model, y_dataE, 0.8, Time_minsE, initial_valuesEy)[0][2])),
             marker='o', markersize=5, capsize=3, ls='none', color='black')
ax2.set_xlabel('Phase', fontsize=18)
ax2.set_ylabel('Norm Residuals', fontsize=18)
# ax2.set_ylim(-1, 1)
ax2.tick_params(axis='both', which='major', labelsize=16, width=2, length=7, right=True)
ax2.tick_params(axis='both', which='minor', labelsize=12, width=1, length=3, right=True)
ax2.minorticks_on()
# plt.savefig('figures/Europax.jpg', dpi=300, bbox_inches='tight')
plt.tight_layout()
# plt.show()


############# Ganymede x plot ####################

plt.figure(figsize=(8, 6))
gs = gridspec.GridSpec(2, 1, height_ratios=[3, 1], hspace=0)

ax1 = plt.subplot(gs[0])
ax1.errorbar(time_phase(Time_minsG, JackKnife(model, x_dataG, 0.8, Time_minsG, initial_valuesGx)[0][1]),
             # *JackKnife(model, x_dataG, 0.8, Time_minsG, initial_valuesGx)[0][1],
             x_dataG, yerr=0.5, marker='o', markersize=5, capsize=3, label='Data', ls='none', color='black')
ax1.plot(smooth_phase,  # *JackKnife(model, x_dataG, 0.8, Time_minsG, initial_valuesGx)[0][1],
         model(smooth_phase * JackKnife(model, x_dataG, 0.8, Time_minsG, initial_valuesGx)[0][1],
               JackKnife(model, x_dataG, 0.8, Time_minsG, initial_valuesGx)[0][0],
               JackKnife(model, x_dataG, 0.8, Time_minsG, initial_valuesGx)[0][1],
               JackKnife(model, x_dataG, 0.8, Time_minsG, initial_valuesGx)[0][2]), label='Fit', linewidth=2,
         color='green')
ax1.set_ylabel('X (Arcsec)', fontsize=18)
ax1.legend(fontsize=12)
ax1.set_title('Ganymede x', fontsize=18)
ax1.tick_params(axis='both', which='major', labelsize=14, width=2, length=7, top=True, right=True)
ax1.tick_params(axis='both', which='minor', labelsize=12, width=1, length=3, top=True, right=True)
ax1.minorticks_on()

ax2 = plt.subplot(gs[1], sharex=ax1)
ax2.axhline(0, color='green', linestyle='--', linewidth=2)  # Reference line at zeroq
ax2.errorbar(time_phase(Time_minsG, JackKnife(model, x_dataG, 0.8, Time_minsG, initial_valuesGx)[0][1]),
             # *JackKnife(model, x_dataG, 0.8, Time_minsG, initial_valuesGx)[0][1],
             normal_resid(x_dataG, model(Time_minsG / (24 * 60),
                                         # (model, time_phase(Time_minsG, JackKnife(model, x_dataG, 0.8, Time_minsG, initial_valuesGx)[0][1]),
                                         JackKnife(model, x_dataG, 0.8, Time_minsG, initial_valuesGx)[0][0],
                                         JackKnife(model, x_dataG, 0.8, Time_minsG, initial_valuesGx)[0][1],
                                         JackKnife(model, x_dataG, 0.8, Time_minsG, initial_valuesGx)[0][2])),
             marker='o', markersize=5, capsize=3, ls='none', color='black')
ax2.set_xlabel('Phase', fontsize=18)
ax2.set_ylabel('Norm Residuals', fontsize=18)
# ax2.set_ylim(-1, 1)
ax2.tick_params(axis='both', which='major', labelsize=16, width=2, length=7, right=True)
ax2.tick_params(axis='both', which='minor', labelsize=12, width=1, length=3, right=True)
ax2.minorticks_on()
# plt.savefig('figures/ganymedex.jpg', dpi=300, bbox_inches='tight')
plt.tight_layout()
# plt.show()


############# Ganymede y plot ####################

plt.figure(figsize=(8, 6))
gs = gridspec.GridSpec(2, 1, height_ratios=[3, 1], hspace=0)

ax1 = plt.subplot(gs[0])
ax1.errorbar(time_phase(Time_minsG, JackKnife(model, y_dataG, 0.8, Time_minsG, initial_valuesGy)[0][1]),
             # *JackKnife(model, y_dataG, 0.8, Time_minsG, initial_valuesGy)[0][1],
             y_dataG, yerr=0.5, marker='o', markersize=5, capsize=3, label='Data', ls='none', color='black')
ax1.plot(smooth_phase,  # *JackKnife(model, y_dataG, 0.8, Time_minsG, initial_valuesGy)[0][1],
         model(smooth_phase * JackKnife(model, y_dataG, 0.8, Time_minsG, initial_valuesGy)[0][1],
               JackKnife(model, y_dataG, 0.8, Time_minsG, initial_valuesGy)[0][0],
               JackKnife(model, y_dataG, 0.8, Time_minsG, initial_valuesGy)[0][1],
               JackKnife(model, y_dataG, 0.8, Time_minsG, initial_valuesGy)[0][2]), label='Fit', linewidth=2,
         color='green')
ax1.set_ylabel('Y (Arcsec)', fontsize=18)
ax1.legend(fontsize=12)
ax1.set_title('Ganymede y', fontsize=18)
ax1.tick_params(axis='both', which='major', labelsize=14, width=2, length=7, top=True, right=True)
ax1.tick_params(axis='both', which='minor', labelsize=12, width=1, length=3, top=True, right=True)
ax1.minorticks_on()

ax2 = plt.subplot(gs[1], sharex=ax1)
ax2.axhline(0, color='green', linestyle='--', linewidth=2)  # Reference line at zeroq
ax2.errorbar(time_phase(Time_minsG, JackKnife(model, y_dataG, 0.8, Time_minsG, initial_valuesGy)[0][1]),
             # *JackKnife(model, y_dataG, 0.8, Time_minsG, initial_valuesGy)[0][1],
             normal_resid(y_dataG, model(Time_minsG / (24 * 60),
                                         # (model, time_phase(Time_minsG, JackKnife(model, y_dataG, 0.8, Time_minsG, initial_valuesGy)[0][1]),
                                         JackKnife(model, y_dataG, 0.8, Time_minsG, initial_valuesGy)[0][0],
                                         JackKnife(model, y_dataG, 0.8, Time_minsG, initial_valuesGy)[0][1],
                                         JackKnife(model, y_dataG, 0.8, Time_minsG, initial_valuesGy)[0][2])),
             marker='o', markersize=5, capsize=3, ls='none', color='black')
ax2.set_xlabel('Phase', fontsize=18)
ax2.set_ylabel('Norm Residuals', fontsize=18)
# ax2.set_ylim(-1, 1)
ax2.tick_params(axis='both', which='major', labelsize=16, width=2, length=7, right=True)
ax2.tick_params(axis='both', which='minor', labelsize=12, width=1, length=3, right=True)
ax2.minorticks_on()
# plt.savefig('figures/ganymedex.jpg', dpi=300, bbox_inches='tight')
plt.tight_layout()
# plt.show()


############# Callisto x plot ####################

plt.figure(figsize=(8, 6))
gs = gridspec.GridSpec(2, 1, height_ratios=[3, 1], hspace=0)

ax1 = plt.subplot(gs[0])
ax1.errorbar(time_phase(Time_minsC, JackKnife(model, x_dataC, 0.8, Time_minsC, initial_valuesCx)[0][1]),
             # *JackKnife(model, x_dataC, 0.8, Time_minsC, initial_valuesCx)[0][1],
             x_dataC, yerr=0.5, marker='o', markersize=5, capsize=3, label='Data', ls='none', color='black')
ax1.plot(smooth_phase,  # *JackKnife(model, x_dataC, 0.8, Time_minsC, initial_valuesCx)[0][1],
         model(smooth_phase * JackKnife(model, x_dataC, 0.8, Time_minsC, initial_valuesCx)[0][1],
               JackKnife(model, x_dataC, 0.8, Time_minsC, initial_valuesCx)[0][0],
               JackKnife(model, x_dataC, 0.8, Time_minsC, initial_valuesCx)[0][1],
               JackKnife(model, x_dataC, 0.8, Time_minsC, initial_valuesCx)[0][2]), label='Fit', linewidth=2,
         color='Blue')
ax1.set_ylabel('X (Arcsec)', fontsize=18)
ax1.legend(fontsize=12)
ax1.set_title('Callisto x', fontsize=18)
ax1.tick_params(axis='both', which='major', labelsize=14, width=2, length=7, top=True, right=True)
ax1.tick_params(axis='both', which='minor', labelsize=12, width=1, length=3, top=True, right=True)
ax1.minorticks_on()

ax2 = plt.subplot(gs[1], sharex=ax1)
ax2.axhline(0, color='blue', linestyle='--', linewidth=2)  # Reference line at zeroq
ax2.errorbar(time_phase(Time_minsC, JackKnife(model, x_dataC, 0.8, Time_minsC, initial_valuesCx)[0][1]),
             # *JackKnife(model, x_dataC, 0.8, Time_minsC, initial_valuesCx)[0][1],
             normal_resid(x_dataC, model(Time_minsC / (24 * 60),
                                         # (model, time_phase(Time_minsC, JackKnife(model, x_dataC, 0.8, Time_minsC, initial_valuesCx)[0][1]),
                                         JackKnife(model, x_dataC, 0.8, Time_minsC, initial_valuesCx)[0][0],
                                         JackKnife(model, x_dataC, 0.8, Time_minsC, initial_valuesCx)[0][1],
                                         JackKnife(model, x_dataC, 0.8, Time_minsC, initial_valuesCx)[0][2])),
             marker='o', markersize=5, capsize=3, ls='none', color='black')
ax2.set_xlabel('Phase', fontsize=18)
ax2.set_ylabel('Norm Residuals', fontsize=18)
# ax2.set_ylim(-1, 1)
ax2.tick_params(axis='both', which='major', labelsize=16, width=2, length=7, right=True)
ax2.tick_params(axis='both', which='minor', labelsize=12, width=1, length=3, right=True)
ax2.minorticks_on()
# plt.savefig('figures/callistox.jpg', dpi=300, bbox_inches='tight')
plt.tight_layout()
# plt.show()


############# Callisto y plot ####################

plt.figure(figsize=(8, 6))
gs = gridspec.GridSpec(2, 1, height_ratios=[3, 1], hspace=0)

ax1 = plt.subplot(gs[0])
ax1.errorbar(time_phase(Time_minsC, JackKnife(model, y_dataC, 0.8, Time_minsC, initial_valuesCy)[0][1]),
             # *JackKnife(model, y_dataC, 0.8, Time_minsC, initial_valuesCy)[0][1],
             y_dataC, yerr=0.5, marker='o', markersize=5, capsize=3, label='Data', ls='none', color='black')
ax1.plot(smooth_phase,  # *JackKnife(model, y_dataC, 0.8, Time_minsC, initial_valuesCy)[0][1],
         model(smooth_phase * JackKnife(model, y_dataC, 0.8, Time_minsC, initial_valuesCy)[0][1],
               JackKnife(model, y_dataC, 0.8, Time_minsC, initial_valuesCy)[0][0],
               JackKnife(model, y_dataC, 0.8, Time_minsC, initial_valuesCy)[0][1],
               JackKnife(model, y_dataC, 0.8, Time_minsC, initial_valuesCy)[0][2]), label='Fit', linewidth=2,
         color='Blue')
ax1.set_ylabel('Y (Arcsec)', fontsize=18)
ax1.legend(fontsize=12)
ax1.set_title('Callisto y', fontsize=18)
ax1.tick_params(axis='both', which='major', labelsize=14, width=2, length=7, top=True, right=True)
ax1.tick_params(axis='both', which='minor', labelsize=12, width=1, length=3, top=True, right=True)
ax1.minorticks_on()

ax2 = plt.subplot(gs[1], sharex=ax1)
ax2.axhline(0, color='blue', linestyle='--', linewidth=2)  # Reference line at zeroq
ax2.errorbar(time_phase(Time_minsC, JackKnife(model, y_dataC, 0.8, Time_minsC, initial_valuesCy)[0][1]),
             # *JackKnife(model, y_dataC, 0.8, Time_minsC, initial_valuesCy)[0][1],
             normal_resid(y_dataC, model(Time_minsC / (24 * 60),
                                         # (model/(24*60), #time_phase(Time_minsC, JackKnife(model, y_dataC, 0.8, Time_minsC, initial_valuesCy)[0][1]),
                                         JackKnife(model, y_dataC, 0.8, Time_minsC, initial_valuesCy)[0][0],
                                         JackKnife(model, y_dataC, 0.8, Time_minsC, initial_valuesCy)[0][1],
                                         JackKnife(model, y_dataC, 0.8, Time_minsC, initial_valuesCy)[0][2])),
             marker='o', markersize=5, capsize=3, ls='none', color='black')
ax2.set_xlabel('Phase', fontsize=18)
ax2.set_ylabel('Norm Residuals', fontsize=18)
# ax2.set_ylim(-1, 1)
ax2.tick_params(axis='both', which='major', labelsize=16, width=2, length=7, right=True)
ax2.tick_params(axis='both', which='minor', labelsize=12, width=1, length=3, right=True)
ax2.minorticks_on()
# plt.savefig('figures/callistox.jpg', dpi=300, bbox_inches='tight')
plt.tight_layout()
plt.show()


#################### 2D Jackknife ####################


def ellipse_fit(x, y):
    # least squares fit of an ellipse using matrix eqn

    J = np.vstack([x ** 2, x * y, y ** 2, x, y]).T
    K = np.ones_like(x)
    JT = J.transpose()
    JTJ = np.dot(JT, J)
    invJTJ = np.linalg.inv(JTJ)
    vector = np.dot(invJTJ, np.dot(JT, K))

    return np.append(vector, -1)


def convert_to_physical(A, B, C, D, E, F):
    x0 = (2 * C * D - B * E) / (B ** 2 - 4 * A * C)
    y0 = (2 * A * E - B * D) / (B ** 2 - 4 * A * C)
    a = -(np.sqrt(2 * (A * E ** 2 + C * D ** 2 - B * D * E + (B ** 2 - 4 * A * C) * F) * (
                (A + C) + (np.sqrt((A - C) ** 2 + B ** 2))))) / (B ** 2 - 4 * A * C)
    b = -(np.sqrt(2 * (A * E ** 2 + C * D ** 2 - B * D * E + (B ** 2 - 4 * A * C) * F) * (
                (A + C) - (np.sqrt((A - C) ** 2 + B ** 2))))) / (B ** 2 - 4 * A * C)
    theta = np.arctan2(-B, C - A) / 2
    return x0, y0, a, b, theta


def model_ellipse(x, y, *params):
    return params[0] * x ** 2 + params[1] * x * y + params[2] * y ** 2 + params[3] * x + params[4] * y + params[5]


def JackKnife2D(model, x, y, confidence):
    JKx, JKy = astats.jackknife_resampling(x), astats.jackknife_resampling(y)
    n = x.shape[0]
    A = np.zeros(len(JKx))
    B = np.zeros(len(JKx))
    C = np.zeros(len(JKx))
    D = np.zeros(len(JKx))
    E = np.zeros(len(JKx))
    F = np.zeros(len(JKx))

    for i in range(n):
        fit = ellipse_fit(JKx[i], JKy[i])
        A[i] = fit[0]
        B[i] = fit[1]
        C[i] = fit[2]
        D[i] = fit[3]
        E[i] = fit[4]
        F[i] = fit[5]

    statA = ellipse_fit(x, y)[0]
    statB = ellipse_fit(x, y)[1]
    statC = ellipse_fit(x, y)[2]
    statD = ellipse_fit(x, y)[3]
    statE = ellipse_fit(x, y)[4]
    statF = ellipse_fit(x, y)[5]

    meanA = np.mean(A)
    meanB = np.mean(B)
    meanC = np.mean(C)
    meanD = np.mean(D)
    meanE = np.mean(E)
    meanF = np.mean(F)

    biasA = (n - 1) * (meanA - statA)
    biasB = (n - 1) * (meanB - statB)
    biasC = (n - 1) * (meanC - statC)
    biasD = (n - 1) * (meanD - statD)
    biasE = (n - 1) * (meanE - statE)
    biasF = (n - 1) * (meanF - statF)

    stdA = np.sqrt((n - 1) * np.mean((A - meanA) * (A - meanA), axis=0))
    stdB = np.sqrt((n - 1) * np.mean((B - meanB) * (B - meanB), axis=0))
    stdC = np.sqrt((n - 1) * np.mean((C - meanC) * (C - meanC), axis=0))
    stdD = np.sqrt((n - 1) * np.mean((D - meanD) * (D - meanD), axis=0))
    stdE = np.sqrt((n - 1) * np.mean((E - meanE) * (E - meanE), axis=0))
    stdF = np.sqrt((n - 1) * np.mean((F - meanF) * (F - meanF), axis=0))

    estimateA = statA - biasA
    estimateB = statB - biasB
    estimateC = statC - biasC
    estimateD = statD - biasD
    estimateE = statE - biasE
    estimateF = statF - biasF

    z_score = np.sqrt(2.0) * erfinv(confidence)

    conf_intervalA = estimateA + z_score * np.array((-stdA, stdA))
    conf_intervalB = estimateB + z_score * np.array((-stdB, stdB))
    conf_intervalC = estimateC + z_score * np.array((-stdC, stdC))
    conf_intervalD = estimateD + z_score * np.array((-stdD, stdD))
    conf_intervalE = estimateE + z_score * np.array((-stdE, stdE))
    conf_intervalF = estimateF + z_score * np.array((-stdF, stdF))

    estimate = np.array([np.abs(estimateA), np.abs(estimateB), np.abs(estimateC), estimateD, estimateE, estimateF])

    return estimate


'''
###### Values of orbital parameters ######
print(convert_to_physical(ellipse_fit(x_dataG, y_dataG)[0], ellipse_fit(x_dataG, y_dataG)[1], ellipse_fit(x_dataG, y_dataG)[2],
                         ellipse_fit(x_dataG, y_dataG)[3], ellipse_fit(x_dataG, y_dataG)[4],ellipse_fit(x_dataG, y_dataG)[5]))

print(convert_to_physical(JackKnife2D(ellipse_fit, x_dataG, y_dataG, 0.8)[0], JackKnife2D(ellipse_fit, x_dataG, y_dataG, 0.8)[1],
                         JackKnife2D(ellipse_fit, x_dataG, y_dataG, 0.8)[2], JackKnife2D(ellipse_fit, x_dataG, y_dataG, 0.8)[3],
                         JackKnife2D(ellipse_fit, x_dataG, y_dataG, 0.8)[4], JackKnife2D(ellipse_fit, x_dataG, y_dataG, 0.8)[5]))
'''

def ellipse_eq(x, y, x0, y0, a, b, theta):
  term1 = ((x - x0) * np.cos(theta) + (y - y0) * np.sin(theta)) ** 2 / a ** 2
  term2 = ((-(x - x0) * np.sin(theta) + (y - y0) * np.cos(theta)) ** 2) / b ** 2
  return term1 + term2 - 1  # ellipse is defined by ellipse_eq == 0

# Define the plotting grid.

N = 5000
x_min, x_max = -700, 700  # use names that don't conflict with x0_model, y0_model
y_min, y_max = -100, 100
xs = np.linspace(x_min, x_max, N)
ys = np.linspace(y_min, y_max, N)
X, Y = np.meshgrid(xs, ys)


def Z(X, Y, xdata, ydata):
    return ellipse_eq(X, Y, convert_to_physical(JackKnife2D(ellipse_fit, xdata, ydata, 0.8)[0],
                                                JackKnife2D(ellipse_fit, xdata, ydata, 0.8)[1],
                                                JackKnife2D(ellipse_fit, xdata, ydata, 0.8)[2],
                                                JackKnife2D(ellipse_fit, xdata, ydata, 0.8)[3],
                                                JackKnife2D(ellipse_fit, xdata, ydata, 0.8)[4],
                                                JackKnife2D(ellipse_fit, xdata, ydata, 0.8)[5])[0],  # x0
                      convert_to_physical(JackKnife2D(ellipse_fit, xdata, ydata, 0.8)[0],
                                          JackKnife2D(ellipse_fit, xdata, ydata, 0.8)[1],
                                          JackKnife2D(ellipse_fit, xdata, ydata, 0.8)[2],
                                          JackKnife2D(ellipse_fit, xdata, ydata, 0.8)[3],
                                          JackKnife2D(ellipse_fit, xdata, ydata, 0.8)[4],
                                          JackKnife2D(ellipse_fit, xdata, ydata, 0.8)[5])[1],  # y0
                      convert_to_physical(JackKnife2D(ellipse_fit, xdata, ydata, 0.8)[0],
                                          JackKnife2D(ellipse_fit, xdata, ydata, 0.8)[1],
                                          JackKnife2D(ellipse_fit, xdata, ydata, 0.8)[2],
                                          JackKnife2D(ellipse_fit, xdata, ydata, 0.8)[3],
                                          JackKnife2D(ellipse_fit, xdata, ydata, 0.8)[4],
                                          JackKnife2D(ellipse_fit, xdata, ydata, 0.8)[5])[2],  # a
                      convert_to_physical(JackKnife2D(ellipse_fit, xdata, ydata, 0.8)[0],
                                          JackKnife2D(ellipse_fit, xdata, ydata, 0.8)[1],
                                          JackKnife2D(ellipse_fit, xdata, ydata, 0.8)[2],
                                          JackKnife2D(ellipse_fit, xdata, ydata, 0.8)[3],
                                          JackKnife2D(ellipse_fit, xdata, ydata, 0.8)[4],
                                          JackKnife2D(ellipse_fit, xdata, ydata, 0.8)[5])[3],  # b
                      convert_to_physical(JackKnife2D(ellipse_fit, xdata, ydata, 0.8)[0],
                                          JackKnife2D(ellipse_fit, xdata, ydata, 0.8)[1],
                                          JackKnife2D(ellipse_fit, xdata, ydata, 0.8)[2],
                                          JackKnife2D(ellipse_fit, xdata, ydata, 0.8)[3],
                                          JackKnife2D(ellipse_fit, xdata, ydata, 0.8)[4],
                                          JackKnife2D(ellipse_fit, xdata, ydata, 0.8)[5])[4])  # theta


############ 2D plot for orbit of all moons #############

plt.figure(figsize=(8, 8))

plt.errorbar(x_dataI, y_dataI, xerr=np.ones(len(x_dataI)), yerr=np.ones(len(x_dataI)), ls='none', color='black',
             markersize=5, capsize=3, marker='o')  # , alpha=0.5)
io = plt.contour(X, Y, Z(X, Y, x_dataI, y_dataI), levels=[0], colors='orange', linewidths=2)

plt.errorbar(x_dataC, y_dataC, xerr=np.ones(len(x_dataC)), yerr=np.ones(len(x_dataC)), ls='none', color='black',
             markersize=5, capsize=3, marker='o')  # , alpha=0.5)
calli = plt.contour(X, Y, Z(X, Y, x_dataC, y_dataC), levels=[0], colors='blue', linewidths=2)

plt.errorbar(x_dataG, y_dataG, xerr=np.ones(len(x_dataG)), yerr=np.ones(len(x_dataG)), ls='none', color='black',
             markersize=5, capsize=3, marker='o')  # , alpha=0.5)
gany = plt.contour(X, Y, Z(X, Y, x_dataG, y_dataG), levels=[0], colors='green', linewidths=2)

#### Europa does not work correctly ####
plt.errorbar(x_dataE, y_dataE, xerr = np.ones(len(x_dataE)), yerr = np.ones(len(x_dataE)), ls = 'none', color='black', markersize=5, capsize=3, marker='o', label='Europa')
euro = plt.contour(X, Y, Z(X, Y, x_dataE, y_dataE), levels=[0], colors='purple')


plt.scatter(0, 0, color='red', label='Jupiter')

proxy = [plt.Rectangle((0, 0), 1, 1, fc='orange'), plt.Rectangle((0, 0), 1, 1, fc='purple'),
         plt.Rectangle((0, 0), 1, 1, fc='green'),
         plt.Rectangle((0, 0), 1, 1, fc='blue')]  # , plt.Rectangle((0,0),1,1,fc = 'red')]
plt.legend(proxy, ['Io', 'Europa', 'Ganymede', 'Callisto'], fontsize=18)

plt.ylim(-80, 80)
plt.xlim(-600, 600)
plt.xlabel("X (Arcsec)", fontsize=18)
plt.ylabel("Y (Arcsec)", fontsize=18)

plt.tick_params(axis='both', which='major', labelsize=16, length=8, width=1.5, top=True, right=True)
plt.tick_params(axis='both', which='minor', labelsize=16, length=4, width=1, top=True, right=True)
plt.minorticks_on()

plt.tight_layout()
plt.show()

'''
##### Io values #####
print(convert_to_physical(ellipse_fit(x_dataI, y_dataI)[0], ellipse_fit(x_dataI, y_dataI)[1], ellipse_fit(x_dataI, y_dataI)[2],
                         ellipse_fit(x_dataI, y_dataI)[3], ellipse_fit(x_dataI, y_dataI)[4],ellipse_fit(x_dataI, y_dataI)[5]))

print(convert_to_physical(JackKnife2D(ellipse_fit, x_dataI, y_dataI, 0.8)[0], JackKnife2D(ellipse_fit, x_dataI, y_dataI, 0.8)[1],
                            JackKnife2D(ellipse_fit, x_dataI, y_dataI, 0.8)[2], JackKnife2D(ellipse_fit, x_dataI, y_dataI, 0.8)[3],
                            JackKnife2D(ellipse_fit, x_dataI, y_dataI, 0.8)[4], JackKnife2D(ellipse_fit, x_dataI, y_dataI, 0.8)[5]))
'''

'''
##### Europa values #####
print(convert_to_physical(ellipse_fit(x_dataE, y_dataE)[0], ellipse_fit(x_dataE, y_dataE)[1], ellipse_fit(x_dataE, y_dataE)[2],
                            ellipse_fit(x_dataE, y_dataE)[3], ellipse_fit(x_dataE, y_dataE)[4],ellipse_fit(x_dataE, y_dataE)[5]))

print(convert_to_physical(JackKnife2D(ellipse_fit, x_dataE, y_dataE, 0.8)[0], JackKnife2D(ellipse_fit, x_dataE, y_dataE, 0.8)[1],
                            JackKnife2D(ellipse_fit, x_dataE, y_dataE, 0.8)[2], JackKnife2D(ellipse_fit, x_dataE, y_dataE, 0.8)[3],
                            JackKnife2D(ellipse_fit, x_dataE, y_dataE, 0.8)[4], JackKnife2D(ellipse_fit, x_dataE, y_dataE, 0.8)[5]))
'''

'''
##### Ganymede values #####
print(convert_to_physical(ellipse_fit(x_dataG, y_dataG)[0], ellipse_fit(x_dataG, y_dataG)[1], ellipse_fit(x_dataG, y_dataG)[2],
                         ellipse_fit(x_dataG, y_dataG)[3], ellipse_fit(x_dataG, y_dataG)[4],ellipse_fit(x_dataG, y_dataG)[5]))

print(convert_to_physical(JackKnife2D(ellipse_fit, x_dataG, y_dataG, 0.8)[0], JackKnife2D(ellipse_fit, x_dataG, y_dataG, 0.8)[1],
                         JackKnife2D(ellipse_fit, x_dataG, y_dataG, 0.8)[2], JackKnife2D(ellipse_fit, x_dataG, y_dataG, 0.8)[3],
                         JackKnife2D(ellipse_fit, x_dataG, y_dataG, 0.8)[4], JackKnife2D(ellipse_fit, x_dataG, y_dataG, 0.8)[5]))
'''

'''
##### Callisto values #####
print(convert_to_physical(ellipse_fit(x_dataC, y_dataC)[0], ellipse_fit(x_dataC, y_dataC)[1], ellipse_fit(x_dataC, y_dataC)[2],
                         ellipse_fit(x_dataC, y_dataC)[3], ellipse_fit(x_dataC, y_dataC)[4],ellipse_fit(x_dataC, y_dataC)[5]))

print(convert_to_physical(JackKnife2D(ellipse_fit, x_dataC, y_dataC, 0.8)[0], JackKnife2D(ellipse_fit, x_dataC, y_dataC, 0.8)[1],
                         JackKnife2D(ellipse_fit, x_dataC, y_dataC, 0.8)[2], JackKnife2D(ellipse_fit, x_dataC, y_dataC, 0.8)[3],
                         JackKnife2D(ellipse_fit, x_dataC, y_dataC, 0.8)[4], JackKnife2D(ellipse_fit, x_dataC, y_dataC, 0.8)[5]))
'''