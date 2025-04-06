import numpy as np
import matplotlib.pyplot as plt
from scipy import optimize as opt
import pandas as pd
import astropy.stats as astats
from matplotlib import gridspec
from scipy.special import erfinv
from data_processing.ellipse_fitting import ellipse_fit, convert_to_physical, model_ellipse, physical_model

from jackknife_data import compiled_data

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

def time_phase(time, period):
    return (time / (24 * 60)) / period % 1

def residuals(x, modelval):
    return x - modelval

def normal_resid(x, modelval):
    return residuals(x, modelval) / np.std(residuals(x, modelval))

def plot_1D(name, xy):
    smooth_phase = np.linspace(0, 1, 300)

    index = 0

    if name.casefold() == 'Io'.casefold():
        index = 1
        col = 'orange'
    elif name.casefold() == 'Europa'.casefold():
        index = 2
        col = 'purple'
    elif name.casefold() == 'Ganymede'.casefold():
        index = 3
        col = 'green'
    elif name.casefold() == 'Callisto'.casefold():
        index = 4
        col = 'Blue'
    else:
        raise ValueError("You must input the name of one of Jupiter's Galilean moons")

    xdata = compiled_data[index][0]
    ydata = compiled_data[index][1]
    xinitial = compiled_data[index][3]
    yinitial = compiled_data[index][4]

    if xy == 'x':
        data = xdata
        IVs = xinitial
    elif xy == 'y':
        data = ydata
        IVs = yinitial

    time = compiled_data[index][2]

    plt.figure(figsize=(8, 6))
    gs = gridspec.GridSpec(2, 1, height_ratios=[3, 1], hspace=0)

    ax1 = plt.subplot(
        gs[0])  # 0.8 is the confidence level (not sure what is should be set to yet)
    ax1.errorbar(time_phase(time, JackKnife(model, data, 0.8, time, IVs)[0][1]),
                 # *JackKnife(model, x_dataI, 0.8, Time_minsI, initial_valuesIx)[0][1],
                 data, yerr=0.5, marker='o', markersize=5, capsize=3, label='Data', ls='none',
                 color='black')  # ^^^including above makes the y-axis be in terms of days instead of phase

    ax1.plot(smooth_phase,  # *JackKnife(model, x_dataI, 0.8, Time_minsI, initial_valuesIx)[0][1],
             model(smooth_phase * JackKnife(model, data, 0.8, time, IVs)[0][1],
                   JackKnife(model, data, 0.8, time, IVs)[0][0],
                   JackKnife(model, data, 0.8, time, IVs)[0][1],
                   JackKnife(model, data, 0.8, time, IVs)[0][2]),
             label='Fit', linewidth=2, color=col)
    ax1.set_ylabel('X (Arcsec)', fontsize=18)
    ax1.legend(fontsize=12)
    ax1.set_title(name, fontsize=18)
    ax1.tick_params(axis='both', which='major', labelsize=14, width=2, length=7, top=True, right=True)
    ax1.tick_params(axis='both', which='minor', labelsize=12, width=1, length=3, top=True, right=True)
    ax1.minorticks_on()

    ax2 = plt.subplot(gs[1], sharex=ax1)
    ax2.axhline(0, color=col, linestyle='--', linewidth=2)  # Reference line at zeroq
    ax2.errorbar(time_phase(time, JackKnife(model, data, 0.8, time, IVs)[0][1]),
                 # *JackKnife(model, x_dataI, 0.8, Time_minsI, initial_valuesIx)[0][1],
                 normal_resid(data, model(time / (24 * 60),
                                             # time_phase(Time_minsI,JackKnife(model, x_dataI, 0.8, Time_minsI, initial_valuesIx)[0][1]),
                                             JackKnife(model, data, 0.8, time, IVs)[0][0],
                                             JackKnife(model, data, 0.8, time, IVs)[0][1],
                                             JackKnife(model, data, 0.8, time, IVs)[0][2])),
                 marker='o', markersize=5, capsize=3, ls='none', color='black')

    ax2.set_xlabel('Phase', fontsize=18)
    ax2.set_ylabel('Norm Residuals', fontsize=18)
    ax2.tick_params(axis='both', which='major', labelsize=16, width=2, length=7, right=True)
    ax2.tick_params(axis='both', which='minor', labelsize=12, width=1, length=3, right=True)
    ax2.minorticks_on()
    plt.tight_layout()
    plt.show()

####################### 2D ########################

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

# def ellipse_eq(x, y, x0, y0, a, b, theta):
#   term1 = ((x - x0) * np.cos(theta) + (y - y0) * np.sin(theta)) ** 2 / a ** 2
#   term2 = ((-(x - x0) * np.sin(theta) + (y - y0) * np.cos(theta)) ** 2) / b ** 2
#   return term1 + term2 - 1  # ellipse is defined by ellipse_eq == 0

def Z(X, Y, xdata, ydata):
    return physical_model(X, Y, convert_to_physical(JackKnife2D(ellipse_fit, xdata, ydata, 0.8)[0],
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

def ellipse_plot():

    N = 5000
    x_min, x_max = -700, 700  # use names that don't conflict with x0_model, y0_model
    y_min, y_max = -100, 100
    xs = np.linspace(x_min, x_max, N)
    ys = np.linspace(y_min, y_max, N)
    X, Y = np.meshgrid(xs, ys)

    x_dataI = compiled_data[1][0]
    y_dataI = compiled_data[1][1]

    x_dataE = compiled_data[2][0]
    y_dataE = compiled_data[2][1]

    x_dataG = compiled_data[3][0]
    y_dataG = compiled_data[3][1]

    x_dataC = compiled_data[4][0]
    y_dataC = compiled_data[4][1]

    plt.figure(figsize=(8, 8))

    plt.errorbar(x_dataI, y_dataI, xerr=np.ones(len(x_dataI)), yerr=np.ones(len(x_dataI)), ls='none', color='orange',
                 markersize=5, capsize=3, marker='o')  # , alpha=0.5)
    io = plt.contour(X, Y, Z(X, Y, x_dataI, y_dataI), levels=[0], colors='orange', linewidths=2)

    plt.errorbar(x_dataC, y_dataC, xerr=np.ones(len(x_dataC)), yerr=np.ones(len(x_dataC)), ls='none', color='blue',
                 markersize=5, capsize=3, marker='o')  # , alpha=0.5)
    calli = plt.contour(X, Y, Z(X, Y, x_dataC, y_dataC), levels=[0], colors='blue', linewidths=2)

    plt.errorbar(x_dataG, y_dataG, xerr=np.ones(len(x_dataG)), yerr=np.ones(len(x_dataG)), ls='none', color='green',
                 markersize=5, capsize=3, marker='o')  # , alpha=0.5)
    gany = plt.contour(X, Y, Z(X, Y, x_dataG, y_dataG), levels=[0], colors='green', linewidths=2)

    #### Europa does not work correctly ####
    plt.errorbar(x_dataE, y_dataE, xerr=np.ones(len(x_dataE)), yerr=np.ones(len(x_dataE)), ls='none', color='purple',
                 markersize=5, capsize=3, marker='o', label='Europa')
    euro = plt.contour(X, Y, Z(X, Y, x_dataE, y_dataE), levels=[0], colors='purple')

    plt.scatter(0, 0, color='red', label='Jupiter')

    proxy = [plt.Rectangle((0, 0), 1, 1, fc='orange'), plt.Rectangle((0, 0), 1, 1, fc='purple'),
             plt.Rectangle((0, 0), 1, 1, fc='green'),
             plt.Rectangle((0, 0), 1, 1, fc='blue')]  # , plt.Rectangle((0,0),1,1,fc = 'red')]
    plt.legend(proxy, ['Io', 'Europa', 'Ganymede', 'Callisto'], fontsize=18)

    plt.ylim(-90, 90)
    plt.xlim(-650, 650)
    plt.xlabel("X (Arcsec)", fontsize=18)
    plt.ylabel("Y (Arcsec)", fontsize=18)

    plt.tick_params(axis='both', which='major', labelsize=16, length=8, width=1.5, top=True, right=True)
    plt.tick_params(axis='both', which='minor', labelsize=16, length=4, width=1, top=True, right=True)
    plt.minorticks_on()

    plt.tight_layout()

def jk_params(name):
    if name.casefold() == 'Io'.casefold():
        index = 1
        col = 'orange'
    elif name.casefold() == 'Europa'.casefold():
        index = 2
        col = 'purple'
    elif name.casefold() == 'Ganymede'.casefold():
        index = 3
        col = 'green'
    elif name.casefold() == 'Callisto'.casefold():
        index = 4
        col = 'Blue'
    else:
        raise ValueError("You must input the name of one of Jupiter's Galilean moons")

    xdata = compiled_data[index][0]
    ydata = compiled_data[index][1]
    xinitial = compiled_data[index][3]
    yinitial = compiled_data[index][4]

    print('Square fit params:  ', convert_to_physical(ellipse_fit(xdata, ydata)[0], ellipse_fit(xdata, ydata)[1],
                              ellipse_fit(xdata, ydata)[2],
                              ellipse_fit(xdata, ydata)[3], ellipse_fit(xdata, ydata)[4],
                              ellipse_fit(xdata, ydata)[5]))

    print('Jackknifed params: ', convert_to_physical(JackKnife2D(ellipse_fit, xdata, ydata, 0.8)[0],
                              JackKnife2D(ellipse_fit, xdata, ydata, 0.8)[1],
                              JackKnife2D(ellipse_fit, xdata, ydata, 0.8)[2],
                              JackKnife2D(ellipse_fit, xdata, ydata, 0.8)[3],
                              JackKnife2D(ellipse_fit, xdata, ydata, 0.8)[4],
                              JackKnife2D(ellipse_fit, xdata, ydata, 0.8)[5]))
    return

if __name__ == '__main__':
    # ellipse_plot()
    jk_params('Io')
    pass