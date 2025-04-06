import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy import optimize as opt
from scipy.optimize import minimize
import emcee
import corner

from MCMCFits.mcmc_finding_physical_params import MCSampler
from data_processing import data as d
from data_processing.find_jupiter_centroid import Centroid
from data_processing.rotate_data import Rotate_Data
from data_processing.find_jupiter_centroid import Centroid

''' DOCUMENTATION:
You can run everything from main.py by importing the packages you need.

#data_processing#

All of the data and pre-data processing (except from the data used by jackknife since that used a different file) is done in the data_processing package, this includes
rotating the coordinates and finding jupiter centroid.
This means that you need to change the filename found in data_processing.data and Jackknife.jackknife_data

so to access the data specifically or any of the functions, you'd have to import it: 

    eg) from data_processing.rotate_data import Rotate_Data
    
    eg) from data_processing import data as d
'''
###################################################
'''
#Jackknifing#
 
this is basically your file but i broke it up so that it doesnt run everything at once: 

    eg) jack_knifer.plot_1D() takes the name of a moon as a string and the dimension you want to plot, either x or y eg) jack_knifer.plot_1D('Callisto', 'x') and it will
    plot callisto x vs time on the sinusoidal fit. 

    eg) jack_knifer.ellipse_plot() will plot the final ellipse plot using the jackknifed ellipse fits.  (u need to add plt.show)

    eg) jk_params() gives u sf params and jk params eg) jack_knifer.jk_params('Io') 

'''
####################################################
''' #MCMCFits#

Basically you create an object using the MCSampler class and you give it the name of a moon as a string and the parameter boundaries of the MCMC

the prior bounds you must give it should be a list 
    
    eg) Io_prior_bounds = [-1, 1, -1, 1, 100, 300, 0, 60, -np.pi, np.pi]
    this corresponds to telling the MCSampler you want the samplers to search in the range
    -1 < x0 < 1, -1 < y0 < 1, 100 < a < 300, 0 < b < 60, -np.pi < theta < np.pi

    eg) Gany = MCSampler('ganymede', prior_bounds = Gany_prior_bounds)
    eg) Io = MCSampler('Io', prior_bounds = Io_prior_bounds)

and then with this object you can run alot of functions in this package:

    eg) Gany.plot() will simply plot the X and Y rotated and wrt Jupiter
    
    eg) Gany.bestfitparams() will return the params: x0, y0, a, b, theta (radians)
    
    eg) Io.run(filename) takes a file as a string that the MCMC will write to in order to store the data to be used 
    
    eg) Io.plot_MCMC(filename) This takes a file that you have created using the .run commmand and then uses it to create an ellipse plot
    witht the parameters found with the MCMC method, i will provide test files.
    
    eg) Io.corner_plot(filename) This takes a file that you have created using the .run commmand and then uses it to create a Corner Plot of the chain. 

'''


Gany_prior_bounds = [-1, 1, -1, 1, 200, 450, 5, 45, -np.pi, np.pi]  # 'GanyTest1' Good bounds and produce good ellipse but x0 is unconstrained
Gany = MCSampler('ganymede', prior_bounds=Gany_prior_bounds)

Euro_prior_bounds = [-1, 1, -1, 1, 150, 450, 0, 100, -np.pi, np.pi]  # 'EuroTest3'
Euro = MCSampler('europa', prior_bounds=Euro_prior_bounds)

Io_prior_bounds = [-1, 1, -1, 1, 100, 300, 0, 60, -np.pi, np.pi]  # 'IoTest1'
Io = MCSampler('Io', prior_bounds=Io_prior_bounds)

Calli_prior_bounds = [-2, 2, -2, 2, 300, 800, 20, 80, -np.pi, np.pi]  # 'CalliTest2'
Calli = MCSampler('Callisto', prior_bounds=Calli_prior_bounds)

Calli.plot_MCMC('C:/Users/ishan/PycharmProjects/JovianSatellites/MCMC Backups/CalliTest1')
Gany.plot_MCMC('C:/Users/ishan/PycharmProjects/JovianSatellites/MCMC Backups/GanyTest1')
Euro.plot_MCMC('C:/Users/ishan/PycharmProjects/JovianSatellites/MCMC Backups/EuroTest3')
Io.plot_MCMC('C:/Users/ishan/PycharmProjects/JovianSatellites/MCMC Backups/IoTest2')

proxy = [plt.Rectangle((0, 0), 1, 1, fc='orange'), plt.Rectangle((0, 0), 1, 1, fc='purple'),
         plt.Rectangle((0, 0), 1, 1, fc='green'), plt.Rectangle((0, 0), 1, 1, fc='blue'), plt.axhline(0, 0, 0, color = 'red')]  # , plt.Rectangle((0,0),1,1,fc = 'red')]
plt.legend(proxy, ['Io', 'Europa', 'Ganymede', 'Callisto', 'Mean Param Fit'], fontsize=18)

plt.ylim(-90, 90)
plt.xlim(-650, 650)
plt.xlabel("X (Arcsec)", fontsize=18)
plt.ylabel("Y (Arcsec)", fontsize=18)
plt.scatter(0, 0, color='red', label='Jupiter')
plt.tick_params(axis='both', which='major', labelsize=16, length=8, width=1.5, top=True, right=True)
plt.tick_params(axis='both', which='minor', labelsize=16, length=4, width=1, top=True, right=True)
plt.minorticks_on()

plt.show()
