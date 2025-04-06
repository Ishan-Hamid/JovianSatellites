''' DOCUMENTATION:
You can run everything from main.py by importing the packages you need. i think you need all the files tho

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
