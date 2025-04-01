import numpy as np
from data_processing.data import scale, flip

""" These functions are for rotating the data from each set based on the 
 CCD angle found using the getrot.py function in the Physics labs"""

## rotate_coords ## rotates a set of coordinates for a single datapoint using 2D rotation matrix
# it scales the points based on the specific plate scale of the telescopes
# it also returns the x and y values of each moon relative to Jupiter

def rotate_coords(Jx, Jy, x, y, theta, scale = scale, flip = flip):

    # input = value, output = vector [[x],[y]]
    # Rotates Coords and calculates relative distance from jupiter
    # according to the getrot function given by theta, this alligns the points to North. also scales according to the pixel scale
    # Flip is a boolean and if True, flips coords by 180
    # this just returns 1 value .

    theta = theta*np.pi/180
    if flip == True:
        theta += np.pi
    J = np.vstack([Jx, Jy])
    r = np.vstack([x, y])
    A = np.array([[np.cos(theta), -np.sin(theta)],[np.sin(theta), np.cos(theta)]])
    r_prime = A@r
    J_prime = A@J

    r_prime = (r_prime - J_prime)*scale

    return r_prime

""" I HAVE RENAMED THIS FUNCTION FROM 'data' to 'Rotate_Data' """
## data ## just applies this rotation to the entire dataset for a given body

def Rotate_Data(Jx, Jy, x, y, theta, s = scale, f = flip):

    # This does the same thing as rotate coords, but for the whole data set, and returns an array of
    # x and y values for the inputted moons

    x_data, y_data = [], []
    for i in range(len(Jx)):
        x_prime, y_prime = rotate_coords(Jx[i], Jy[i], x[i], y[i], theta[i], s[i], f[i])
        x_data = np.append(x_data, x_prime)
        y_data = np.append(y_data, y_prime)
    return x_data, y_data
