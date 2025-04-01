from data_processing import data as d
from data_processing.ellipse_fitting import ellipse_fit, convert_to_physical
from data_processing.rotate_data import Rotate_Data
from data_processing import ellipse_fitting
import emcee
import numpy as np
import matplotlib.pyplot as plt
import corner

""" This allows you to run an MC sampler around the parameter space defined in the prior
    from the list: 
    prior_bounds = [LowerX0, UpperX0, LowerY0, UpperY0, Lower_a, Upper_a, Lower_b, Upper_b, lower_theta, Upper_theta]
    All Values in list must be floats and theta must be in radians.
     
     NOTE: We are assuming an error on the positions as +-1 pixel"""

class MCSampler:
    def __init__(self, moon, prior_bounds):
        self.name = moon
        self.prior_bounds = prior_bounds

        J = d.compile[0]
        angles = d.compile[5]
        scales = d.compile[6]
        flips = d.compile[7]
        time = d.compile[8]

        if self.name.casefold() == 'Io'.casefold():
            self.index = 1
        elif self.name.casefold() == 'Europa'.casefold():
            self.index = 2
        elif self.name.casefold() == 'Ganymede'.casefold():
            self.index = 3
        elif self.name.casefold() == 'Callisto'.casefold():
            self.index = 4
        else:
            raise ValueError("You must input the name of one of Jupiter's Galilean moons")

        ## format data ##
        moonX = d.compile[self.index][0]
        moonY = d.compile[self.index][1]

        data_stack = np.column_stack([J[0], J[1], moonX, moonY])
        valid_mask = ~np.isnan(data_stack).any(axis=1)

        Jx, Jy = J[0][valid_mask], J[1][valid_mask]
        moonX, moonY = moonX[valid_mask],moonY[valid_mask]

        self.Moon_X = Rotate_Data(Jx, Jy, moonX, moonY, angles)[0]
        self.Moon_Y = Rotate_Data(Jx, Jy, moonX, moonY, angles)[1]
        self.xerr = np.ones(len(self.Moon_X))
        self.yerr = np.ones(len(self.Moon_Y))

    def bestfit_params(self):
        fit = ellipse_fit(self.Moon_X, self.Moon_Y)
        A, B, C, D, E, F = fit[0], fit[1], fit[2], fit[3], fit[4], fit[5]
        x0, y0, a, b, theta = convert_to_physical(A, B, C, D, E, F)
        self.parameters = x0, y0, a, b, theta
        return self.parameters

    def log_likelihood_physical(self, params, sigma_x=1.0, sigma_y=1.0):
        x, y = self.Moon_X, self.Moon_Y
        # Unpack params
        x0, y0, a, b, theta = params

        # Evaluate the ellipse function at each (x, y)
        f_val = ((x - x0) * np.cos(theta) + (y - y0) * np.sin(theta)) ** 2 / a ** 2 + (
                    -(x - x0) * np.sin(theta) + (y - y0) * np.cos(theta)) ** 2 / b ** 2 - 1

        # Compute the gradients
        df_dx = (2 * np.cos(theta) * ((x - x0) * np.cos(theta) + (y - y0) * np.sin(theta))) / a ** 2 - (
                    2 * np.sin(theta) * ((x0 - x) * np.sin(theta) + (y - y0) * np.cos(theta))) / b ** 2
        df_dy = (2 * np.sin(theta) * ((x - x0) * np.cos(theta) + (y - y0) * np.sin(theta))) / a ** 2 + (
                    2 * np.cos(theta) * (-(x - x0) * np.sin(theta) - ((y0 - y) * np.cos(theta)))) / b ** 2
        grad_norm = np.sqrt(df_dx ** 2 + df_dy ** 2)

        # Calculate the orthodonal distance from the data point to the ellipse
        d = np.abs(f_val) / grad_norm

        # Compute the unit normal components
        n_x = df_dx / grad_norm
        n_y = df_dy / grad_norm

        # Calculate the effective uncertainty along the normal direction
        sigma_normal = np.sqrt((n_x * sigma_x) ** 2 + (n_y * sigma_y) ** 2)

        # Protect against division by zero
        sigma_normal[sigma_normal == 0] = 1e-10

        # Log likelihood for each data point (assuming independent points)
        logL = -0.5 * ((d / sigma_normal) ** 2 + np.log(2 * np.pi * sigma_normal ** 2))
        return np.sum(logL)

    def log_prior_physical(self, params):
        LowerX0, UpperX0, LowerY0, UpperY0, Lower_a, Upper_a, Lower_b, Upper_b, lower_theta, Upper_theta = self.prior_bounds
        x0, y0, a, b, theta = params
        if LowerX0 < x0 < UpperX0 and LowerY0 < y0 < UpperY0 and Lower_a < a < Upper_a and Lower_b < b < Upper_b and lower_theta < theta < Upper_theta:
            return 0.0
        return -np.inf

    def log_probability_physical(self, params):
        lp = self.log_prior_physical(params)
        if not np.isfinite(lp):
            return -np.inf
        return lp + self.log_likelihood_physical(params)

    def run(self, file='C:/Users/ishan/PycharmProjects/JovianSatellites/MCMC Backups/Test'):
        initials = np.array(self.bestfit_params())
        scales = np.array([1e-1, 1e-1, 1e-1, 1e-1, 1e-3])
        nwalkers = 32
        ndim = initials.shape[0]
        pos = initials + np.random.randn(nwalkers, ndim) * scales

        backend = emcee.backends.HDFBackend(file)
        backend.reset(nwalkers, ndim)

        params = np.array(self.bestfit_params())
        sampler = emcee.EnsembleSampler(nwalkers, ndim, self.log_probability_physical, backend=backend)
        sampler.run_mcmc(pos, 50000, progress=True, store=True)

        samples = sampler.get_chain()

        fig, axes = plt.subplots(5, figsize=(10, 7), sharex=True)
        labels = ["x0", "y0", "a", "b", "theta"]
        for i in range(ndim):
            ax = axes[i]
            ax.plot(samples[:, :, i], "k", alpha=0.3)
            ax.set_xlim(0, len(samples))
            ax.set_ylabel(labels[i])
            ax.yaxis.set_label_coords(-0.1, 0.5)

        axes[-1].set_xlabel("step number");

        flat_samples = sampler.get_chain(discard=1700, thin=30, flat=True)
        print(flat_samples.shape)  # This gives you the total number of samples

        fig = corner.corner(flat_samples, labels=labels)
        plt.show()
        return samples

if __name__ == "__main__":
    Gany_prior_bounds = [-50*4, 8*2, -10*2, 10*2, 310*0.5, 400*2, 10, 30, -0.2, 0]
    Gany = MCSampler('ganymede', prior_bounds=Gany_prior_bounds)

    Euro_prior_bounds = [-10, 50, -100, 100, 100, 800, 0, 400, -1, 0] # Excellent Bounds but x0 was cut off, 'Euro1Test'
    # Euro_prior_bounds = [0, 90, -100, 100, 100, 800, 0, 400, -1, 0] # 'Euro2' breaks
    Euro = MCSampler('europa', prior_bounds=Euro_prior_bounds)

    Io_prior_bounds = [-100, 100, -100, 100, -100, 500, 0, 400, 5, 0] # 'Io1'
    Io = MCSampler('Io', prior_bounds=Io_prior_bounds)

    # print(Io.bestfit_params())
    Io.run('C:/Users/ishan/PycharmProjects/JovianSatellites/MCMC Backups/Io1')
    # Euro.run('C:/Users/ishan/PycharmProjects/JovianSatellites/MCMC Backups/Euro2')

    pass
