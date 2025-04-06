# from data_processing import data as d
# from data_processing.ellipse_fitting import ellipse_fit, convert_to_physical
# from data_processing.rotate_data import Rotate_Data
# from data_processing import ellipse_fitting
# import emcee
# import numpy as np
# import matplotlib.pyplot as plt
# import corner
#
# """ This allows you to run an MC sampler around the parameter space defined in the prior
#     from the list:
#     prior_bounds = [LowerX0, UpperX0, LowerY0, UpperY0, Lower_a, Upper_a, Lower_b, Upper_b, lower_theta, Upper_theta]
#     All Values in list must be floats and theta must be in radians.
#
#      NOTE: We are assuming an error on the positions as +-1 pixel"""
#
# class MCSampler:
#     def __init__(self, moon, prior_bounds):
#         self.name = moon
#         self.prior_bounds = prior_bounds
#
#         J = d.compile[0]
#         angles = d.compile[5]
#         scales = d.compile[6]
#         flips = d.compile[7]
#         time = d.compile[8]
#
#         if self.name.casefold() == 'Io'.casefold():
#             self.index = 1
#         elif self.name.casefold() == 'Europa'.casefold():
#             self.index = 2
#         elif self.name.casefold() == 'Ganymede'.casefold():
#             self.index = 3
#         elif self.name.casefold() == 'Callisto'.casefold():
#             self.index = 4
#         else:
#             raise ValueError("You must input the name of one of Jupiter's Galilean moons")
#
#         ## format data ##
#         moonX = d.compile[self.index][0]
#         moonY = d.compile[self.index][1]
#
#         data_stack = np.column_stack([J[0], J[1], moonX, moonY])
#         valid_mask = ~np.isnan(data_stack).any(axis=1)
#
#         Jx, Jy = J[0][valid_mask], J[1][valid_mask]
#         moonX, moonY = moonX[valid_mask],moonY[valid_mask]
#
#         self.Moon_X = Rotate_Data(Jx, Jy, moonX, moonY, angles)[0]
#         self.Moon_Y = Rotate_Data(Jx, Jy, moonX, moonY, angles)[1]
#         self.xerr = np.ones(len(self.Moon_X))
#         self.yerr = np.ones(len(self.Moon_Y))
#
#     def bestfit_params(self):
#         fit = ellipse_fit(self.Moon_X, self.Moon_Y)
#         A, B, C, D, E, F = fit[0], fit[1], fit[2], fit[3], fit[4], fit[5]
#         x0, y0, a, b, theta = convert_to_physical(A, B, C, D, E, F)
#         self.parameters = x0, y0, a, b, theta
#         return self.parameters
#
#     def log_likelihood_physical(self, params, sigma_x=1.0, sigma_y=1.0):
#         x, y = self.Moon_X, self.Moon_Y
#         # Unpack params
#         x0, y0, a, b, theta = params
#
#         # Evaluate the ellipse function at each (x, y)
#         f_val = ((x - x0) * np.cos(theta) + (y - y0) * np.sin(theta)) ** 2 / a ** 2 + (
#                     -(x - x0) * np.sin(theta) + (y - y0) * np.cos(theta)) ** 2 / b ** 2 - 1
#
#         # Compute the gradients
#         df_dx = (2 * np.cos(theta) * ((x - x0) * np.cos(theta) + (y - y0) * np.sin(theta))) / a ** 2 - (
#                     2 * np.sin(theta) * ((x0 - x) * np.sin(theta) + (y - y0) * np.cos(theta))) / b ** 2
#         df_dy = (2 * np.sin(theta) * ((x - x0) * np.cos(theta) + (y - y0) * np.sin(theta))) / a ** 2 + (
#                     2 * np.cos(theta) * (-(x - x0) * np.sin(theta) - ((y0 - y) * np.cos(theta)))) / b ** 2
#         grad_norm = np.sqrt(df_dx ** 2 + df_dy ** 2)
#
#         # Calculate the orthodonal distance from the data point to the ellipse
#         d = np.abs(f_val) / grad_norm
#
#         # Compute the unit normal components
#         n_x = df_dx / grad_norm
#         n_y = df_dy / grad_norm
#
#         # Calculate the effective uncertainty along the normal direction
#         sigma_normal = np.sqrt((n_x * sigma_x) ** 2 + (n_y * sigma_y) ** 2)
#
#         # Protect against division by zero
#         sigma_normal[sigma_normal == 0] = 1e-10
#
#         # Log likelihood for each data point (assuming independent points)
#         logL = -0.5 * ((d / sigma_normal) ** 2 + np.log(2 * np.pi * sigma_normal ** 2))
#         return np.sum(logL)
#
#     def log_prior_physical(self, params):
#         LowerX0, UpperX0, LowerY0, UpperY0, Lower_a, Upper_a, Lower_b, Upper_b, lower_theta, Upper_theta = self.prior_bounds
#         x0, y0, a, b, theta = params
#         if LowerX0 < x0 < UpperX0 and LowerY0 < y0 < UpperY0 and Lower_a < a < Upper_a and Lower_b < b < Upper_b and lower_theta < theta < Upper_theta:
#             return 0.0
#         return -np.inf
#
#     def log_probability_physical(self, params):
#         lp = self.log_prior_physical(params)
#         if not np.isfinite(lp):
#             return -np.inf
#         return lp + self.log_likelihood_physical(params)
#
#     def run(self, file='C:/Users/ishan/PycharmProjects/JovianSatellites/MCMC Backups/Test'):
#         initials = np.array(self.bestfit_params())
#         scales = np.array([1e-1, 1e-1, 1e-1, 1e-1, 1e-3])
#         nwalkers = 32
#         ndim = initials.shape[0]
#         pos = initials + np.random.randn(nwalkers, ndim) * scales
#
#         backend = emcee.backends.HDFBackend(file)
#         backend.reset(nwalkers, ndim)
#
#         params = np.array(self.bestfit_params())
#         sampler = emcee.EnsembleSampler(nwalkers, ndim, self.log_probability_physical, backend=backend)
#         sampler.run_mcmc(pos, 15000, progress=True, store=True)
#
#         samples = sampler.get_chain()
#
#         fig, axes = plt.subplots(5, figsize=(10, 7), sharex=True)
#         labels = ["x0", "y0", "a", "b", "theta"]
#         for i in range(ndim):
#             ax = axes[i]
#             ax.plot(samples[:, :, i], "k", alpha=0.3)
#             ax.set_xlim(0, len(samples))
#             ax.set_ylabel(labels[i])
#             ax.yaxis.set_label_coords(-0.1, 0.5)
#
#         axes[-1].set_xlabel("step number");
#
#         flat_samples = sampler.get_chain(discard=1700, thin=30, flat=True)
#         print(flat_samples.shape)  # This gives you the total number of samples
#
#         fig = corner.corner(flat_samples, labels=labels)
#         plt.show()
#         return samples
#
# if __name__ == "__main__":
#     Gany_prior_bounds = [-50*4, 8*2, -10*2, 10*2, 310*0.5, 400*2, 10, 30, -0.2, 0]
#     Gany = MCSampler('ganymede', prior_bounds=Gany_prior_bounds)
#
#     Euro_prior_bounds = [-10, 50, -100, 100, 100, 800, 0, 400, -1, 0] # Excellent Bounds but x0 was cut off, 'Euro1Test'
#     # Euro_prior_bounds = [0, 90, -100, 100, 100, 800, 0, 400, -1, 0] # 'Euro2' breaks
#     Euro = MCSampler('europa', prior_bounds=Euro_prior_bounds)
#
#     # Io_prior_bounds = [0, 40, 10, 60, 100, 150, 80, 150, 0.4, 0.6] # 'Io2' Good Bounds, theta and x0 cut off
#     # Io_prior_bounds = [5, 50, 10, 60, 100, 150, 0.1, 0.99, -0.55, 0.55] # 'Io Eccentricity 2' good uising ecentricirty model
#     Io_prior_bounds = [-1, 1, -1, 1, 100, 150, 80, 150, 0.2, 0.55]
#     Io = MCSampler('Io', prior_bounds=Io_prior_bounds)
#
#     # print(Io.bestfit_params())
#     Io.run('C:/Users/ishan/PycharmProjects/JovianSatellites/MCMC Backups/Io3')
#     # Euro.run('C:/Users/ishan/PycharmProjects/JovianSatellites/MCMC Backups/Euro2')
#
#     pass

# from data_processing import data as d
# from data_processing.ellipse_fitting import ellipse_fit, convert_to_physical
# from data_processing.rotate_data import Rotate_Data
# from data_processing import ellipse_fitting
# import emcee
# import numpy as np
# import matplotlib.pyplot as plt
# import corner
#
# """ This allows you to run an MC sampler around the parameter space defined in the prior
#     from the list:
#     prior_bounds = [LowerX0, UpperX0, LowerY0, UpperY0, Lower_a, Upper_a, Lower_b, Upper_b, lower_theta, Upper_theta]
#     All Values in list must be floats and theta must be in radians.
#
#      NOTE: We are assuming an error on the positions as +-1 pixel"""
#
# class MCSampler:
#     def __init__(self, moon, prior_bounds):
#         self.name = moon
#         self.prior_bounds = prior_bounds
#
#         J = d.compile[0]
#         angles = d.compile[5]
#         scales = d.compile[6]
#         flips = d.compile[7]
#         time = d.compile[8]
#
#         if self.name.casefold() == 'Io'.casefold():
#             self.index = 1
#         elif self.name.casefold() == 'Europa'.casefold():
#             self.index = 2
#         elif self.name.casefold() == 'Ganymede'.casefold():
#             self.index = 3
#         elif self.name.casefold() == 'Callisto'.casefold():
#             self.index = 4
#         else:
#             raise ValueError("You must input the name of one of Jupiter's Galilean moons")
#
#         ## format data ##
#         moonX = d.compile[self.index][0]
#         moonY = d.compile[self.index][1]
#
#         data_stack = np.column_stack([J[0], J[1], moonX, moonY])
#         valid_mask = ~np.isnan(data_stack).any(axis=1)
#
#         Jx, Jy = J[0][valid_mask], J[1][valid_mask]
#         moonX, moonY = moonX[valid_mask],moonY[valid_mask]
#
#         self.Moon_X = Rotate_Data(Jx, Jy, moonX, moonY, angles)[0]
#         self.Moon_Y = Rotate_Data(Jx, Jy, moonX, moonY, angles)[1]
#         self.xerr = np.ones(len(self.Moon_X))
#         self.yerr = np.ones(len(self.Moon_Y))
#
#     def bestfit_params(self):
#         fit = ellipse_fit(self.Moon_X, self.Moon_Y)
#         A, B, C, D, E, F = fit[0], fit[1], fit[2], fit[3], fit[4], fit[5]
#         x0, y0, a, b, theta = convert_to_physical(A, B, C, D, E, F)
#         self.parameters = x0, y0, a, b, theta
#         return self.parameters
#
#     def log_likelihood_physical(self, params, sigma_x=1.0, sigma_y=1.0):
#         x, y = self.Moon_X, self.Moon_Y
#
#         # Unpack params
#         a, b, theta = params
#
#         x0 = np.random.normal(0, 1)
#         y0 = np.random.normal(0, 1)
#
#         # Evaluate the ellipse function at each (x, y)
#         f_val = ((x - x0) * np.cos(theta) + (y - y0) * np.sin(theta)) ** 2 / a ** 2 + (
#                     -(x - x0) * np.sin(theta) + (y - y0) * np.cos(theta)) ** 2 / b ** 2 - 1
#
#         # Compute the gradients
#         df_dx = (2 * np.cos(theta) * ((x - x0) * np.cos(theta) + (y - y0) * np.sin(theta))) / a ** 2 - (
#                     2 * np.sin(theta) * ((x0 - x) * np.sin(theta) + (y - y0) * np.cos(theta))) / b ** 2
#         df_dy = (2 * np.sin(theta) * ((x - x0) * np.cos(theta) + (y - y0) * np.sin(theta))) / a ** 2 + (
#                     2 * np.cos(theta) * (-(x - x0) * np.sin(theta) - ((y0 - y) * np.cos(theta)))) / b ** 2
#         grad_norm = np.sqrt(df_dx ** 2 + df_dy ** 2)
#
#         # Calculate the orthodonal distance from the data point to the ellipse
#         d = np.abs(f_val) / grad_norm
#
#         # Compute the unit normal components
#         n_x = df_dx / grad_norm
#         n_y = df_dy / grad_norm
#
#         # Calculate the effective uncertainty along the normal direction
#         sigma_normal = np.sqrt((n_x * sigma_x) ** 2 + (n_y * sigma_y) ** 2)
#
#         # Protect against division by zero
#         sigma_normal[sigma_normal == 0] = 1e-10
#
#         # Log likelihood for each data point (assuming independent points)
#         logL = -0.5 * ((d / sigma_normal) ** 2 + np.log(2 * np.pi * sigma_normal ** 2))
#         return np.sum(logL)
#
#     def log_prior_physical(self, params):
#         Lower_a, Upper_a, Lower_b, Upper_b, lower_theta, Upper_theta = self.prior_bounds
#         a, b, theta = params
#         if Lower_a < a < Upper_a and Lower_b < b < Upper_b and lower_theta < theta < Upper_theta:
#             return 0.0
#         return -np.inf
#
#     def log_probability_physical(self, params):
#         lp = self.log_prior_physical(params)
#         if not np.isfinite(lp):
#             return -np.inf
#         return lp + self.log_likelihood_physical(params)
#
#     def run(self, file='C:/Users/ishan/PycharmProjects/JovianSatellites/MCMC Backups/Test'):
#         initials = np.array(self.bestfit_params())[2::]
#         scales = np.array([1e-1, 1e-1, 1e-3])
#         nwalkers = 32
#         ndim = initials.shape[0]
#         pos = initials + np.random.randn(nwalkers, ndim) * scales
#
#         backend = emcee.backends.HDFBackend(file)
#         backend.reset(nwalkers, ndim)
#
#         params = np.array(self.bestfit_params())
#         sampler = emcee.EnsembleSampler(nwalkers, ndim, self.log_probability_physical, backend=backend)
#         sampler.run_mcmc(pos, 15000, progress=True, store=True)
#
#         samples = sampler.get_chain()
#
#         fig, axes = plt.subplots(3, figsize=(10, 7), sharex=True)
#         labels = ["a", "b", "theta"]
#         for i in range(ndim):
#             ax = axes[i]
#             ax.plot(samples[:, :, i], "k", alpha=0.3)
#             ax.set_xlim(0, len(samples))
#             ax.set_ylabel(labels[i])
#             ax.yaxis.set_label_coords(-0.1, 0.5)
#
#         axes[-1].set_xlabel("step number");
#
#         flat_samples = sampler.get_chain(discard=1700, thin=30, flat=True)
#         print(flat_samples.shape)  # This gives you the total number of samples
#
#         fig = corner.corner(flat_samples, labels=labels)
#         plt.show()
#         return samples
#
# if __name__ == "__main__":
#     Gany_prior_bounds = [-50*4, 8*2, -10*2, 10*2, 310*0.5, 400*2, 10, 30, -0.2, 0]
#     Gany = MCSampler('ganymede', prior_bounds=Gany_prior_bounds)
#
#     Euro_prior_bounds = [-10, 50, -100, 100, 100, 800, 0, 400, -1, 0] # Excellent Bounds but x0 was cut off, 'Euro1Test'
#     # Euro_prior_bounds = [0, 90, -100, 100, 100, 800, 0, 400, -1, 0] # 'Euro2' breaks
#     Euro = MCSampler('europa', prior_bounds=Euro_prior_bounds)
#
#     # Io_prior_bounds = [0, 40, 10, 60, 100, 150, 80, 150, 0.4, 0.6] # 'Io2' Good Bounds, theta and x0 cut off
#     # Io_prior_bounds = [5, 50, 10, 60, 100, 150, 0.1, 0.99, -0.55, 0.55] # 'Io Eccentricity 2' good uising ecentricirty model
#
#
#     Io_prior_bounds = [50, 150, 20, 200, -0.55, 0.55]
#     Io = MCSampler('Io', prior_bounds=Io_prior_bounds)
#
#     # print(Io.bestfit_params())
#     Io.run('C:/Users/ishan/PycharmProjects/JovianSatellites/MCMC Backups/TestingSimpleModel')
#     # Euro.run('C:/Users/ishan/PycharmProjects/JovianSatellites/MCMC Backups/Euro2')
#
#     pass
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
            self.color = 'orange'
        elif self.name.casefold() == 'Europa'.casefold():
            self.index = 2
            self.color = 'purple'
        elif self.name.casefold() == 'Ganymede'.casefold():
            self.index = 3
            self.color = 'darkgreen'
        elif self.name.casefold() == 'Callisto'.casefold():
            self.index = 4
            self.color = 'blue'
        else:
            raise ValueError("You must input the name of one of Jupiter's Galilean moons")

        ## format data ##
        moonX = d.compile[self.index][0]
        moonY = d.compile[self.index][1]

        Jx, Jy = J[0], J[1]

        self.Moon_X = Rotate_Data(Jx, Jy, moonX, moonY, angles)[0]
        self.Moon_Y = Rotate_Data(Jx, Jy, moonX, moonY, angles)[1]
        # remove nan values
        self.Moon_X = self.Moon_X[~np.isnan(self.Moon_X)]
        self.Moon_Y = self.Moon_Y[~np.isnan(self.Moon_Y)]
        self.xerr = np.ones(len(self.Moon_X))
        self.yerr = np.ones(len(self.Moon_Y))

    def plot(self):
        plt.scatter(self.Moon_X, self.Moon_Y)
        plt.show()

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
        initials = np.array([0,0, self.bestfit_params()[2], self.bestfit_params()[3], self.bestfit_params()[4]])
        scales = np.array([1e-3, 1e-3, 1e-1, 1e-1, 1e-3])
        nwalkers = 32
        ndim = initials.shape[0]
        pos = initials + np.random.randn(nwalkers, ndim) * scales

        backend = emcee.backends.HDFBackend(file)
        backend.reset(nwalkers, ndim)

        params = np.array(self.bestfit_params())
        sampler = emcee.EnsembleSampler(nwalkers, ndim, self.log_probability_physical, backend=backend)
        sampler.run_mcmc(pos, 15000, progress=True, store=True)

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

    def plot_MCMC(self, archive):
        data = emcee.backends.HDFBackend(archive)
        samples = data.get_chain(discard=300)

        y = np.linspace(0, 1000, 1001)
        z = samples[::, 2, 1]
        # plt.plot(samples[0,0,0],samples[::, 0, 0] , "k", alpha=0.3)
        # plt.ylim(min(samples[::100, 0, 0]), max(samples[::100, 0, 0]))
        # plt.show()

        # fig, axes = plt.subplots(5, figsize=(10, 7), sharex=True)
        # labels = ["A", "B", "C", "D", "E", "F"]
        # for i in range(5):
        #     ax = axes[i]
        #     ax.plot(samples[:, :, i], "k", alpha=0.3)
        #     ax.set_xlim(0, len(samples))
        #     ax.set_ylabel(labels[i])
        #     ax.yaxis.set_label_coords(-0.1, 0.5)

        # axes[-1].set_xlabel("step number");
        flat_samples = data.get_chain(discard=300, thin=300, flat=True)
        # fig = corner.corner(flat_samples, labels=labels)

        x0 = np.percentile(flat_samples[:, 0], [16, 50, 84])
        y0 = np.percentile(flat_samples[:, 1], [16, 50, 84])
        a = np.percentile(flat_samples[:, 2], [16, 50, 84])
        b = np.percentile(flat_samples[:, 3], [16, 50, 84])
        theta = np.percentile(flat_samples[:, 4], [16, 50, 84])


        # Suppose these are your ellipse parameters from the MCMC median estimates:
        x0_model = x0[1]  # median x-center from your MCMC
        y0_model = y0[1]  # median y-center from your MCMC
        a_model = a[1]  # median semi-major axis
        b_model = b[1]  # median semi-minor axis
        theta_model = theta[1]  # median rotation angle

        # Define the ellipse equation:
        def ellipse_eq(x, y, x0, y0, a, b, theta):
            term1 = ((x - x0) * np.cos(theta) + (y - y0) * np.sin(theta)) ** 2 / a ** 2
            term2 = ((-(x - x0) * np.sin(theta) + (y - y0) * np.cos(theta)) ** 2) / b ** 2
            return term1 + term2 - 1  # ellipse is defined by ellipse_eq == 0

        # Define the plotting grid.

        N = 2000
        x_min, x_max = -800, 800  # use names that don't conflict with x0_model, y0_model
        y_min, y_max = -150, 150
        xs = np.linspace(x_min, x_max, N)
        ys = np.linspace(y_min, y_max, N)
        X, Y = np.meshgrid(xs, ys)

        # Evaluate the ellipse equation on the grid.
        Z = ellipse_eq(X, Y, x0_model, y0_model, a_model, b_model, theta_model)

        # Plot the contour corresponding to the ellipse (where Z == 0).

        inds = np.random.randint(len(flat_samples), size=100)
        for ind in inds:
            x0 = flat_samples[ind, 0]
            y0 = flat_samples[ind, 1]
            a = flat_samples[ind, 2]
            b = flat_samples[ind, 3]
            theta = flat_samples[ind, 4]
            Z = ellipse_eq(X, Y, x0, y0, a, b, theta)
            plt.contour(X, Y, Z, levels=[0], colors=self.color, alpha=0.1)

        plt.errorbar(self.Moon_X, self.Moon_Y, xerr=np.ones(len(self.Moon_X)), yerr=np.ones(len(self.Moon_Y)),
                     ls='none', color=self.color, capsize=3, markersize=5)
        plt.contour(X, Y, Z, levels=[0], colors="red")

        # plt.scatter(self.Moon_X, self.Moon_Y, color= self.color)

        plt.xlabel("X (arcsec)")
        plt.ylabel("Y (arcsec)")

    def corner_plot(self, archive):
        initials = np.array([0, 0, self.bestfit_params()[2], self.bestfit_params()[3], self.bestfit_params()[4]])
        ndim = initials.shape[0]

        data = emcee.backends.HDFBackend(archive)
        samples = data.get_chain(discard=300)
        fig, axes = plt.subplots(5, figsize=(10, 7), sharex=True)
        labels = ["x0", "y0", "a", "b", "theta"]
        for i in range(ndim):
            ax = axes[i]
            ax.plot(samples[:, :, i], "k", alpha=0.3)
            ax.set_xlim(0, len(samples))
            ax.set_ylabel(labels[i])
            ax.yaxis.set_label_coords(-0.1, 0.5)

        axes[-1].set_xlabel("step number");

        flat_samples = data.get_chain(discard=1700, thin=30, flat=True)
        print(flat_samples.shape)  # This gives you the total number of samples

        fig = corner.corner(flat_samples, labels=labels)
        plt.show()



if __name__ == "__main__":
    Gany_prior_bounds = [-1, 1, -1, 1, 200, 450, 5, 45, -np.pi, np.pi] # 'GanyTest1' Good bounds and produce good ellipse but x0 is unconstrained
    # Gany_prior_bounds = [-4, 4, -1, 1, 200, 450, 5, 45, -np.pi, np.pi] # 'Gany Test2' Beggining to get unconstrained
    # Gany_prior_bounds = [-10, 2, -1, 1, 200, 450, 5, 45, -np.pi, np.pi] # 'Ganytest3' oops i overwrote it, buyt now x0 constrained and y0 is not
    Gany = MCSampler('ganymede', prior_bounds=Gany_prior_bounds)

    Euro_prior_bounds = [-1, 1, -1, 1, 150, 450, 0, 100, -np.pi, np.pi] # 'EuroTest3'
    Euro = MCSampler('europa', prior_bounds=Euro_prior_bounds)

    Io_prior_bounds = [-1, 1, -1, 1, 100, 300, 0, 60, -np.pi, np.pi] # 'IoTest1'
    Io = MCSampler('Io', prior_bounds=Io_prior_bounds)

    Calli_prior_bounds = [-1, 1, -1, 1, 300, 800, 20, 80, -np.pi, np.pi] # 'CalliTest1'
    Calli_prior_bounds = [-2, 2, -2, 2, 300, 800, 20, 80, -np.pi, np.pi] # 'CalliTest2'
    Calli = MCSampler('Callisto', prior_bounds = Calli_prior_bounds)

    Gany.run('C:/Users/ishan/PycharmProjects/JovianSatellites/MCMC Backups/GanyTest3')
    Gany.plot_MCMC('C:/Users/ishan/PycharmProjects/JovianSatellites/MCMC Backups/GanyTest3')

    # Euro.run('C:/Users/ishan/PycharmProjects/JovianSatellites/MCMC Backups/EuroTest3')
    # Euro.plot_MCMC('C:/Users/ishan/PycharmProjects/JovianSatellites/MCMC Backups/EuroTest3')

    # Calli.run('C:/Users/ishan/PycharmProjects/JovianSatellites/MCMC Backups/CalliTest2')


    ############# PLOT #################
    Io.corner_plot('C:/Users/ishan/PycharmProjects/JovianSatellites/MCMC Backups/IoTest2')

    # Calli.plot_MCMC('C:/Users/ishan/PycharmProjects/JovianSatellites/MCMC Backups/CalliTest1')
    # Gany.plot_MCMC('C:/Users/ishan/PycharmProjects/JovianSatellites/MCMC Backups/GanyTest1')
    # Euro.plot_MCMC('C:/Users/ishan/PycharmProjects/JovianSatellites/MCMC Backups/EuroTest3')
    # Io.plot_MCMC('C:/Users/ishan/PycharmProjects/JovianSatellites/MCMC Backups/IoTest2')
    #
    # proxy = [plt.Rectangle((0, 0), 1, 1, fc='orange'), plt.Rectangle((0, 0), 1, 1, fc='purple'),
    #          plt.Rectangle((0, 0), 1, 1, fc='green'),
    #          plt.Rectangle((0, 0), 1, 1, fc='blue'), plt.axhline(0, 0, 0, color = 'red')]  # , plt.Rectangle((0,0),1,1,fc = 'red')]
    # plt.legend(proxy, ['Io', 'Europa', 'Ganymede', 'Callisto', 'Mean Param Fit'], fontsize=18)
    #
    # plt.ylim(-90, 90)
    # plt.xlim(-650, 650)
    # plt.xlabel("X (Arcsec)", fontsize=18)
    # plt.ylabel("Y (Arcsec)", fontsize=18)
    # plt.scatter(0, 0, color='red', label='Jupiter')
    # plt.tick_params(axis='both', which='major', labelsize=16, length=8, width=1.5, top=True, right=True)
    # plt.tick_params(axis='both', which='minor', labelsize=16, length=4, width=1, top=True, right=True)
    # plt.minorticks_on()

    plt.show()

    pass