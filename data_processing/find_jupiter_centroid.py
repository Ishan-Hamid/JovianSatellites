import numpy as np
from data_processing.ellipse_fitting import polyToParams, ellipse_fit
import matplotlib.pyplot as plt

class Centroid:

    def __init__(self, file):
        self.file = file

        with open(self.file, 'r') as f:
            content = f.read()

        # Split the content into datasets based on double blank lines
        data_sets = content.strip().split("\n\n")

        # Process each dataset into a list of lists (convert to float)
        self.all_data = [
            [list(map(float, line.split())) for line in dataset.strip().split("\n")]
            for dataset in data_sets
        ]

    def All_Data(self):
        return self.all_data

    def centroid(self):

        data_file = self.all_data

        N = len(data_file)
        centroid_xs = np.zeros(N)
        centroid_ys = np.zeros(N)
        for i in range(N):
            data_set = np.array([item for sublist in data_file[i] for item in sublist])
            x = data_set[0::2]
            y = data_set[1::2]
            centroid_x = polyToParams(ellipse_fit(x, y), False)[0]
            centroid_y = polyToParams(ellipse_fit(x, y), False)[1]
            centroid_xs[i] = centroid_x
            centroid_ys[i] = centroid_y

        mean_x = np.mean(centroid_xs)
        mean_y = np.mean(centroid_ys)
        error_x = np.std(centroid_xs)/np.sqrt(N)
        error_y = np.std(centroid_ys)/np.sqrt(N)

        print('The Mean X = ', mean_x, '+-',error_x)
        print('The Mean Y = ', mean_y, '+-',error_y)

        return [mean_x, mean_y, error_x, error_y]

    def eccentricity(self):
        data_file = self.all_data
        N = len(data_file)
        ax1s = np.zeros(N)
        ax2s = np.zeros(N)
        for i in range(N):
            data_set = np.array([item for sublist in data_file[i] for item in sublist])
            x = data_set[0::2]
            y = data_set[1::2]
            ax1 = polyToParams(ellipse_fit(x, y), False)[2]
            ax2 = polyToParams(ellipse_fit(x, y), False)[3]
            ax1s[i] = ax1
            ax2s[i] = ax2

        mean_ax1 = np.mean(ax1s)
        mean_ax2 = np.mean(ax2s)
        error_ax1 = np.std(ax1s) / np.sqrt(N)
        error_ax2 = np.std(ax2s) / np.sqrt(N)

        eccentricity = np.sqrt(1 - (mean_ax2 / mean_ax1) ** 2)
        error_eccentricity = np.sqrt(
            abs(np.sqrt(1 - ((mean_ax2 + error_ax2) / mean_ax1) ** 2) - eccentricity) ** 2 + abs(
                np.sqrt(1 - (mean_ax2 / (mean_ax1 + error_ax1)) ** 2) - eccentricity) ** 2)
        print('The Mean eccentricity = ', eccentricity, '+-', error_eccentricity)

        return [eccentricity, error_eccentricity]

    def plot(self, index):

        ## plot of model fitting an ellipse to the iso-contours of Jupiter ##

        packet = self.all_data
        data_set = np.array(packet[index]).flatten()
        x = data_set[0::2]
        y = data_set[1::2]
        print(x, y)

        def model_ellipse(x, y):
            v = ellipse_fit(data_set[0::2], data_set[1::2])
            A, B, C, D, E, F = v[0], v[1], v[2], v[3], v[4], v[5]
            return A * x ** 2 + B * x * y + C * y ** 2 + D * x + E * y + F

        ###### ellipse plot #######
        N = 1000
        x0, x1 = 100, 1000
        y0, y1 = 100, 1000
        xs = np.linspace(x0, x1, N)
        ys = np.linspace(y0, y1, N)
        mesh = np.meshgrid(xs, ys)

        emesh = model_ellipse(mesh[0], mesh[1])

        plt.contour(mesh[0], mesh[1], emesh, 10)
        plt.scatter(x, y, marker='x', color='black')
        plt.scatter(self.centroid()[0], self.centroid()[1], marker='x', color='red')
        plt.ylim(min(y)-10,max(y)+10)
        plt.xlim(min(x)-10,max(x)+10)

        return

if __name__ == "__main__":
    C1 = Centroid('C:/Users/ishan/Desktop/Uni Files/Year 3/Labs/Jupiter Data/test_2.con')
    # print(C1.centroid())
    # # print(C1.eccentricity())
    C1.plot(0)
    C1.plot(1)
    plt.show()
    pass