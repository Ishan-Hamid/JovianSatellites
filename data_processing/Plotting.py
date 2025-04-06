import data_processing.data as d
from data_processing.rotate_data import Rotate_Data
import matplotlib.pyplot as plt

def plot(index):
    moonX = d.compile[index][0]
    moonY = d.compile[index][1]

    print(moonX)
    Jx = d.compile[0][0]
    Jy = d.compile[0][1]

    p = Rotate_Data(Jx, Jy, moonX, moonY, d.theta)

    print(p)

    plt.scatter(p[0], p[1])


if __name__ == '__main__':
    plot(2)
    plot(1)
    plot(4)
    plot(3)
    plt.show()
