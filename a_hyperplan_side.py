import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import a_hyperplan as ah


# ======================================
def make_points():
    """ Make few Specific points """
    point_on_hyperplane = np.array([0, 3])  # x1,x2
    point_positive_side = np.array([2, 3])
    point_negative_side = np.array([1, 0])
    points = np.array([point_on_hyperplane,
                       point_positive_side,
                       point_negative_side])
    return points


# ======================================
def classify_thispoints(points, hc):
    """ Check the side for all points given the hyperplan hc """
    classes = []
    for i, point in enumerate(points):

        value = np.dot(hc[1:], point) + hc[0]  # Define the side of a point given the hc

        if value > 0:
            classes.append("+")
        elif value < 0:
            classes.append("-")
        else:
            classes.append("=")

    return classes


# ======================================
def main(hc):
    # Create the 2D plot with the axes going through the origin (0, 0)
    fig, axs = plt.subplots(1, 2)  # one rwo , two col
    ah.main(axs)

    p = make_points()
    c = classify_thispoints(p, hc)
    print(c)

    ah.plot_points(axs[1], p, c)


# ======================================
if __name__ == "__main__":
    # 2D Hyperplane coefficients : w0, w1, w2
    hc = np.array([-3, 2, 1])

    main(hc)

    # Show the plot
    plt.show()