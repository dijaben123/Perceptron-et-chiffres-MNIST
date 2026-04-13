"""
File :a_hyperplan_side3D
Author :  G.MENEZ
"""
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import a_hyperplan as ah
import a_hyperplan_side as ahs


# ======================================
def plot_hyperplan3D(ax, hc):
    """
    Plot the hyperplane in 3D :
    hc are coefficients of hyperplane (w0,w1,w2,w3)
    """
    X, Y = np.meshgrid(range(-5, 6), range(-5, 6))  # On pourrait mettre ca en parametre
    Z = (-hc[0] - hc[1] * X - hc[2] * Y) / hc[3]  # Explain Z as a function of X and Y

    ax.plot_surface(X, Y, Z, alpha=0.8, color='green', label='Hyperplane')


# ======================================
def make_specific_points():
    """
    Make three Specific points for the hyperplane to the demo
    """
    point_on_hyperplane = np.array([1, -2, 0])  # x1, x2, x3
    point_positive_side = np.array([3, 0, -5])
    point_negative_side = np.array([-3, -2, 6])
    points = np.array([point_on_hyperplane, point_positive_side, point_negative_side])
    return points


def randrange(n, vmin, vmax):
    """
    Helper function to make an array of random numbers having shape (n, )
    with each number distributed Uniform(vmin, vmax).
    """
    return (vmax - vmin) * np.random.rand(n) + vmin


def random_population(xbox, ybox, zbox, n=10):
    """
    return n points in a 3D box with an Uniform random draw.
    """
    low, high = xbox
    xs = randrange(n, low, high)
    low, high = ybox
    ys = randrange(n, low, high)
    low, high = zbox
    zs = randrange(n, low, high)

    return xs, ys, zs


def shuffle_forward(l):
    np.random.seed(42)  # ou autre choose que 42  ... qui depend du temps ?
    order = np.arange(len(l))
    print("Indices originaux:", order)
    np.random.shuffle(order)
    print("Indices mélangés:", order)
    return l[order], order  # Réorganiser le tableau en utilisant les indices mélangés


def shuffle_backward(l, order):
    tmp = np.zeros(l.shape[0], dtype=int)
    tmp[order] = np.arange(l.shape[0])
    l_out = l[tmp]
    return l_out


# ======================================
def main1(ax, hc, p):
    plot_hyperplan3D(ax, hc)  # Plot the hyperplane in 3D
    ax.view_init(elev=30, azim=9, roll=0)  # pour mieux le voir

    c = ahs.classify_thispoints(p, hc)
    # print(c)
    ah.plot_points(ax, p, c)


# ======================================
def main2(ax, hc, xsn, ysn, zsn, xsp, ysp, zsp):
    # Plot les éléments en partie neg
    ax.scatter(xsn, ysn, zsn, marker='o', color='tab:red', label="Negative Side")  # Plot hem
    red_proxy = plt.Rectangle((0, 0), 1, 1, fc="tab:red")

    # Plot les éléments en partie pos
    ax.scatter(xsp, ysp, zsp, marker='^', color='tab:blue', label="Positive Side")  # Plot hem
    blue_proxy = plt.Rectangle((0, 0), 1, 1, fc="tab:blue")

    #  Plot the hyperplane in 3D
    plot_hyperplan3D(ax, hc)
    ax.view_init(elev=8, azim=36, roll=0)  # with the good observation point of view

    # Setting plot limits and labels
    # ax.set_xlim([-5, 5])
    # ax.set_ylim([-5, 5])
    # ax.set_zlim([-5, 5])
    ax.set_xlabel('X axis')
    ax.set_ylabel('Y axis')
    ax.set_zlabel('Z axis')

    # https://stackoverflow.com/questions/5803015/how-to-create-a-legend-for-3d-bar/5807175#5807175
    ax.legend([red_proxy, blue_proxy], ['Negative Side', 'Positive Side'])


# ======================================
if __name__ == "__main__":
    # Visualization
    fig = plt.figure(figsize=plt.figaspect(0.5))  # set up a figure twice as wide as it is tall
    # set up the Axes for two plots
    ax1 = fig.add_subplot(1, 2, 1, projection='3d')
    ax2 = fig.add_subplot(1, 2, 2, projection='3d')

    # Hyperplane coefficients :  w0,w1,w2,w3
    hc = np.array([3, -1, 1, 1])

    """ ============ main1 =================="""
    p = make_specific_points()  # Make few Specific points
    main1(ax1, hc, p)

    """ ============ main2 =================="""
    """ On génére deux populations :                                               """
    # éléments en partie neg
    xsn, ysn, zsn = random_population(xbox=(2, 4), ybox=(-4, -2), zbox=(-2, 0), n=20)
    # éléments en partie pos
    xsp, ysp, zsp = random_population(xbox=(-2, 0), ybox=(2, 4), zbox=(0, 2), n=20)
    main2(ax2, hc, xsn, ysn, zsn, xsp, ysp, zsp)

    plt.show()

    """  == On rajoute les étiquettes aux éléments = """
    esn = np.zeros(xsn.shape[0])  # positive c'est 1 et
    alln = np.column_stack((xsn, ysn, zsn, esn))

    esp = np.ones(xsp.shape[0])  # negative c'est 0
    allp = np.column_stack((xsp, ysp, zsp, esp))

    all = np.vstack((alln, allp))
    print(all)

    """ ==== On mélange les classes dans les données. ========= """
    shuffled_all, order = shuffle_forward(all)
    print("Tableau mélangé:\n", shuffled_all)

    # tmp = shuffle_backward(shuffled_all, order)
    # print("Tableau remis en ordre:\n", tmp)

    """ === TODO : Faut classifier les élements de "shuffled_all"  === """
    # selon leur position / Hyperplan ... et vous avez la vérité pour vérifier vos résultats !
