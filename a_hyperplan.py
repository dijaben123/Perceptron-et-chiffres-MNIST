"""
Fichier : a_hyperplan.py
Auteur : G.MENEZ / 2025
"""
import numpy as np
import matplotlib.pyplot as plt


# ======================================
def plot_points(ax, points, classes=None, defcolor='gray'):
    """ Plot each point and annotate the side for each """
    for i, p in enumerate(points):
        if classes is None:
            ax.scatter(*p, color=defcolor)
        elif classes[i] == "+":
            ax.scatter(*p, color='blue')
            ax.text(*p + 0.1, "Positive Side")
        elif classes[i] == "-":
            ax.scatter(*p, color='red')
            ax.text(*p + 0.1, "Negative Side")
        else:
            ax.scatter(*p, color='green')
            ax.text(*p + 0.1, 'On Hyperplane')


# ======================================
def plot_hyperplan2D(ax, hc, box=[-2, 2], lab=None):
    """
    Plot the points on the hyperplane in 2D space
    hc are coefficients of hyperplane (w0,w1,w2)
    """
    w0, w1, w2 = hc

    # On prend quelques points sur l'hyperplan w2*y + w1*x+ w0  = 0
    x1 = np.linspace(box[0], box[1], 400)  # Create a range of x1 values
    if w2 != 0:
        x2 = - (w1 * x1 + w0) / w2  # les x2 qui permettent d'appartenir à l'hyperplan
    else:
        x2 = np.zeros(400)
    # On plot ces points
    if lab is None:
        lab = f'Hyperplane:  {w2:.2f}*x2 + {w1:.2f}*x1 +{w0:.2f} = 0'
    ax.plot(x1, x2, label=lab)


# ======================================
def plot_decorate(ax):
    """ Pour décorer les axes """
    # Setting labels
    ax.set_xlabel('X axis')
    ax.set_ylabel('Y axis')

    # Move the spines to go through the origin
    ax.axhline(0, color='black', linewidth=0.5)
    ax.axvline(0, color='black', linewidth=0.5)

    ax.spines['left'].set_position('zero')
    ax.spines['bottom'].set_position('zero')
    ax.spines['right'].set_color('none')
    ax.spines['top'].set_color('none')

    # Set the aspect of the plot to be equal
    ax.axis('equal')

    # Setting plot limits for better visualization
    ax.set_xlim([-2, 2])
    ax.set_ylim([-2, 5])

    # Adding a legend
    ax.legend()


# ======================================
def plot_normal_vect(ax, origin, normal_vec):
    """ Plot the normal vector """

    l = f'Normal Vector : {normal_vec}'
    # Plot the normal vector with precise control over its length
    ax.quiver(*origin, *normal_vec, scale_units='xy', angles='xy',
              scale=1,
              color='red', label=l)

    # Display the value of the normal vector on the plot
    ax.text(1.2, 1.2, l, color='red')


# ======================================
def main(axs):
    ax1, ax2 = axs

    """ ============ in ax1 =================="""
    # Droite :  y =  mx + d
    m = -2
    d = 3
    x = np.linspace(-2, 2, 400)  # Create a range of x values
    y = m * x + d  # Calculate the corresponding y values based on the updated hyperplane equation
    ax1.plot(x, y, label=f'Droite : y = {m}*x + {d}')
    plot_decorate(ax1)

    """ ============ in ax2 =================="""
    # Pour montrer l'équivalence  avec la représentation "Droite"
    # Hyperplane :   w2*x2+w1*x1+w0 = 0
    # a partir de  y - mx - d = 0 on pose
    w0 = -d
    w1 = -m
    w2 = 1

    origin = [0, 0]  # Origin point for the normal vector
    normal_vec = np.array([w1, w2])  # The normal vector to the hyperplane is [w1, w2]
    plot_normal_vect(ax2, origin, normal_vec)
    plot_hyperplan2D(ax2, [w0, w1, w2])
    plot_decorate(ax2)


# ======================================
if __name__ == "__main__":
    # Create the 2D plot with the axes going through the origin (0, 0)
    fig, axs = plt.subplots(1, 2)  # one row , two cols

    main(axs)

    # Show the plot
    plt.show()