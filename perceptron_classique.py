"""
Fichier : perceptron_classique.py
Author : G.MENEZ (2025)
"""
import numpy as np
import matplotlib.pyplot as plt
import a_hyperplan as ah
import a_hyperplan_side as ahs


# ======================================
def perceptron_predict(X, w, b, fa):
    """
    Rend les prédictions des entrées X opérées par un perceptron
    (w,b) utilisant une fonction d'activation fa.
    Args:
        X (ndarray): Matrice d'entrées (N x d).
        w (ndarray): Poids ajustés (d,).
        b (float): Biais ajusté.
        fa : fonction d'activation
    Returns:
        ndarray: Probabilités prédictives (N,).
    """
    z = np.dot(X, w) + b
    return fa(z)


# ======================================
def perceptron_train(X, y, winit, binit, fa, learning_rate=0.01, epochs=1000, ax=None):
    """
    Entraîne un perceptron avec une fonction de décision fa
    Args:
        X (ndarray): Matrice d'entrée (N x d), où N est le nombre d'exemples et d le nombre de caractéristiques.
        y (ndarray): Étiquettes correspondantes (N,). Les valeurs doivent être -1 ou 1.
        winit, binit : hyperplan initial
        learning_rate (float): Taux d'apprentissage.
        epochs (int): Nombre d'itérations sur les données.
    Returns:
        w (ndarray): Poids ajustés (d,).
        b (float): Biais ajusté.
        errors (list): Liste des erreurs de classification au fil des époques.
    """
    w = winit
    b = binit
    errors = []
    n_samples, n_features = X.shape
    for epoch in range(epochs):
        errors.append(0)
        for i in range(n_samples):  # Pour chaque Xi
            # Calcul de la prédiction
            y_pred = perceptron_predict(X[i], w, b, fa)
            # On en déduit l'erreur
            err = y[i] - y_pred
            # Mise à jour des poids et du biais
            w += learning_rate * err * X[i]
            b += learning_rate * err
            if err != 0:  # On comptabilise le nombre d'erreur à cette epoch
                errors[epoch] += 1

        if (errors[epoch] == 0):  # Le perceptron defini classe sans erreur
            break  # alors on arrete

        if ax is not None:  # On dessine l'hyperplan résultant de l'entrainement
            ww = [b, w[0], w[1]]
            ah.plot_hyperplan2D(ax, ww, [-5, 6], f'Hyperplane:{epoch} {np.flip(w)} {b}')

    return w, b, errors


# ========================================
if __name__ == "__main__":
    fig, axs = plt.subplots(1, 2)  # one rwo , two col

    # Données d'exemple (2 caractéristiques, 4 exemples)
    X = np.array([
        [0, 5],
        [0, 2],
        [-1, 3],
        [2, 3],
        [2, 6],
        [5, 5],
        [2, 2]  # ce point conditionne l'inclinaison de l'hyperplan .. -1/1
        # et il influe sur la convergence => voir la courbre des erreurs
    ])
    y = np.array([1, -1, -1, 1, 1, 1, -1])  # Étiquettes vérités

    # Caractéristiques de l'Entraînement
    learning_rate = 1
    epochs = 20  # nombre d'epochs
    n_samples, n_features = X.shape

    # Hyperplan initial
    winit = np.zeros(n_features)
    binit = 0
    ah.plot_hyperplan2D(axs[0], np.hstack((winit, [binit])))

    # Entrainement de l'hyperplan
    w, b, errors = perceptron_train(X, y, winit, binit, np.sign, learning_rate=learning_rate, epochs=epochs, ax=axs[0])
    print(f"Poids ajustés: {w}")
    print(f"Biais ajusté: {b}")

    # On dessine l'hyperplan résultant de l'entrainement
    ww = [b, w[0], w[1]]
    ah.plot_hyperplan2D(axs[0], ww)

    # On place les points
    c = ahs.classify_thispoints(X, ww)
    ah.plot_points(axs[0], X, c)
    axs[0].legend()
    axs[0].grid(True)

    # Visualisation de la convergence de l'erreur
    axs[1].plot(range(len(errors)), errors, "o-", label="Erreurs de classification")
    axs[1].set_xlabel("Époques")
    axs[1].set_ylabel("Nombre d'erreurs sur l'epoch")
    axs[1].set_title("Convergence de l'erreur")
    axs[1].legend()
    axs[1].grid(True)

    # Prédictions on tests
    X_test = np.array([
        [1, 2],
        [3, 3],
        [0, 0]
    ])
    # Prédiction des classes des tests points
    predictions = perceptron_predict(X_test, w, b, np.sign)
    print(f"Prédictions (classes) : {predictions}")

    ah.plot_points(axs[0], X_test, defcolor="orange")  # Plot tests points
    axs[0].legend()
    axs[0].grid(True)
    plt.show()