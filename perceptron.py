"""
perceptron.py
Author :  G.MENEZ (2025)
"""
import numpy as np
import matplotlib.pyplot as plt
import a_hyperplan as ah
import a_hyperplan as ahs

#======================================
def sigmoid(x, a=1):
    """  Fonction sigmoïde. """
    return 1 / (1 + np.exp(-a*x))

def sigmoid_derivative(x, a=1):
    """  Dérivée de la fonction sigmoïde
    https://fr.wikipedia.org/wiki/Sigmo%C3%AFde_(math%C3%A9matiques)
    https://www.youtube.com/watch?v=u7uCoN9GwH4
    """
    return a*sigmoid(x) * (1 - sigmoid(x))

#======================================
def forward_activation(X, w, b, fa=sigmoid):
    """Rend la sortie d'un perceptron (w,b) pour des entrées X en
    utilisant une fonction d'activation (sigmoïde par défaut).
    Args:
        X (ndarray) : Matrice d'entrées (m x d).
        w (ndarray) : Poids  (d,).
        b (float)       : Biais .
        fa : Fonction d'activation
    Returns:
        ndarray: Probabilités prédictives (N,) si fa est la sigmoid.

    """
    z = np.dot(X, w) + b
    return fa(z)

#======================================
def predict(X_test, w, b, fa) :
    """ Prediction de la classe d'un ensemble d'éléments  """

    linear_model = np.dot(X_test, w) + b
    y_pred = fa(linear_model)

    # On seuil cette proba pour obtenir la classe => o ou 1
    predictions = np.where(y_pred >= 0.5, 1, 0)
    return predictions

#======================================
def mse_loss(y_true, y_pred):
    """ mse  loss function"""
    loss =np.mean((y_true - y_pred) ** 2)
    return loss

#======================================
def log_loss(y_true, y_pred, epsilon = 1e-10):
    """ log loss function """
    y_pred = np.clip(y_pred, epsilon, 1 - epsilon) # eviter que certains des termes qui suivent soient nuls.
    loss = -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))
    return loss

#======================================
def backward_propagation_mse(X, y, yp):
    """  Calcul des gradients pour la propagation arrière avec mse"""
    m,d = X.shape
    dW = (1 / m) * np.dot(X.T, (yp - y) * yp * (1 - yp))
    db = (1 / m) * np.sum((yp - y) * yp * (1 - yp))
    return dW, db

#======================================
def backward_propagation_log(X, y, yp):
    """  Calcul des gradients pour la propagation arrière avec log loss"""
    m,d = X.shape
    dW = (1 / m) * np.dot(X.T, (yp - y))
    db = (1 / m) * np.sum((yp - y))
    return dW, db

#======================================
def update_parameters(W, b, dW, db, learning_rate):
    """ Mise à jour des poids """
    W -= learning_rate * dW
    b -= learning_rate * db
    return W, b

#======================================
def perceptron_train(X, y, winit, binit, fa=sigmoid, learning_rate=0.01, epochs=1000):
    """
    Entraîne un perceptron avec une fonction de décision sigmoïde (par defaut).

    Args:
        X (ndarray) : Matrice d'entrée (m x d), où m est le nombre d'exemples et d le nombre de caractéristiques.
        y (ndarray) : Étiquettes correspondantes (m,). Les valeurs doivent être 0 ou 1.
        winit, binit : Modèle initial,
        learning_rate (float) : Taux d'apprentissage.
        epochs (int) : Nombre d'itérations sur les données.

    Returns:
        w (ndarray): Poids ajustés (d,).
        b (float): Biais ajusté.
        errors (list): Liste des erreurs quadratiques au fil des époques.
    """
    W = winit
    b =  binit
    losses = []

    for epoch in range(epochs):
        # Calcul de la prédiction probabiliste
        y_out = forward_activation(X, W, b, fa)

        # Calcul des gradients
        dW, db  = backward_propagation_mse(X, y, y_out)

        # Mise à jour des poids et du biais
        W ,b = update_parameters(W, b, dW, db, learning_rate)

        # Stocker l'erreur quadratique moyenne pour cette époque
        losses.append(mse_loss(y, y_out))

    return W, b, losses


#========================================
if __name__ == "__main__":

    # Données d'exemple (2 caractéristiques, 4 exemples)
    X = np.array([
        [2, 3],
        [1, 1],
        [2, 1],
        [3, 2]
    ])
    y = np.array([1, 0, 0, 1])  # Étiquettes correspondantes (0 ou 1)

    fig, (ax1, ax2) = plt.subplots(1, 2) #one rwo , two col

    # Entraînement
    epochs=1000
    n_samples, n_features = X.shape
    winit = np.zeros(n_features) # b est w[0]
    binit  = 0
    ah.plot_hyperplan2D(ax1 ,np.hstack((winit,[binit])))

    w, b, errors = perceptron_train(X, y, winit, binit, learning_rate=0.1, epochs=epochs)
    print(f"Poids ajustés: {w}")
    print(f"Biais ajusté: {b}")

    ww = [b, w[0], w[1]]
    c = ahs.classify_thispoints(X, ww)
    ah.plot_points(ax1, X,c)
    ah.plot_hyperplan2D(ax1 ,ww)

    # Visualisation de la convergence de l'erreur
    ax2.plot(range(epochs), errors, label="Erreur quadratique")
    ax2.set_xlabel("Époques")
    ax2.set_ylabel("Erreur quadratique moyenne")
    ax2.set_title("Convergence de l'erreur")
    ax2.legend()
    ax2.grid(True)

    # Prédictions
    X_test = np.array([
        [1, 2],
        [3, 3],
        [2, 2],
        [1,0.5]
    ])
    predictions = predict(X_test, w, b, fa=sigmoid)
    print(f"Prédictions (classes) : {predictions}")

    ah.plot_points(ax1, X_test, defcolor="orange")
    ax1.legend()
    ax1.grid(True)

    plt.show()