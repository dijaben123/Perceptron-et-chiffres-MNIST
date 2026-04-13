import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import time
from sklearn.metrics import accuracy_score, confusion_matrix
import dataquest_mnist as dm


def perceptron_classique_predict(X, w, b):
    z = np.dot(X, w) + b
    return np.sign(z)


def perceptron_classique_train(X, y, learning_rate=0.01, epochs=100):
    n_samples, n_features = X.shape
    w = np.zeros(n_features)
    b = 0
    errors = []

    for epoch in range(epochs):
        epoch_errors = 0
        for i in range(n_samples):
            y_pred = perceptron_classique_predict(X[i:i+1], w, b)[0]
            err = y[i] - y_pred
            if err != 0:
                w += learning_rate * err * X[i]
                b += learning_rate * err
                epoch_errors += 1

        errors.append(epoch_errors)
        if epoch_errors == 0:
            print(f"  Convergence atteinte à l'époque {epoch+1}")
            break

    return w, b, errors


def sigmoid(x, a=1):
    return 1 / (1 + np.exp(-np.clip(a*x, -500, 500)))


def perceptron_mse_predict(X, w, b):
    z = np.dot(X, w) + b
    return sigmoid(z)


def backward_propagation_mse(X, y, yp):
    m, d = X.shape
    dW = (1 / m) * np.dot(X.T, (yp - y) * yp * (1 - yp))
    db = (1 / m) * np.sum((yp - y) * yp * (1 - yp))
    return dW, db


def mse_loss(y_true, y_pred):
    return np.mean((y_true - y_pred) ** 2)


def perceptron_mse_train(X, y, learning_rate=0.01, epochs=100):
    n_samples, n_features = X.shape
    w = np.zeros(n_features)
    b = 0
    losses = []

    for epoch in range(epochs):
        y_out = perceptron_mse_predict(X, w, b)
        dW, db = backward_propagation_mse(X, y, y_out)
        w -= learning_rate * dW
        b -= learning_rate * db
        loss = mse_loss(y, y_out)
        losses.append(loss)

        if epoch % 20 == 0:
            print(f"    Époque {epoch}: Loss = {loss:.6f}")

    return w, b, losses


def perceptron_logloss_predict(X, w, b):
    z = np.dot(X, w) + b
    return sigmoid(z)


def backward_propagation_logloss(X, y, yp):
    m, d = X.shape
    dW = (1 / m) * np.dot(X.T, (yp - y))
    db = (1 / m) * np.sum((yp - y))
    return dW, db


def logloss(y_true, y_pred, epsilon=1e-10):
    y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
    loss = -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))
    return loss


def perceptron_logloss_train(X, y, learning_rate=0.01, epochs=100):
    n_samples, n_features = X.shape
    w = np.zeros(n_features)
    b = 0
    losses = []

    for epoch in range(epochs):
        y_out = perceptron_logloss_predict(X, w, b)
        dW, db = backward_propagation_logloss(X, y, y_out)
        w -= learning_rate * dW
        b -= learning_rate * db
        loss = logloss(y, y_out)
        losses.append(loss)

        if epoch % 20 == 0:
            print(f"    Époque {epoch}: Loss = {loss:.6f}")

    return w, b, losses


def convert_to_binary_classification(y, digit_pos, digit_neg):
    y_binary = np.zeros_like(y)
    y_binary[y == digit_pos] = 1
    y_binary[y == digit_neg] = 0
    mask = (y == digit_pos) | (y == digit_neg)
    return y_binary, mask


def flatten_images(images):
    n_samples = len(images)
    return np.array([img.flatten() for img in images])


if __name__ == "__main__":
    print("\n" + "="*70)
    print("COMPARAISON DES 3 MÉTHODES DE CLASSIFICATION SUR MNIST")
    print("="*70)

    img_path = '../../MNIST/trainingSet/trainingSet/'
    digit_pos = 1
    digit_neg = 0
    test_size = 500

    print(f"\nChargement des données MNIST...")
    try:
        X_train_list, X_test_list, y_train_full, y_test_full = dm.get_images_fromdisk(
            img_path=img_path, verbose=0)
        X_train_full = np.array(X_train_list)
        X_test_full = np.array(X_test_list)
        print(f"✓ Données MNIST chargées: {len(X_train_full)} images d'entraînement")
        using_real_mnist = True
    except Exception as e:
        print(f"⚠ Données MNIST non disponibles: {e}")
        print("  Génération de données synthétiques (28×28 aléatoires)...")

        np.random.seed(42)
        n_train = 1000
        n_test = 200
        X_train_full = np.random.rand(n_train, 28, 28)
        X_test_full = np.random.rand(n_test, 28, 28)
        y_train_full = np.zeros((n_train, 10))
        y_test_full = np.zeros((n_test, 10))

        for i in range(n_train):
            y_train_full[i, np.random.randint(0, 10)] = 1
        for i in range(n_test):
            y_test_full[i, np.random.randint(0, 10)] = 1

        print(f"✓ Données synthétiques générées: {len(X_train_full)} images d'entraînement")
        using_real_mnist = False

    y_train_indices = np.argmax(y_train_full, axis=1)
    y_test_indices = np.argmax(y_test_full, axis=1)

    train_mask = (y_train_indices == digit_pos) | (y_train_indices == digit_neg)
    X_train = flatten_images(X_train_full[train_mask])
    y_train = np.zeros(len(X_train))
    y_train[y_train_indices[train_mask] == digit_pos] = 1

    test_mask = (y_test_indices == digit_pos) | (y_test_indices == digit_neg)
    X_test = flatten_images(X_test_full[test_mask])
    y_test = np.zeros(len(X_test))
    y_test[y_test_indices[test_mask] == digit_pos] = 1

    X_train = X_train[:test_size]
    y_train = y_train[:test_size]
    X_test = X_test[:test_size]
    y_test = y_test[:test_size]

    print(f"✓ Train: {len(X_train)} images, Test: {len(X_test)} images")
    print(f"  Features: {X_train.shape[1]} (28×28 pixels aplatis)")

    results = {}

    print("\n" + "-"*70)
    print("1. PERCEPTRON CLASSIQUE (fonction Sign)")
    print("-"*70)

    y_train_sign = np.where(y_train == 1, 1, -1)
    y_test_sign = np.where(y_test == 1, 1, -1)

    start_time = time.time()
    print("Entraînement...")
    w_classique, b_classique, errors_classique = perceptron_classique_train(
        X_train, y_train_sign, learning_rate=0.001, epochs=50)
    train_time_classique = time.time() - start_time

    y_pred_classique_train = perceptron_classique_predict(X_train, w_classique, b_classique)
    y_pred_classique_test = perceptron_classique_predict(X_test, w_classique, b_classique)

    y_pred_classique_train = np.where(y_pred_classique_train == 1, 1, 0)
    y_pred_classique_test = np.where(y_pred_classique_test == 1, 1, 0)

    acc_train_classique = accuracy_score(y_train, y_pred_classique_train)
    acc_test_classique = accuracy_score(y_test, y_pred_classique_test)

    results['Classique'] = {
        'train_acc': acc_train_classique,
        'test_acc': acc_test_classique,
        'time': train_time_classique,
        'errors': errors_classique
    }

    print(f"Accuracy Train: {acc_train_classique:.4f}")
    print(f"Accuracy Test:  {acc_test_classique:.4f}")
    print(f"Temps d'entraînement: {train_time_classique:.2f}s")

    print("\n" + "-"*70)
    print("2. PERCEPTRON MSE (Mean Squared Error)")
    print("-"*70)

    start_time = time.time()
    print("Entraînement...")
    w_mse, b_mse, losses_mse = perceptron_mse_train(
        X_train, y_train, learning_rate=0.001, epochs=100)
    train_time_mse = time.time() - start_time

    y_pred_mse_train = perceptron_mse_predict(X_train, w_mse, b_mse)
    y_pred_mse_test = perceptron_mse_predict(X_test, w_mse, b_mse)

    y_pred_mse_train = np.where(y_pred_mse_train >= 0.5, 1, 0)
    y_pred_mse_test = np.where(y_pred_mse_test >= 0.5, 1, 0)

    acc_train_mse = accuracy_score(y_train, y_pred_mse_train)
    acc_test_mse = accuracy_score(y_test, y_pred_mse_test)

    results['MSE'] = {
        'train_acc': acc_train_mse,
        'test_acc': acc_test_mse,
        'time': train_time_mse,
        'losses': losses_mse
    }

    print(f"Accuracy Train: {acc_train_mse:.4f}")
    print(f"Accuracy Test:  {acc_test_mse:.4f}")
    print(f"Temps d'entraînement: {train_time_mse:.2f}s")

    print("\n" + "-"*70)
    print("3. PERCEPTRON LOGLOSS (Log Loss)")
    print("-"*70)

    start_time = time.time()
    print("Entraînement...")
    w_logloss, b_logloss, losses_logloss = perceptron_logloss_train(
        X_train, y_train, learning_rate=0.001, epochs=100)
    train_time_logloss = time.time() - start_time

    y_pred_logloss_train = perceptron_logloss_predict(X_train, w_logloss, b_logloss)
    y_pred_logloss_test = perceptron_logloss_predict(X_test, w_logloss, b_logloss)

    y_pred_logloss_train = np.where(y_pred_logloss_train >= 0.5, 1, 0)
    y_pred_logloss_test = np.where(y_pred_logloss_test >= 0.5, 1, 0)

    acc_train_logloss = accuracy_score(y_train, y_pred_logloss_train)
    acc_test_logloss = accuracy_score(y_test, y_pred_logloss_test)

    results['Logloss'] = {
        'train_acc': acc_train_logloss,
        'test_acc': acc_test_logloss,
        'time': train_time_logloss,
        'losses': losses_logloss
    }

    print(f"Accuracy Train: {acc_train_logloss:.4f}")
    print(f"Accuracy Test:  {acc_test_logloss:.4f}")
    print(f"Temps d'entraînement: {train_time_logloss:.2f}s")

    print("\n" + "="*70)
    print("RÉSUMÉ COMPARATIF")
    print("="*70)

    print(f"\n{'Méthode':<15} {'Train Acc':<12} {'Test Acc':<12} {'Temps (s)':<12}")
    print("-"*70)
    for method in ['Classique', 'MSE', 'Logloss']:
        r = results[method]
        print(f"{method:<15} {r['train_acc']:<12.4f} {r['test_acc']:<12.4f} {r['time']:<12.4f}")

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    methods = ['Classique', 'MSE', 'Logloss']
    train_accs = [results[m]['train_acc'] for m in methods]
    test_accs = [results[m]['test_acc'] for m in methods]

    x = np.arange(len(methods))
    width = 0.35
    axes[0, 0].bar(x - width/2, train_accs, width, label='Train', color='steelblue')
    axes[0, 0].bar(x + width/2, test_accs, width, label='Test', color='coral')
    axes[0, 0].set_ylabel('Accuracy')
    axes[0, 0].set_title('Comparaison des Accuracies')
    axes[0, 0].set_xticks(x)
    axes[0, 0].set_xticklabels(methods)
    axes[0, 0].legend()
    axes[0, 0].grid(axis='y', alpha=0.3)

    times = [results[m]['time'] for m in methods]
    axes[0, 1].bar(methods, times, color=['steelblue', 'green', 'orange'])
    axes[0, 1].set_ylabel('Temps (secondes)')
    axes[0, 1].set_title('Temps d\'entraînement')
    axes[0, 1].grid(axis='y', alpha=0.3)

    axes[1, 0].plot(results['Classique']['errors'], marker='o', label='Erreurs par époque')
    axes[1, 0].set_xlabel('Époque')
    axes[1, 0].set_ylabel('Nombre d\'erreurs')
    axes[1, 0].set_title('Perceptron Classique - Convergence')
    axes[1, 0].legend()
    axes[1, 0].grid(alpha=0.3)

    axes[1, 1].plot(results['MSE']['losses'], label='MSE', marker='+')
    axes[1, 1].plot(results['Logloss']['losses'], label='Logloss', marker='x')
    axes[1, 1].set_xlabel('Époque')
    axes[1, 1].set_ylabel('Perte')
    axes[1, 1].set_title('Convergence des Pertes')
    axes[1, 1].legend()
    axes[1, 1].grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig('comparison_mnist.png', dpi=150)
    print("\n✓ Graphique sauvegardé: comparison_mnist.png")

    print("\n" + "="*70)
    print("MATRICES DE CONFUSION")
    print("="*70)

    with open('matrices_confusion.txt', 'w') as f:
        f.write("MATRICES DE CONFUSION - Comparaison des 3 méthodes sur MNIST (1 vs 0)\n")
        f.write("="*70 + "\n\n")

    cm_classique = confusion_matrix(y_test, y_pred_classique_test)
    print("\n1. PERCEPTRON CLASSIQUE:")
    print("Matrice de confusion:")
    print(cm_classique)
    print(f"  Vrais Positifs (TP): {cm_classique[1,1]}")
    print(f"  Faux Positifs (FP): {cm_classique[0,1]}")
    print(f"  Faux Négatifs (FN): {cm_classique[1,0]}")
    print(f"  Vrais Négatifs (TN): {cm_classique[0,0]}")

    with open('matrices_confusion.txt', 'a') as f:
        f.write("1. PERCEPTRON CLASSIQUE:\n")
        f.write(f"{cm_classique}\n")
        f.write(f"TP: {cm_classique[1,1]}, FP: {cm_classique[0,1]}, FN: {cm_classique[1,0]}, TN: {cm_classique[0,0]}\n\n")

    cm_mse = confusion_matrix(y_test, y_pred_mse_test)
    print("\n2. PERCEPTRON MSE:")
    print("Matrice de confusion:")
    print(cm_mse)
    print(f"  Vrais Positifs (TP): {cm_mse[1,1]}")
    print(f"  Faux Positifs (FP): {cm_mse[0,1]}")
    print(f"  Faux Négatifs (FN): {cm_mse[1,0]}")
    print(f"  Vrais Négatifs (TN): {cm_mse[0,0]}")

    with open('matrices_confusion.txt', 'a') as f:
        f.write("2. PERCEPTRON MSE:\n")
        f.write(f"{cm_mse}\n")
        f.write(f"TP: {cm_mse[1,1]}, FP: {cm_mse[0,1]}, FN: {cm_mse[1,0]}, TN: {cm_mse[0,0]}\n\n")

    cm_logloss = confusion_matrix(y_test, y_pred_logloss_test)
    print("\n3. PERCEPTRON LOGLOSS:")
    print("Matrice de confusion:")
    print(cm_logloss)
    print(f"  Vrais Positifs (TP): {cm_logloss[1,1]}")
    print(f"  Faux Positifs (FP): {cm_logloss[0,1]}")
    print(f"  Faux Négatifs (FN): {cm_logloss[1,0]}")
    print(f"  Vrais Négatifs (TN): {cm_logloss[0,0]}")

    with open('matrices_confusion.txt', 'a') as f:
        f.write("3. PERCEPTRON LOGLOSS:\n")
        f.write(f"{cm_logloss}\n")
        f.write(f"TP: {cm_logloss[1,1]}, FP: {cm_logloss[0,1]}, FN: {cm_logloss[1,0]}, TN: {cm_logloss[0,0]}\n\n")

    print("\n" + "="*70)
    print("VISUALISATION DES MATRICES DE CONFUSION")
    print("="*70)

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    methods = ['Classique', 'MSE', 'Logloss']
    cms = [cm_classique, cm_mse, cm_logloss]
    colors = ['Blues', 'Greens', 'Oranges']

    for i, (method, cm, color) in enumerate(zip(methods, cms, colors)):
        ax = axes[i]

        im = ax.imshow(cm, interpolation='nearest', cmap=color)
        ax.figure.colorbar(im, ax=ax)

        thresh = cm.max() / 2.
        for j in range(cm.shape[0]):
            for k in range(cm.shape[1]):
                ax.text(k, j, format(cm[j, k], 'd'),
                       ha="center", va="center",
                       color="white" if cm[j, k] > thresh else "black",
                       fontsize=16, fontweight='bold')

        ax.set_xticks([0, 1])
        ax.set_yticks([0, 1])
        ax.set_xticklabels(['Prédit 0', 'Prédit 1'])
        ax.set_yticklabels(['Réel 0', 'Réel 1'])
        ax.set_xlabel('Prédiction')
        ax.set_ylabel('Réalité')
        ax.set_title(f'{method}\nAccuracy: {results[method]["test_acc"]:.1%}')

        tp, fp, fn, tn = cm[1,1], cm[0,1], cm[1,0], cm[0,0]
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0

        ax.text(0.02, 0.98, f'Précision: {precision:.3f}\nRappel: {recall:.3f}',
               transform=ax.transAxes, fontsize=10, verticalalignment='top',
               bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    plt.tight_layout()
    plt.savefig('matrices_confusion.png', dpi=150, bbox_inches='tight')
    print("✓ Graphique des matrices de confusion sauvegardé: matrices_confusion.png")

    print("\n" + "="*70)
    print("TABLEAU RÉCAPITULATIF COMPLET")
    print("="*70)

    header = f"{'Méthode':<15} {'Accuracy':<10} {'Précision':<12} {'Rappel':<10} {'F1-Score':<12} {'Erreurs':<10}"
    print(header)
    print("-" * len(header))

    for method in ['Classique', 'MSE', 'Logloss']:
        r = results[method]
        if method == 'Classique':
            cm = cm_classique
        elif method == 'MSE':
            cm = cm_mse
        else:
            cm = cm_logloss

        tp, fp, fn, tn = cm[1,1], cm[0,1], cm[1,0], cm[0,0]
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        erreurs = fp + fn

        print(f"{method:<15} {r['test_acc']:<10.4f} {precision:<12.4f} {recall:<10.4f} {f1:<12.4f} {erreurs:<10d}")

    with open('matrices_confusion.txt', 'a') as f:
        f.write("\nTABLEAU RÉCAPITULATIF\n")
        f.write("="*70 + "\n\n")
        f.write(header + "\n")
        f.write("-" * len(header) + "\n")

        for method in ['Classique', 'MSE', 'Logloss']:
            r = results[method]
            if method == 'Classique':
                cm = cm_classique
            elif method == 'MSE':
                cm = cm_mse
            else:
                cm = cm_logloss

            tp, fp, fn, tn = cm[1,1], cm[0,1], cm[1,0], cm[0,0]
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
            erreurs = fp + fn

            f.write(f"{method:<15} {r['test_acc']:<10.4f} {precision:<12.4f} {recall:<10.4f} {f1:<12.4f} {erreurs:<10d}\n")

        f.write("\nLégende:\n")
        f.write("- Accuracy: Pourcentage de prédictions correctes\n")
        f.write("- Précision: TP/(TP+FP) - Quand on prédit 1, a-t-on raison ?\n")
        f.write("- Rappel: TP/(TP+FN) - Trouve-t-on tous les 1 ?\n")
        f.write("- F1-Score: Moyenne harmonique précision/rappel\n")
        f.write("- Erreurs: Nombre total d'erreurs (FP + FN)\n")

    print("\n✓ Matrices de confusion sauvegardées: matrices_confusion.txt")

    print("\n" + "="*70)
    print("Comparaison terminée!")
    print("="*70)
