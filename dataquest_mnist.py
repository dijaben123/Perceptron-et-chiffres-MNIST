"""
Fichier : dataquest_mnist.py
"""
import numpy as np
import random
import cv2
import os

import matplotlib.pyplot as plt

import sklearn as skl
from sklearn.utils import shuffle
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split


#####################################################
def list_image_paths(root_dir, valid_exts=(".jpg", ".jpeg", ".png", ".bmp", ".gif")):
    """
    version  by deepseek .. pour ne pas avoir à installer imutils et utiliser son submodules "paths"
    #from imutils import paths # https://github.com/PyImageSearch/imutils

    Génère les chemins des fichiers d'images contenues dans un dossier et ses sous-dossiers (recursivement)
    """
    for (root, dirs, files) in os.walk(root_dir):
        for filename in files:
            # Vérifier l'extension du fichier
            ext = os.path.splitext(filename)[1].lower()
            if ext.endswith(valid_exts):
                # Générer le chemin complet
                yield os.path.join(root, filename)


def list_image_paths_sorted(root_dir):
    """
    Version triée équivalente à imutils
    Retourne les chemins triés comme imutils.paths
    """
    image_paths = list(list_image_paths(root_dir))
    return sorted(image_paths)


#####################################################
def load_from_file_and_process_images(paths, verbose=None):
    '''
    Load and preprocess image of the base

    Keyword arguments:
    paths --  path of repository of images =>  expects images for each class in seperate dir,
    e.g all digits in 0 class should be in the directory named 0
    verbose -- whant to see a trace during loading ?

    Returns :
    data -- a list of images greyscaled/normalized (0,1)/flattened
               pixels are floating numbers
    labels -- a list of labels/string representing the digit associated
    '''
    data = list()  # a list of images
    labels = list()  # a list of labels

    # loop over the input images
    for (i, imgpath) in enumerate(paths):
        # image/pixel load  from file
        im_gray = cv2.imread(imgpath, cv2.IMREAD_GRAYSCALE)
        # image processings
        image = np.array(im_gray)
        image = image / 255  # scale the image to [0, 1]
        # image=np.expand_dims(image, axis=0) # will move dimension  to (1,28,28)
        # image = image.flatten() # image is now a vector of 28*28 = 784
        data.append(image)  # and append to list

        # and extract the class labels => label deduced from file path (a string) !
        label = imgpath.split(os.path.sep)[-2]
        labels.append(label)

        # show an update every `verbose` images
        if verbose is not None and verbose > 0 and i > 0 and (i + 1) % verbose == 0:
            print("[INFO] processed {}/{}".format(i + 1, len(paths)))

    # return a tuple of the data and labels
    return data, labels


# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

def get_images_fromdisk(img_path='../../MNIST/trainingSet/trainingSet/', verbose=0):
    """
    img path to your mnist data folder :  => 42000 fichiers dans 10 répertoires : 0, 1, 2, ...
    """
    if verbose != 0:
        print(f"\nNous allons travailler avec la base située là : {img_path}")

    # On récupere les paths des images
    image_paths = list_image_paths_sorted(img_path)
    if verbose != 0:
        print("\n5 premiers fichiers :\n")  # Les 5 premiers noms de fichiers
        print("\n".join(image_paths[:5], ))

    # fn_list est la liste des noms terminaux (sans le path) de tous les fichiers contenus dans notre base d'images
    fn_list = [os.path.basename(i) for i in image_paths]

    # Pour chacun de ces fichiers, on veut obtenir l'image (liste des pixels) et le label (le chiffre) auquel elle correspond.
    # A la fin on obtient deux listes :
    # a) il : la liste des images/pixels
    # b) et ll la liste des labels
    if verbose != 0:
        print("\nMNIST base loading and preprocessing ...\n")
    il, ll = load_from_file_and_process_images(image_paths, verbose=10000)
    if verbose != 0:
        print("\nL'image 0 : \n\t Nom : {} \n\t Label  : {} \n\t Pixels : \n {}".format(fn_list[:1], ll[:1], il[:1]))
        print("Image Shape => ", il[0].shape)

    # On a besoin d'une représentation "hotone" ... des labels voir le doc
    # Comme ceux sont des chiffres .. on peut "binariser"  avec une méthode de sklearn.preprocessing
    lb = skl.preprocessing.LabelBinarizer()  # Pour binariser the labels
    llho = lb.fit_transform(ll)
    if verbose != 0:
        print("\nLes 5 premiers labels  :\n", ll[:5])
        print("sous une forme \"hot one\"  :\n", llho[:5])

    # from sklearn => split data into training and test set
    X_train, X_test, y_train, y_test = train_test_split(il, llho,
                                                        test_size=0.1,
                                                        random_state=19)

    # pour voir et etre certain des dimensions (cas MNIST) !
    if verbose != 0:
        print("Size X_train ", len(X_train), " of ", X_train[0].shape)
        print("Size y_train ", len(y_train), " of ", y_train[0].shape)
        for i in X_train:
            if i.shape != (28, 28):
                print("Alert !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
        for i in y_train:
            if i.shape != (10,):
                print("Alert !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")

    if verbose != 0:
        print("Size X_test ", len(X_test), " of ", X_test[0].shape)
        print("Size y_test ", len(y_test), " of ", y_test[0].shape)
        for i in X_test:
            if i.shape != (28, 28):
                print("Alert !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
        for i in y_test:
            if i.shape != (10,):
                print("Alert !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")

    return X_train, X_test, y_train, y_test


# ===============================================
if __name__ == "__main__":
    get_images_fromdisk(verbose=1)