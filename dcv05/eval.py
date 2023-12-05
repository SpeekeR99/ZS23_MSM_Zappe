import numpy as np
from sklearn.model_selection import KFold
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import roc_curve


def confusion_matrix(test_y, predicted_y, classes_dict):
    """
    Vypocet matice zamen
    :param test_y: Testovaci tridy
    :param predicted_y: Predikovane tridy
    :param classes_dict: Slovnik trid (trida -> cislo)
    :return: Matice zamen
    """
    conf_mat = np.zeros((len(classes_dict), len(classes_dict)))

    for i in range(len(predicted_y)):
        conf_mat[test_y[i]][predicted_y[i]] += 1

    return conf_mat


def cross_validation(samples, classes):
    """
    Klasifikace pomoci LDA s k-fold a leave one out metodou
    :param samples: Vzorky (po PCA)
    :param classes: Tridy
    """
    # Cross validate with k-fold and leave one out methods

    # K-fold
    kfolds = [2, 5, 10]
    for k in kfolds:
        kf = KFold(n_splits=k)
        acc = []
        for train_index, test_index in kf.split(samples):
            train_x, test_x = samples[train_index], samples[test_index]
            train_y, test_y = classes[train_index], classes[test_index]

            model = LinearDiscriminantAnalysis()
            model.fit(train_x, train_y)

            predicted_y = model.predict(test_x)
            acc.append(np.sum(predicted_y == test_y) / len(test_y))
        print(f"Accuracy of classification with k-fold (k = {k}): {np.mean(acc) * 100} %")

    # Leave one out
    loo = KFold(n_splits=len(samples))
    acc = []
    for train_index, test_index in loo.split(samples):
        train_x, test_x = samples[train_index], samples[test_index]
        train_y, test_y = classes[train_index], classes[test_index]

        model = LinearDiscriminantAnalysis()
        model.fit(train_x, train_y)

        predicted_y = model.predict(test_x)
        acc.append(np.sum(predicted_y == test_y) / len(test_y))
    print(f"Accuracy of classification with leave one out: {np.mean(acc) * 100} %")


def roc(test_x, test_y, classes_dict):
    """
    Vypocet ROC krivky
    :param test_x: Testovaci vzorky
    :param test_y: Testovaci tridy
    :param classes_dict: Slovnik trid (trida -> cislo)
    :return: ROC krivky
    """
    # ROC krivky
    rocs = []
    for i in range(len(classes_dict)):
        model = LinearDiscriminantAnalysis()

        # Tady musim udelat prakticky klasifikaci one vs all, protoze ROC je pouze pro binarni klasifikaci
        # => bude 6 ROC krivek pro 6 trid sutru
        y = np.zeros(len(test_y))  # Transformace trid na binarni vektor
        for j in range(len(test_y)):
            if test_y[j] == i:
                y[j] = 1
            else:
                y[j] = 0

        if (np.sum(y) == 0):  # Pokud trida v testovacich datech neni, je nutne ji preskocit
            rocs.append(([], []))
            continue

        model.fit(test_x, y)
        proba_y = model.predict_proba(test_x)

        fpr, trp, _ = roc_curve(y, proba_y[:, 1])
        rocs.append((fpr, trp))

    return rocs
