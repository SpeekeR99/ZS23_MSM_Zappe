import numpy as np
import matplotlib.pyplot as plt

colors = ['r', 'g', 'b', 'y', 'm', 'c']  # Barvicky pro konzistentni vykreslovani (matplotlib)
my_cmap = plt.cm.colors.ListedColormap(colors)  # Barvickova mapa


def plot_pareto(pca):
    """
    Pareto graf
    :param pca: PCA objekt z sklearn
    """
    fig = plt.figure()
    ax = fig.add_subplot(111)

    # Vykresleni bar grafu
    ax.bar(np.arange(len(pca.explained_variance_ratio_)), pca.explained_variance_ratio_, color='green')

    ax.set_xticks(np.arange(len(pca.explained_variance_ratio_)))
    ax.set_xticklabels(np.arange(1, len(pca.explained_variance_ratio_) + 1))
    ax.set_yticks(np.arange(0, 1.1, 0.1))
    ax.set_ylim(0, 1)

    ax.set_xlabel('Principal component')
    ax.set_ylabel('Proportion of variance explained')
    ax.set_title('Pareto graph')

    # Vykresleni kumulativni krivky
    ax.plot(np.cumsum(pca.explained_variance_ratio_), color='red')

    plt.grid()
    plt.show(block=False)


def plot_data_in_2D_space(y, classes, classes_dict):
    """
    Vykresleni dat v 2D prostoru
    :param y: Komponenty PCA
    :param classes: Tridy
    :param classes_dict: Slovnik trid
    """
    plt.figure()

    # Vykresleni scatter grafu
    scatter = plt.scatter(y[:, 0], y[:, 1], c=classes, cmap=my_cmap)

    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.title("PCA")

    plt.legend(handles=scatter.legend_elements()[0], labels=classes_dict.keys())
    plt.grid()
    plt.show(block=False)


def plot_biplot(pca, y, data, classes, classes_dict, number_of_unimportant_columns):
    """
    Biplot
    :param pca: PCA objekt z sklearn
    :param samples_sc: Transformovane vzorky
    :param data: Data
    :param classes: Tridy
    :param classes_dict: Slovnik trid
    :param number_of_unimportant_columns: Pocet nevypovednich sloupcu (pocet sloupcu, co nemaji data)
    """
    fig = plt.figure()
    ax = fig.add_subplot(111)

    # Vykresleni vektoru (sipek)
    loadings = pca.components_
    features = list(data.keys())[number_of_unimportant_columns:]

    for i, feature in enumerate(features):
        # Vykresleni sipky
        ax.arrow(0, 0, loadings[0, i], loadings[1, i], head_width=0.02, head_length=0.05, fc='k', ec='k')
        # Vykresleni popisku sipky
        ax.text(loadings[0, i] * 1.15, loadings[1, i] * 1.15, feature, color='k', ha='center', va='center', fontsize=10)

    # Vykresleni scatter grafu
    PC1 = y[:, 0]
    PC2 = y[:, 1]
    scalePC1 = 1.0 / (PC1.max() - PC1.min())
    scalePC2 = 1.0 / (PC2.max() - PC2.min())
    scatter = plt.scatter(PC1 * scalePC1, PC2 * scalePC2, c=classes, cmap=my_cmap)

    ax.set_xlabel('PC1')
    ax.set_ylabel('PC2')
    ax.set_title('Biplot')

    plt.legend(handles=scatter.legend_elements()[0], labels=classes_dict.keys())
    plt.grid()
    plt.show(block=False)


def plot_conf_mat(conf_mat, labels):
    """
    Vykresleni confusion matrix
    :param conf_mat: Confusion matrix
    :param labels: Popisky
    """
    fig = plt.figure()
    ax = fig.add_subplot(111)

    # Vykresleni matice
    mat = ax.matshow(conf_mat, cmap="Greens")
    plt.colorbar(mat)

    # Vykresleni popisku
    for (i, j), z in np.ndenumerate(conf_mat):
        ax.text(j, i, '{:0.1f}'.format(z), ha='center', va='center', fontsize=10)

    ax.set_xticks(np.arange(len(labels)))
    ax.set_yticks(np.arange(len(labels)))
    ax.set_xticklabels(labels)
    ax.set_yticklabels(labels)

    ax.set_xlabel('Predicted label')
    ax.set_ylabel('True label')
    ax.set_title('Confusion matrix')

    plt.show(block=False)


def plot_roc_curves(rocs, classes_dict):
    """
    Vykresleni ROC krivek
    :param rocs: ROC krivky
    :param classes_dict: Slovnik trid
    """
    plt.figure()
    for i, roc_ in enumerate(rocs):
        fpr, tpr = roc_[0], roc_[1]
        plt.plot(fpr, tpr, color=colors[i])

    plt.xlabel("False positive rate")
    plt.ylabel("True positive rate")
    plt.title("ROC curves")

    plt.legend([f"ROC curve for class {i}" for i in classes_dict.keys()])
    plt.grid()
    plt.show()
