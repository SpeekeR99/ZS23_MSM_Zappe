import sys
import numpy as np
import scipy.linalg as la
from scipy.stats import f
import matplotlib.pyplot as plt
from sklearn import datasets
import pandas as pd
import plotly.graph_objs as go

# Globalni promenna pro cislovani figur
fig_id = 1


def confidence_set(data, alpha=0.05, color='red', label='set', fig=None):
    """
    Funkce pro vypocet a vykresleni konfidencni mnoziny pro predana data a hladinu vyznamnosti alfa
    :param data: Data pro vypocet konfidencni mnoziny (ocekava se 2D matice)
    :param alpha: Hladina vyznamnosti
    :param color: Barva vykresleni konfidencni mnoziny
    :param label: Nazev pro legendu
    :param fig: Figura pro vykresleni
    """
    n, p = data.shape  # n - pocet vzorku, p - dimenze
    # Odhad stredni hodnoty a kovariancni matice
    mean_hat = np.mean(data, axis=0)
    sigma_hat = (np.dot(data.T, data) / n - np.dot(mean_hat.reshape(p, 1), mean_hat.reshape(1, p))) * n / (n - 1)

    # Odmocnina z kovariancni matice a odhadovane kovariancni matice
    # S = la.sqrtm(sigma)  # Pozustatek z testovani na nahodnych datech
    S_hat = la.sqrtm(sigma_hat)

    # Matice pro transformaci do normalizovaneho prostoru
    Z = (np.linalg.inv(S_hat) @ (data - np.tile(mean_hat, (n, 1))).T).T

    # Funkcni hodnoty inverznich distribucnich funkci
    # F_teor = chi2.ppf(1 - alpha, df=p)  # Pozustatek z testovani na nahodnych datech
    F_esti = f.ppf(1 - alpha, p, n - p) * p * (n - 1) / (n - p)
    if p == 2:  # (2D)
        space = np.linspace(0, 2 * np.pi, 100)  # Linearni prostor pro vykresleni
    else:  # p == 3 (3D)
        space = np.meshgrid(np.linspace(0, np.pi, 100), np.linspace(0, 2 * np.pi, 100))  # Linearni prostor pro vykresleni

    # Body normalizovanych kruznic pro vykresleni
    # zz_teor = np.array([np.sqrt(F_teor) * np.cos(t), np.sqrt(F_teor) * np.sin(t)]).T  # Pozustatek z testovani na nahodnych datech
    if p == 2:  # (2D)
        zz_esti = np.array([np.sqrt(F_esti) * np.cos(space),
                            np.sqrt(F_esti) * np.sin(space)]).T
    else:  # p == 3 (3D)
        zz_esti = np.array([np.sqrt(F_esti) * np.sin(space[0]) * np.cos(space[1]),
                            np.sqrt(F_esti) * np.sin(space[0]) * np.sin(space[1]),
                            np.sqrt(F_esti) * np.cos(space[0])]).T
    # Body elips v puvodnim prostoru pro vykresleni
    # xx_teor = np.dot(zz_teor, S) + np.tile(mu, (len(zz_teor), 1))  # Pozustatek z testovani na nahodnych datech
    xx_esti = np.dot(zz_esti, S_hat) + np.tile(mean_hat, (len(zz_esti), 1))

    # Vypocet Mahalanobisovy vzdalenosti
    mah_dist_z = np.zeros((n, 1))
    for i in range(n):
        mah_dist_z[i, 0] = np.dot(Z[i, :], Z[i, :])  # Zde se muze pocitat klidne v normalizovanem prostoru
    # Detekce outlieru
    data_out = data[np.where(mah_dist_z > F_esti)[0], :]

    if p == 2:  # (2D)
        # Vykresleni elips(y)
        # plt.fill(zz_teor[:, 0], zz_teor[:, 1], 'y', alpha=0.05, edgecolor='y', linestyle='--', linewidth=1)  # Pozustatek z testovani na nahodnych datech
        # plt.fill(zz_esti[:, 0], zz_esti[:, 1], 'r', alpha=0.05, edgecolor='r', linestyle='--', linewidth=1)  # Pozustatek z testovani na nahodnych datech
        # plt.fill(xx_teor[:, 0], xx_teor[:, 1], 'g', alpha=0.2, edgecolor='g', linestyle='--', linewidth=1)  # Pozustatek z testovani na nahodnych datech
        plt.fill(xx_esti[:, 0], xx_esti[:, 1],
                 color=color, alpha=0.2, edgecolor=color, linestyle='--', linewidth=1, label=label)

        # Vykresleni dat a outlieru
        plt.plot(data[:, 0], data[:, 1], '+', color=color)
        plt.plot(data_out[:, 0], data_out[:, 1], 'o', color=color)
    else:  # p == 3 (3D)
        # Vykresleni elipsoid(y)
        temp = np.zeros((xx_esti.shape[0] ** 2, 3))  # Transformace do pozadovaneho formatu
        temp[:, 0] = xx_esti[:, :, 0].reshape(-1, order='F')
        temp[:, 1] = xx_esti[:, :, 1].reshape(-1, order='F')
        temp[:, 2] = xx_esti[:, :, 2].reshape(-1, order='F')
        xx_esti = temp
        df = pd.DataFrame(xx_esti, columns=['x', 'y', 'z'])  # Zalozeni dataframe
        fig.add_trace(go.Scatter3d(x=df['x'], y=df['y'], z=df['z'],
                                   mode='markers', marker=dict(size=1, color=color), opacity=0.2, name=label))

        # Vykresleni dat a outlieru
        fig.add_trace(go.Scatter3d(x=data[:, 0], y=data[:, 1], z=data[:, 2],
                                   mode='markers', marker=dict(size=3, color=color), name=label))
        fig.add_trace(go.Scatter3d(x=data_out[:, 0], y=data_out[:, 1], z=data_out[:, 2],
                                   mode='markers', marker=dict(size=8, color=color), name=label + ' outlier'))

        fig.update_layout(margin=dict(l=0, r=0, b=0, t=0))


def plot_2D_confidence_sets_for_iris(data, target, axes_labels, alpha=0.05, block=False):
    """
    Funkce pro vykresleni konfidencnich mnozin pro data Iris ve 2D
    :param data: 2D data z datasetu Iris (2 sloupce v matici)
    :param target: Trida dat (0, 1, 2) pro odliseni jednotlivych trid (clusteru)
    :param axes_labels: Popisky os
    :param alpha: Hladina vyznamnosti
    :param block: Blokovani vykresleni (True/False) (aby se vykreslila vsechna okna najednou)
    """
    # Rozdeleni dat do jednotlivych trid (clusteru)
    data1 = data[np.where(target == 0)[0], :]
    data2 = data[np.where(target == 1)[0], :]
    data3 = data[np.where(target == 2)[0], :]

    # Zalozeni noveho okna pro vykresleni
    global fig_id
    plt.figure(fig_id)
    fig_id += 1  # Zvyseni ID figury pro dalsi vykresleni

    # Vykresleni konfidencnich mnozin pro jednotlive tridy
    confidence_set(data1, alpha=alpha, color='red', label='setosa')
    confidence_set(data2, alpha=alpha, color='green', label='versicolor')
    confidence_set(data3, alpha=alpha, color='blue', label='virginica')

    # Vylepseni vykresleni :)
    plt.title('Confidence sets for {} and {}'.format(axes_labels[0], axes_labels[1]))
    plt.xlabel(axes_labels[0])
    plt.ylabel(axes_labels[1])
    plt.grid()
    plt.axis('equal')
    plt.legend()
    plt.show(block=block)


def plot_3D_confidence_sets_for_iris(data, target, alpha=0.05):
    """
    Funkce pro vykresleni konfidencnich mnozin pro data Iris ve 3D
    :param data: 3D data z datasetu Iris (3 sloupce v matici)
    :param target: Trida dat (0, 1, 2) pro odliseni jednotlivych trid (clusteru)
    :param alpha: Hladina vyznamnosti
    """
    # Rozdeleni dat do jednotlivych trid (clusteru)
    data1 = data[np.where(target == 0)[0], :]
    data2 = data[np.where(target == 1)[0], :]
    data3 = data[np.where(target == 2)[0], :]

    # Zalozeni nove figury pro vykresleni
    fig = go.Figure()

    # Vykresleni konfidencnich mnozin pro jednotlive tridy
    confidence_set(data1, alpha=alpha, color='red', label='setosa', fig=fig)
    confidence_set(data2, alpha=alpha, color='green', label='versicolor', fig=fig)
    confidence_set(data3, alpha=alpha, color='blue', label='virginica', fig=fig)

    # Vykresleni
    fig.show()


if __name__ == '__main__':
    # Nacteni dat
    iris = datasets.load_iris()
    data = iris.data
    target = iris.target
    axes_labels = np.array(iris.feature_names)

    # Nastaveni hladiny vyznamnosti
    alpha = 0.05  # Defaultni hodnota
    if len(sys.argv) > 1:
        alpha = float(sys.argv[1])  # Nacteni z argumentu pri spusteni
    print('alpha = {}'.format(alpha))

    # Vykresleni konfidencnich mnozin pro jednotlive dvojice atributu (sloupcu)
    [plot_2D_confidence_sets_for_iris(
        data[:, [i, j]], target, axes_labels[[i, j]], alpha=alpha, block=(i == 2 and j == 3)
    ) for i in range(4) for j in range(i + 1, 4)]
    # Vykresleni konfidencnich mnozin pro jednotlive trojice atributu (sloupcu)
    [plot_3D_confidence_sets_for_iris(
        data[:, [i, j, k]], target, alpha=alpha
    ) for i in range(4) for j in range(i + 1, 4) for k in range(j + 1, 4)]
