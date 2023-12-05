import numpy as np
import scipy.linalg as la
import scipy.stats as stats
import matplotlib.pyplot as plt
import plotly.graph_objs as go
import pandas as pd
import sys

circle_viable = (sys.version_info[0] == 3 and sys.version_info[1] >= 8)  # PyCirclize funguje pouze pro Python 3.8 a vyssi
if circle_viable:
    from pycirclize import Circos

# # # ------------------------ GLOBALS --------------------------------------------------------------------------- # # #


fig_id = 1  # Pouzito pro vykreslovani vice elips do stejne figury
colors = ['r', 'g', 'b', 'y', 'm', 'c']  # Barvicky pro konzistentni vykreslovani (matplotlib)
colors_plotly = ['red', 'green', 'blue', 'yellow', 'magenta', 'cyan']  # Barvicky pro konzistentni vykreslovani (plotly)
colors_alpha_50 = [[1, 0, 0, 0.5],  # Barvicky pro konzistentni vykreslovani (matplotlib) s pruhlednosti 50 % (boxploty)
                   [0, 1, 0, 0.5],
                   [0, 0, 1, 0.5],
                   [1, 1, 0, 0.5],
                   [1, 0, 1, 0.5],
                   [0, 1, 1, 0.5]]


# # # ------------------------ SAMPLES --------------------------------------------------------------------------- # # #


def plot_samples(samples, combinations, my_choice, classes_dict_inv, codes=False, dim=1):
    """
    Vykresli vzorky v zavislosti na dimenzi (1D, 2D)
    :param samples: Data a jejich tridy
    :param my_choice: Nazvy vybranych sloupcu
    :param classes_dict_inv: Slovnik trid (prevod cisel na nazvy)
    :param codes: Priznak, jestli do figury vypsat i kody vzorku (True / False)
    :param dim: Dimenze kresleni (1D, 2D)
    :return:
    """
    if dim == 1:  # 1D
        plot_samples_1D(samples, combinations, my_choice, classes_dict_inv, codes)
    elif dim == 2:  # 2D
        plot_samples_2D(samples, combinations, my_choice, classes_dict_inv, codes)


def plot_samples_1D(samples, combinations, my_choice, classes_dict_inv, codes=False):
    """
    Vykresli vzorky v 1D
    :param samples: Data a jejich tridy
    :param combinations: Kombinace sloupcu v dane dimenzi
    :param my_choice: Nazvy vybranych sloupcu
    :param classes_dict_inv: Slovnik trid (prevod cisel na nazvy)
    :param codes: Priznak, jestli do figury vypsat i kody vzorku (True / False)
    :return:
    """
    fig = plt.figure(figsize=(10, 8))
    fig.suptitle("1D data")

    for i in range(len(combinations)):
        ax = fig.add_subplot(101 + len(combinations) * 10 + i)  # Tri figury vedle sebe

        for stone in classes_dict_inv.keys():  # Pro kazdou tridu
            data = samples["Data"][samples["Classes"] == stone][:, i]
            ax.scatter(np.ones(len(data)) * stone, data, color=colors[stone], marker="x", s=50)

        if codes:  # Pokud se maji vypsat kody
            for j, txt in enumerate(samples["Codes"]):
                ax.annotate(txt, (samples["Classes"][j], samples["Data"][j, i]), fontsize=8)

        ax.xaxis.set_ticks(np.arange(0, len(classes_dict_inv), 1))
        ax.xaxis.set_ticklabels([classes_dict_inv[c] for c in classes_dict_inv], rotation=45)
        ax.title.set_text(my_choice[i])
        ax.grid()

    plt.show(block=False)


def plot_samples_2D(samples, combinations, my_choice, classes_dict_inv, codes=False):
    """
    Vykresli vzorky v 2D
    :param samples: Data a jejich tridy
    :param combinations: Kombinace sloupcu v dane dimenzi
    :param my_choice: Nazvy vybranych sloupcu
    :param classes_dict_inv: Slovnik trid (prevod cisel na nazvy)
    :param codes: Priznak, jestli do figury vypsat i kody vzorku (True / False)
    :return:
    """
    fig = plt.figure(figsize=(10, 8))
    fig.suptitle("2D data")

    for i in range(len(combinations)):
        for j in range(i + 1, len(combinations)):
            ax = fig.add_subplot(131 + j - 1 + i)  # Tri figury vedle sebe

            for stone in classes_dict_inv.keys():  # Pro kazdou tridu
                data = samples["Data"][samples["Classes"] == stone]
                ax.scatter(data[:, i], data[:, j], color=colors[stone], marker="x", label=classes_dict_inv[stone], s=50)

            if codes:  # Pokud se maji vypsat kody
                for k, txt in enumerate(samples["Codes"]):
                    ax.annotate(txt, (samples["Data"][k, i], samples["Data"][k, j]), fontsize=8)

            ax.title.set_text(my_choice[i] + " + " + my_choice[j])
            ax.set_xlabel(my_choice[i])
            ax.set_ylabel(my_choice[j])
            ax.legend()
            ax.grid()

    plt.show(block=codes)  # Blokujici, kdyz se maji vypsat kody (vychazi to z posloupnost prikazu v mainu)


def plot_boxplots(samples, my_choice, classes_dict_inv, codes=False):
    """
    Vykresli boxploty 1D dat
    :param samples: Data a jejich tridy
    :param my_choice: Nazvy vybranych sloupcu
    :param classes_dict_inv: Slovnik trid (prevod cisel na nazvy)
    :param codes: Priznak, jestli do figury vypsat i kody vzorku (True / False)
    :return:
    """
    fig = plt.figure(figsize=(10, 8))
    fig.suptitle("Boxplots of 1D data")

    for i in range(len(my_choice)):
        ax = fig.add_subplot(101 + len(my_choice) * 10 + i)  # Tri figury vedle sebe

        for stone in classes_dict_inv.keys():  # Pro kazdou tridu
            data = samples["Data"][samples["Classes"] == stone][:, i]
            ax.boxplot(data, positions=[stone], widths=0.5, patch_artist=True,
                       boxprops=dict(facecolor=colors_alpha_50[stone]))

        if codes:  # Pokud se maji vypsat kody
            for j, txt in enumerate(samples["Codes"]):
                ax.annotate(txt, (samples["Classes"][j], samples["Data"][j, i]), fontsize=8)

        ax.xaxis.set_ticks(np.arange(0, len(classes_dict_inv), 1))
        ax.xaxis.set_ticklabels([classes_dict_inv[c] for c in classes_dict_inv], rotation=45)
        ax.title.set_text(my_choice[i])
        ax.grid()

    plt.show(block=codes)  # Opet vychazi z posloupnosti prikazu v mainu


def plot_elipses(samples, combinations, my_choice, classes_dict_inv, codes=False, dim=2):
    """
    Vykresli elipsy 2D dat (prevzano od sebe z minuleho ukolu)
    :param samples: Data a jejich tridy
    :param combinations: Komibnace sloupcu v dane dimenzi
    :param my_choice: Nazvy vybranych sloupcu
    :param classes_dict_inv: Slovnik trid (prevod cisel na nazvy)
    :param codes: Priznak, jestli do figury vypsat i kody vzorku (True / False)
    :param dim: Dimenze dat (2 / 3)
    :return:
    """
    max_iter = samples["Data"].shape[0]  # Pocet iteraci

    for i, data in enumerate(samples["Data"]):  # Pro kazdy sloupec v datech (kombinace)
        global fig_id
        plt.figure(fig_id)
        fig_id += 1

        if dim != 2:  # Pokud je dimenze 3, tak se vykresli 3D graf pres plotly
            fig = go.Figure()

        for class_ in classes_dict_inv.keys():  # Pro kazdou tridu
            plot_data = data[samples["Classes"] == class_]  # Vyberu data dane tridy
            code_column = samples["Codes"][samples["Classes"] == class_]  # Vyberu kody dane tridy
            if dim == 2:  # Pokud je dimenze 2, tak se vykresli 2D graf pres matplotlib
                confidence_set(plot_data, alpha=0.05, color=colors[class_], label=classes_dict_inv[class_],
                               code_column=code_column, codes=codes)
            else:  # Pokud je dimenze 3, tak se vykresli 3D graf pres plotly
                confidence_set(plot_data, alpha=0.05, color=colors_plotly[class_], label=classes_dict_inv[class_],
                               code_column=code_column, codes=codes, fig=fig)

        if dim == 2:  # Pokud je dimenze 2, tak se vykresli 2D graf pres matplotlib
            plt.title(
                'Confidence sets for {} and {}'.format(my_choice[combinations[i][0]], my_choice[combinations[i][1]]))
            plt.xlabel(my_choice[combinations[i][0]])
            plt.ylabel(my_choice[combinations[i][1]])
            plt.grid()
            plt.axis('equal')
            plt.legend()
            plt.show(block=(codes and i == max_iter - 1))
        else:  # Pokud je dimenze 3, tak se vykresli 3D graf pres plotly
            fig.show()


def confidence_set(data, alpha=0.05, color='red', label='set', fig=None, code_column=None, codes=False):
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
    S_hat = la.sqrtm(sigma_hat)

    # Matice pro transformaci do normalizovaneho prostoru
    Z = (np.linalg.pinv(S_hat) @ (data - np.tile(mean_hat, (n, 1))).T).T

    # Funkcni hodnoty inverznich distribucnich funkci
    F_esti = stats.f.ppf(1 - alpha, p, n - p) * p * (n - 1) / (n - p)
    if p == 2:  # (2D)
        space = np.linspace(0, 2 * np.pi, 100)  # Linearni prostor pro vykresleni
    else:  # p == 3 (3D)
        space = np.meshgrid(np.linspace(0, np.pi, 100),
                            np.linspace(0, 2 * np.pi, 100))  # Linearni prostor pro vykresleni

    # Body normalizovanych kruznic pro vykresleni
    if p == 2:  # (2D)
        zz_esti = np.array([np.sqrt(F_esti) * np.cos(space),
                            np.sqrt(F_esti) * np.sin(space)]).T
    else:  # p == 3 (3D)
        zz_esti = np.array([np.sqrt(F_esti) * np.sin(space[0]) * np.cos(space[1]),
                            np.sqrt(F_esti) * np.sin(space[0]) * np.sin(space[1]),
                            np.sqrt(F_esti) * np.cos(space[0])]).T
    # Body elips v puvodnim prostoru pro vykresleni
    xx_esti = np.dot(zz_esti, S_hat) + np.tile(mean_hat, (len(zz_esti), 1))

    # Vypocet Mahalanobisovy vzdalenosti
    mah_dist_z = np.zeros((n, 1))
    for i in range(n):
        mah_dist_z[i, 0] = np.dot(Z[i, :], Z[i, :])  # Zde se muze pocitat klidne v normalizovanem prostoru
    # Detekce outlieru
    data_out = data[np.where(mah_dist_z > F_esti)[0], :]

    if p == 2:  # (2D)
        # Vykresleni elips(y)
        plt.fill(xx_esti[:, 0], xx_esti[:, 1],
                 color=color, alpha=0.2, edgecolor=color, linestyle='--', linewidth=1, label=label)

        # Vykresleni dat a outlieru
        plt.plot(data[:, 0], data[:, 1], '+', color=color, ms=4)
        plt.plot(data_out[:, 0], data_out[:, 1], 'o', color=color, ms=4)
        if codes:
            for i, txt in enumerate(code_column):
                plt.annotate(txt, (data[i, 0], data[i, 1]), fontsize=8)
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
        fig.add_trace(go.Scatter3d(x=data[:, 0], y=data[:, 1], z=data[:, 2],
                                   mode='markers', marker=dict(size=3, color=color), name=label))
        fig.add_trace(go.Scatter3d(x=data_out[:, 0], y=data_out[:, 1], z=data_out[:, 2],
                                   mode='markers', marker=dict(size=8, color=color), name=label + ' outlier'))
        if codes:  # Anotace bodu dat
            old = list(fig['layout']['scene']['annotations'])  # Stare anotace
            new = [dict(showarrow=False, x=data[i, 0], y=data[i, 1], z=data[i, 2], text=code_column[i], opacity=0.8,
                        font=dict(color='black', size=10), yshift=15) for i in range(len(data))]  # Nove anotace
            fig.update_layout(scene=dict(annotations=old + new))  # Aktualizace anotaci

        fig.update_layout(margin=dict(l=0, r=0, b=0, t=0))


# # # ------------------------ TEST RESULTS ---------------------------------------------------------------------- # # #


def plot_test_results(results, combinations, my_choice, classes_dict, classes_dict_inv, dim=1,
                      num_of_tests=-1):
    """
    Funkce pro vykresleni vysledku testu (heatmapa a kruhovy graf)
    :param results: Vysledky testu
    :param combinations: Komibnace v dane dimenzi
    :param my_choice: Vybrane sloupce dat
    :param classes_dict: Slovnik trid (nazev -> cislo)
    :param classes_dict_inv: Slovnik trid (cislo -> nazev)
    :param dim: Dimenze dat
    :param num_of_tests: Pocty testu
    :return:
    """
    iteration = 0  # Pocitadlo testu, pouzito pro blokujici vykresleni

    for what in results.keys():  # Pro kazdou vybranou charakteristiku
        for par in results[what].keys():  # Pro kazdy parametricky / neparametricky test
            for test_name in results[what][par].keys():  # Pro kazdy test
                plot_test_results_heatmap(results, combinations, my_choice, classes_dict, classes_dict_inv, what, par,
                                          test_name, dim, block=(not circle_viable and iteration == num_of_tests - 1))
                if circle_viable:
                    plot_test_results_circular(results, combinations, my_choice, classes_dict, classes_dict_inv, what,
                                               par, test_name, dim, block=(iteration == num_of_tests - 1))
                iteration += 1


def plot_test_results_heatmap(results, combinations, my_choice, classes_dict, classes_dict_inv, what, par, test_name,
                              dim=1, block=False):
    """
    Funkce pro vykresleni vysledku testu
    :param results: Vysledky testu
    :param combinations: Komibnace v dane dimenzi
    :param my_choice: Vybrane sloupce dat
    :param classes_dict: Slovnik trid (nazev -> cislo)
    :param classes_dict_inv: Slovnik trid (cislo -> nazev)
    :param what: Vybrana charakteristika
    :param par: Parametricky / neparametricky test
    :param test_name: Nazev testu
    :param dim: Dimenze dat
    :param block: Blokujici vykresleni
    :return:
    """
    fig = plt.figure(figsize=(6 * len(combinations), 6))  # Vytvoreni figury

    for i in range(len(combinations)):  # Pro kazdou kombinaci
        ax = fig.add_subplot(101 + len(combinations) * 10 + i)  # Tri grafy vedle sebe

        # Vytvoreni nazvu
        if dim == 1:
            title = my_choice[i] + ": " + what + " " + par + " " + test_name + " results"
        else:
            title = ""
            for j in range(len(combinations[i])):
                if j == 0:
                    title = my_choice[combinations[i][j]]
                else:
                    title += " + " + my_choice[combinations[i][j]]
            title += ": " + what + " " + par + " " + test_name + " results"
        ax.title.set_text(title)

        # Naplneni matice cisly
        ax.matshow(results[what][par][test_name][i], cmap=plt.cm.Greens)
        for j in range(len(classes_dict.keys())):
            for k in range(len(classes_dict.keys())):
                value = results[what][par][test_name][i, j, k]
                # Vlozeni textu do matice (pokud je mensi nez 0.6, cerna barva, jinak bila)
                ax.text(j, k, f"{value:.3g}", ha="center", va="center", color="k" if value < 0.6 else "w")

        ax.set_xticks(np.arange(len(classes_dict.keys())))
        ax.set_yticks(np.arange(len(classes_dict.keys())))
        ax.set_xticklabels([classes_dict_inv[c] for c in classes_dict_inv])
        ax.set_yticklabels([classes_dict_inv[c] for c in classes_dict_inv])

    plt.show(block=block)  # Vykresleni + blokace


def plot_test_results_circular(results, combinations, my_choice, classes_dict, classes_dict_inv,
                               what, par, test_name, dim=1, block=False):
    """
    Funkce pro vykresleni vysledku testu (kruhovy graf)
    :param results: Vysledky testu
    :param combinations: Komibnace v dane dimenzi
    :param my_choice: Vybrane sloupce dat (nazvy)
    :param classes_dict: Slovnik trid (nazev -> cislo)
    :param classes_dict_inv: Slovnik trid (cislo -> nazev)
    :param what: Vybrana charakteristika
    :param par: Parametricky / neparametricky test
    :param test_name: Nazev testu
    :param dim: Dimenze dat
    :param block: Blokujici vykresleni
    :return:
    """
    # Priprava sektoru kruhoveho grafu
    sectors = {}  # Sektor je zaoblena cast grafu, ktera ma nejakou velikost (format slovniku)
    for i, key in enumerate(classes_dict.keys()):  # Pocet sektoru jsem urcil jako pocet trid
        sectors[key] = len(classes_dict.keys())  # Velikost jsem zde urcil puvodne jako pocet prvku ve tride
        # Pozdeji jsem usoudil, ze stejne velke sektory jsou lepsi, z duvodu pozdejsiho vypoctu sirky cary
        # Chtel jsem docilit toho, ze z tloustky cary je videt, jak moc jsou si tridy podobne, ale pokud by byla
        # velikost sektoru zavisla na poctu prvku, tak malo cetne tridy by mely velmi tenke cary, coz by nebylo ok

    cmap = {}
    for j in range(len(classes_dict.keys())):
        cmap[classes_dict_inv[j]] = colors_plotly[j]

    fig = plt.figure(figsize=(6 * len(combinations), 6))  # Vytvoreni figury

    for i in range(len(combinations)):  # Pro kazdou kombinaci
        ax = fig.add_subplot(101 + len(combinations) * 10 + i, polar=True)  # Tri grafy vedle sebe

        # Vytvoreni nazvu
        if dim == 1:
            title = my_choice[i] + ": " + what + " " + par + " " + test_name + " results"
        else:
            title = ""
            for j in range(len(combinations[i])):
                if j == 0:
                    title = my_choice[combinations[i][j]]
                else:
                    title += " + " + my_choice[combinations[i][j]]
            title += ": " + what + " " + par + " " + test_name + " results"

        # Vytvoreni kruhoveho grafu
        circos = Circos(sectors, space=3)  # Vytvoreni grafu ze sektoru s danou mezerou mezi sektory (3)
        for j, sector in enumerate(circos.sectors):  # Pro kazdy sektor
            track = sector.add_track((93, 100))  # Vytvoreni nejak tlusteho prouzku (zde 7 %)
            track.axis(fc=colors[j])  # Nastaveni barvy pruhu (opet odpovida predchozim barvam)
            track.text(sector.name, color="white", size=12)  # Vlozeni textu do pruhu v sektoru

        for j, row in enumerate(results[what][par][test_name][i]):  # Pro kazdy radek matice
            for k, val in enumerate(row):  # Pro kazdy prvek v radku
                if j == k:  # Sam se sebou neni potreba porovnavat
                    continue

                row_name = classes_dict_inv[j]  # Nazev radku
                row_size = sectors[row_name]  # Velikost radku
                # row_sector_half = row_size / 2  # Pulka sektoru je "input", druha pulka "output"
                row_sector_part = row_size / len(classes_dict.keys())  # N-tina sektoru (zde sestina)
                col_name = classes_dict_inv[k]  # Nazev sloupce
                col_size = sectors[col_name]  # Velikost sloupce
                # col_sector_half = col_size / 2  # Pulka sektoru je "input", druha pulka "output"
                col_sector_part = col_size / len(classes_dict.keys())  # N-tina sektoru (zde sestina)

                # Moje myslenka zde je, ze leva pulka sektoru slouzi jako "input" a prava jako "output"
                # To znamena ze prave pulky sektoru maji jenom zacatky sipek, leve pulky jenom konce sipek
                # Navic jsou cary tluste podle velikosti p-hodnoty, takze pokud je p-hodnota mala, tak je
                # spojnice tenka, naopak pokud je p-hodnota velka, tak je spojnice tlusta
                # Spojnice jsou mimo jine take vycentrovane na stredy "pul sektoru"
                # row_width = row_sector_half * val  # Sirka spojnice v radku
                # row_start = (row_sector_half - row_width) / 2  # Zacatek spojnice v radku
                # row_end = row_start + row_width  # Konec spojnice v radku
                # col_width = col_sector_half * val  # Sirka spojnice ve sloupci
                # col_start = col_sector_half + (col_sector_half - col_width) / 2  # Zacatek spojnice ve sloupci
                # col_end = col_start + col_width  # Konec spojnice ve sloupci

                # Novejsi myslenka je, ze sektor se nedeli na dve casti - "input" a "output", ale na tolik casti,
                # kolik je trid - zde 6; to znamena, ze kazdy sektor ma jakoby 5 output casti a 1 input cast
                # Vznikaji tim slabsi spojnice, ale graf je celkove prehlednejsi; hure se vycte p-hodnota, protoze
                # jsou vsechny spojnice o dost tenci, ale za tu prehlednost to stoji
                row_width = row_sector_part * val
                row_start = row_sector_part * k + (row_sector_part - row_width) / 2
                row_end = row_start + row_width
                col_width = col_sector_part * val
                col_start = col_sector_part * k + (col_sector_part - col_width) / 2
                col_end = col_start + col_width

                # Pridani spojnice mezi radkem a sloupcem (zde je velikost spojnice urcena hodnotou v matici)
                circos.link((row_name, row_start, row_end),  # Spoj radek na sirce od row_start do row_end
                            (col_name, col_end, col_start),  # Prohozeno start a end, aby se spojnice nekrizily
                            color=colors[j],  # Barva opet konzistentni
                            direction=1)  # Direction dela ze spojnice smerovou spojnici (sipku)

        circos.text(title, color="black", deg=0, r=120, size=12)  # Vlozeni textu do grafu (nazev grafu)

        circos.plotfig(ax=ax)  # Vykresleni grafu
    plt.show(block=block)  # Vykresleni + blokace
