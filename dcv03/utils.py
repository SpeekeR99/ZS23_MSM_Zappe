import numpy as np
import pandas as pd
import scipy.stats as stats


# # # ------------------------ DATA ------------------------------------------------------------------------------ # # #


def load_data(my_choice, filepath="Rocks.xls", sheet_name="Data"):
    """
    Nacte data z excelu a vrati slovnik obsahujici kody hornin, jejich tridy a data, ktera jsou v my_choice
    Poznamka: chyba v datech na radku 40 (excel) resp. 39 (tabulka) ("granite ") (mezera navic)
    :param my_choice: Vyber sloupcu, ktere chceme pouzit
    :param filepath: Cesta k souboru
    :param sheet_name: Jmeno listu v excelu
    :return: Slovnik obsahujici kody hornin, jejich tridy a data, ktera jsou v my_choice
    """
    data = pd.read_excel(filepath, sheet_name=sheet_name)

    codes = data["Code"].to_numpy()
    classes = data["Class"].to_numpy()
    cols = np.zeros((len(my_choice), len(classes)))  # Vybrane sloupce
    for i in range(len(my_choice)):
        cols[i] = data[my_choice[i]].to_numpy()
    unique_list = []  # Unikatni tridy
    for x in classes:
        if x not in unique_list:
            unique_list.append(x)
    classes_dict = {c: i for i, c in enumerate(unique_list)}  # Slovnik trid (trida -> cislo)
    classes_dict_inv = {v: k for k, v in classes_dict.items()}  # Slovnik trid (cislo -> trida)

    samples = {"Codes": codes.T, "Classes": np.array([classes_dict[c] for c in classes]).T, "Data": cols.T}
    return samples, classes_dict, classes_dict_inv


# # # ------------------------ STATISTICS ------------------------------------------------------------------------ # # #


def descriptive_statistics(samples, my_choice, classes_dict_inv):
    """
    Vypocita deskriptivni statistiky pro vybrane sloupce a tridy
    :param samples: Data
    :param my_choice: Vyberane sloupce
    :param classes_dict_inv: Slovnik trid (cislo -> trida)
    :return: Slovnik obsahujici deskriptivni statistiky pro vybrane sloupce a tridy
    """
    statistics = {}

    for class_ in classes_dict_inv.keys():  # Pro kazdou tridu
        statistics[classes_dict_inv[class_]] = {}  # Vytvorime slovnik pro danou tridu

        for column, thing in enumerate(my_choice):  # Pro kazdy sloupec, respektivne pro kazdou charakteristiku
            data = samples["Data"][samples["Classes"] == class_][:, column]  # Vybereme data pro danou tridu a sloupec
            number_of_samples = len(data)  # Pocet vzorku
            mean = np.mean(data)  # Prumer
            var = np.var(data, ddof=1)  # Rozptyl
            std = np.std(data)  # Smerodatna odchylka
            sem = stats.sem(data)  # Standardni chyba prumeru
            median = np.median(data)  # Median
            q1 = np.quantile(data, 0.25)  # Prvni kvartil
            q3 = np.quantile(data, 0.75)  # Treti kvartil
            iqr = q3 - q1  # Interkvartilovy rozsah
            normality_test_p_val = stats.shapiro(data)[1]  # P-hodnota normality testu
            statistics[classes_dict_inv[class_]][thing] = [number_of_samples, mean, var, std, sem, median, q1, q3, iqr,
                                                           normality_test_p_val]

    return statistics


def print_nicely(stats, my_choice, classes_dict_inv):
    """
    Vytiskne statistiky (hezky :) )
    :param stats: Statistiky co se maji vytisknout (format slovniku z descriptive_statistics)
    :param my_choice: Vybrane sloupce
    :param classes_dict_inv: Slovnik trid (cislo -> trida)
    :return:
    """
    for class_ in classes_dict_inv.keys():  # Pro kazdou tridu
        print(f"Class: {classes_dict_inv[class_]}")  # Vytiskneme tridu

        for column, thing in enumerate(my_choice):  # Pro kazdy sloupec, respektivne pro kazdou charakteristiku
            # Prilis hezke vytisknuti statistik :)
            print(f"\t{thing}:")
            print(f"\t\tNumber of samples:\t\t{stats[classes_dict_inv[class_]][thing][0]}")
            print(f"\t\tMean:\t\t\t\t{stats[classes_dict_inv[class_]][thing][1]}")
            print(f"\t\tVariance:\t\t\t{stats[classes_dict_inv[class_]][thing][2]}")
            print(f"\t\tStandard deviation:\t\t{stats[classes_dict_inv[class_]][thing][3]}")
            print(f"\t\tStandard error mean:\t\t{stats[classes_dict_inv[class_]][thing][4]}")
            print(f"\t\tMedian:\t\t\t\t{stats[classes_dict_inv[class_]][thing][5]}")
            print(f"\t\tFirst quartile:\t\t\t{stats[classes_dict_inv[class_]][thing][6]}")
            print(f"\t\tThird quartile:\t\t\t{stats[classes_dict_inv[class_]][thing][7]}")
            print(f"\t\tInterquartile range:\t\t{stats[classes_dict_inv[class_]][thing][8]}")
            print(f"\t\tNormality test p-value:\t\t{stats[classes_dict_inv[class_]][thing][9]}")
        print()
