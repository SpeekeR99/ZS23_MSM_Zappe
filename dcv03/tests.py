import numpy as np
import scipy.stats as stats
import copy


# # # ------------------------ 1D TESTS -------------------------------------------------------------------------- # # #


def one_dim_tests(samples, my_choice, classes_dict):
    """
    Provede vsechny testy pro jednorozmerna data
    :param samples: Data
    :param my_choice: Vybrane sloupce
    :param classes_dict: Slovnik trid (nazev -> cislo)
    :return: Slovnik vysledku testu (stredni hodnota, rozptyl)
    """
    mean_results = one_dim_mean_tests(samples, my_choice, classes_dict)
    var_results = one_dim_var_tests(samples, my_choice, classes_dict)
    return {"Mean": mean_results, "Var": var_results}


def one_dim_mean_tests(samples, my_choice, classes_dict):
    """
    Provede vsechny testy pro stredni hodnotu jednorozmernych dat
    :param samples: Data
    :param my_choice: Vybrane sloupce
    :param classes_dict: Slovnik trid (nazev -> cislo)
    :return: Slovnik vysledku testu (parametricke, neparametricke)
    """
    parametric_results = one_dim_parametric_mean_tests(samples, my_choice, classes_dict)
    nonparametric_results = one_dim_nonparametric_mean_tests(samples, my_choice, classes_dict)
    return {"Param": parametric_results, "Nonparam": nonparametric_results}


def one_dim_parametric_mean_tests(samples, my_choice, classes_dict):
    """
    Provede parametricke testy pro stredni hodnotu jednorozmernych dat
    :param samples: Data
    :param my_choice: Vybrane sloupce
    :param classes_dict: Slovnik trid (nazev -> cislo)
    :return: Slovnik vysledku testu (t-test, ANOVA)
    """
    # Pripraveni vysledku
    ttest_results = np.zeros((len(my_choice), len(classes_dict.keys()), len(classes_dict.keys())))
    pair_anova_results = np.zeros((len(my_choice), len(classes_dict.keys()), len(classes_dict.keys())))

    for i in range(len(my_choice)):  # Pro kazdy sloupec
        for j in range(len(classes_dict.keys())):  # Pro kazdou tridu
            for k in range(j, len(classes_dict.keys())):  # Pro kazdou dvojici trid
                x = samples["Data"][samples["Classes"] == j][:, i]  # Data tridy j
                y = samples["Data"][samples["Classes"] == k][:, i]  # Data tridy k
                ttest_results[i, j, k] = ttest_results[i, k, j] = stats.ttest_ind(x, y)[1]
                pair_anova_results[i, j, k] = pair_anova_results[i, k, j] = stats.f_oneway(x, y)[1]
    pair_anova_results[np.isnan(pair_anova_results)] = 1  # NaN -> 1 (korekce SciPy vysledku)

    return {"T-test": ttest_results, "Pairwise ANOVA": pair_anova_results}


def one_dim_nonparametric_mean_tests(samples, my_choice, classes_dict):
    """
    Provede neparametricke testy pro stredni hodnotu jednorozmernych dat
    :param samples: Data
    :param my_choice: Vybrane sloupce
    :param classes_dict: Slovnik trid (nazev -> cislo)
    :return: Slovnik vysledku testu (Kruskal-Wallis, Mann-Whitney)
    """
    # Pripraveni vysledku
    kruskal_wallis_results = np.zeros((len(my_choice), len(classes_dict.keys()), len(classes_dict.keys())))
    man_whitney_results = np.zeros((len(my_choice), len(classes_dict.keys()), len(classes_dict.keys())))

    for i in range(len(my_choice)):  # Pro kazdy sloupec
        for j in range(len(classes_dict.keys())):  # Pro kazdou tridu
            for k in range(j, len(classes_dict.keys())):  # Pro kazdou dvojici trid
                x = samples["Data"][samples["Classes"] == j][:, i]  # Data tridy j
                y = samples["Data"][samples["Classes"] == k][:, i]  # Data tridy k
                kruskal_wallis_results[i, j, k] = kruskal_wallis_results[i, k, j] = stats.kruskal(x, y)[1]
                man_whitney_results[i, j, k] = man_whitney_results[i, k, j] = stats.mannwhitneyu(x, y)[1]

    return {"Kruskal-Wallis": kruskal_wallis_results, "Mann-Whitney": man_whitney_results}


def one_dim_var_tests(samples, my_choice, classes_dict):
    """
    Provede vsechny testy pro rozptyl jednorozmernych dat
    :param samples: Data
    :param my_choice: Vybrane sloupce
    :param classes_dict: Slovnik trid (nazev -> cislo)
    :return: Slovnik vysledku testu (parametricke, neparametricke)
    """
    parametric_results = one_dim_parametric_var_tests(samples, my_choice, classes_dict)
    nonparametric_results = one_dim_nonparametric_var_tests(samples, my_choice, classes_dict)
    return {"Param": parametric_results, "Nonparam": nonparametric_results}


def one_dim_parametric_var_tests(samples, my_choice, classes_dict):
    """
    Provede parametricke testy pro rozptyl jednorozmernych dat
    :param samples: Data
    :param my_choice: Vybrane sloupce
    :param classes_dict: Slovnik trid (nazev -> cislo)
    :return: Slovnik vysledku testu (F-test, Bartlett)
    """
    # Pripraveni vysledku
    bartlett_results = np.zeros((len(my_choice), len(classes_dict.keys()), len(classes_dict.keys())))
    levene_results = np.zeros((len(my_choice), len(classes_dict.keys()), len(classes_dict.keys())))

    for i in range(len(my_choice)):  # Pro kazdy sloupec
        for j in range(len(classes_dict.keys())):  # Pro kazdou tridu
            for k in range(j, len(classes_dict.keys())):  # Pro kazdou dvojici trid
                x = samples["Data"][samples["Classes"] == j][:, i]  # Data tridy j
                y = samples["Data"][samples["Classes"] == k][:, i]  # Data tridy k
                bartlett_results[i, j, k] = bartlett_results[i, k, j] = stats.bartlett(x, y)[1]
                levene_results[i, j, k] = levene_results[i, k, j] = stats.levene(x, y, center="mean")[1]

    return {"Bartlett": bartlett_results, "Levene (ANOVA)": levene_results}


def one_dim_nonparametric_var_tests(samples, my_choice, classes_dict):
    """
    Provede neparametricke testy pro rozptyl jednorozmernych dat
    :param samples: Data
    :param my_choice: Vybrane sloupce
    :param classes_dict: Slovnik trid (nazev -> cislo)
    :return: Slovnik vysledku testu (Fligner-Killeen, Levene)
    """
    # Pripraveni vysledku
    fligner_killeen_results = np.zeros((len(my_choice), len(classes_dict.keys()), len(classes_dict.keys())))
    levene_results = np.zeros((len(my_choice), len(classes_dict.keys()), len(classes_dict.keys())))

    for i in range(len(my_choice)):  # Pro kazdy sloupec
        for j in range(len(classes_dict.keys())):  # Pro kazdou tridu
            for k in range(j, len(classes_dict.keys())):  # Pro kazdou dvojici trid
                x = samples["Data"][samples["Classes"] == j][:, i]  # Data tridy j
                y = samples["Data"][samples["Classes"] == k][:, i]  # Data tridy k
                fligner_killeen_results[i, j, k] = fligner_killeen_results[i, k, j] = \
                    stats.fligner(x, y, center="mean")[1]
                # Levene test se provadi na absolutni hodnotu odchylky od stredni hodnoty
                x_levene = np.abs(x - np.mean(x))
                y_levene = np.abs(y - np.mean(y))
                levene_results[i, j, k] = levene_results[i, k, j] = stats.kruskal(x_levene, y_levene)[1]

    return {"Fligner-Killeen": fligner_killeen_results, "Levene (Kruskal-Wallis)": levene_results}


# # # ------------------------ N-DIM TESTS ----------------------------------------------------------------------- # # #


def prepare_n_dim_data(samples, combinations, dim):
    """
    Pripravi data pro n-dim testy (vytvori kombinace sloupcu)
    :param samples: Data
    :param combinations: Pozadovane kombinace v dane dimenzi
    :param dim: Dimenze
    :return: Vice dimenzionalni data
    """
    bigger_dim_data = np.zeros((len(combinations), samples["Data"].shape[0], dim))

    new_samples = copy.deepcopy(samples)  # Vytvoreni kopie dat
    for i in range(0, len(combinations)):
        for j in range(0, dim):
            bigger_dim_data[i, :, j] = samples["Data"][:, combinations[i][j]]  # Vytvoreni kombinaci sloupcu
    new_samples["Data"] = bigger_dim_data  # Ulozeni vysledku
    return new_samples


def n_dim_tests(samples, combinations, classes_dict, dim):
    """
    Provede testy pro n-dim data
    :param samples: Data
    :param combinations: Kombinace v dane dimenzi
    :param classes_dict: Slovnik trid (nazev -> cislo)
    :param dim: Dimenze
    :return: Slovnik vysledku testu (Mean, Var)
    """
    mean_results = n_dim_mean_tests(samples, len(combinations), classes_dict, dim=dim)
    var_results = n_dim_var_tests(samples, len(combinations), classes_dict, dim=dim)
    return {"Mean": mean_results, "Var": var_results}


def n_dim_mean_tests(samples, num_comb, classes_dict, dim):
    """
    Provede testy pro stredni hodnotu n-dim dat
    (pouze parametricke testy, proto$ze neparametricke testy nejdou pro vice dimenzi)
    :param samples: Data
    :param num_comb: Pocet kombinaci
    :param classes_dict: Slovnik trid (nazev -> cislo)
    :param dim: Dimenze
    :return: Slovnik vysledku testu (Param)
    """
    parametric_results = n_dim_parametric_mean_tests(samples, num_comb, classes_dict, dim=dim)
    return {"Param": parametric_results}


def n_dim_parametric_mean_tests(samples, num_comb, classes_dict, dim):
    """
    Provede parametricke testy pro stredni hodnotu n-dim dat
    Vzorecky jsou prevzaty z prednasky 4 (str. 16)
    :param samples: Data
    :param num_comb: Pocet kombinaci
    :param classes_dict: Slovnik trid (nazev -> cislo)
    :param dim: Dimenze
    :return: Slovnik vysledku testu (Chi2)
    """
    # Pripraveni matice strednich hodnot a kovariancnich matic
    averages = np.zeros((num_comb, len(classes_dict.keys()), dim))
    cov_matrix = np.zeros((num_comb, len(classes_dict.keys()), dim, dim))

    for i in range(num_comb):  # Pro kazdou kombinaci
        for j in range(len(classes_dict.keys())):  # Pro kazdou tridu
            data = samples["Data"][i][samples["Classes"] == j]  # Data tridy j
            averages[i, j] = np.mean(data, axis=0)  # Vypocet stredni hodnoty
            # cov_matrix[i, j] = (np.dot(data.T, data) / len(data) - np.dot(averages[i, j].reshape(dim, 1), averages[i, j].reshape(1, dim))) * len(data) / (len(data) - 1)
            cov_matrix[i, j] = np.cov(data, rowvar=False)  # Vypocet kovariancni matice

    # Vypocet chi2 testu
    chi2_results = np.zeros((num_comb, len(classes_dict.keys()), len(classes_dict.keys())))

    for i in range(num_comb):  # Pro kazdou kombinaci
        for j in range(len(classes_dict.keys())):  # Pro kazdou tridu
            for k in range(j, len(classes_dict.keys())):  # Pro kazdou dvojici trid
                data_j = samples["Data"][i][samples["Classes"] == j]  # Data tridy j
                data_k = samples["Data"][i][samples["Classes"] == k]  # Data tridy k
                diff = averages[i, j] - averages[i, k]  # Rozdil strednich hodnot
                cov = cov_matrix[i, j] / len(data_j) + cov_matrix[i, k] / len(data_k)  # Kovariancni matice
                stat = np.dot(np.dot(diff.T, np.linalg.inv(cov)), diff)  # Vypocet chi2 statistiky
                chi2_results[i, j, k] = chi2_results[i, k, j] = stats.chi2.sf(stat, dim)  # Vypocet p-hodnoty

    return {"Chi2": chi2_results}


def n_dim_var_tests(samples, num_comb, classes_dict, dim):
    """
    Provede testy pro rozptyl n-dim data
    (pouze parametricke testy, protoze neparametricke testy nejdou pro vice dimenzi)
    :param samples: Data
    :param num_comb: Poct kombinaci
    :param classes_dict: Slovnik trid (nazev -> cislo)
    :param dim: Dimenze
    :return: Slovnik vysledku testu (Param)
    """
    parametric_results = n_dim_parametric_var_tests(samples, num_comb, classes_dict, dim=dim)
    return {"Param": parametric_results}


def n_dim_parametric_var_tests(samples, num_comb, classes_dict, dim):
    """
    Provede parametricke testy pro rozptyl n-dim dat
    Vzorecky jsou prevzaty z prednasky 4 (str. 17)
    :param samples: Data
    :param num_comb: Poct kombinaci
    :param classes_dict: Slovnik trid (nazev -> cislo)
    :param dim: Dimenze
    :return: Slovnik vysledku testu (Chi2)
    """
    # Chi2 test
    chi2_results = np.zeros((num_comb, len(classes_dict.keys()), len(classes_dict.keys())))

    for i in range(num_comb):  # Pro kazdou kombinaci
        for j in range(len(classes_dict.keys())):  # Pro kazdou tridu
            for k in range(j, len(classes_dict.keys())):  # Pro kazdou dvojici trid
                data_j = samples["Data"][i][samples["Classes"] == j]  # Data tridy j
                data_k = samples["Data"][i][samples["Classes"] == k]  # Data tridy k
                cov_j = np.cov(data_j, rowvar=False)  # Kovariancni matice tridy j
                cov_k = np.cov(data_k, rowvar=False)  # Kovariancni matice tridy k
                n_j = len(data_j)  # Pocet dat tridy j
                n_k = len(data_k)  # Pocet dat tridy k
                n = n_j + n_k  # Celkovy pocet dat
                cov = (n_j * cov_j + n_k * cov_k) / n  # Kovariancni matice
                left_term = n * np.log(np.linalg.det(cov))  # Levy clen rozdilu
                right_term = n_j * np.log(np.linalg.det(cov_j)) + n_k * np.log(
                    np.linalg.det(cov_k))  # Pravy clen rozdilu
                diff = left_term - right_term  # Rozdil
                chi2_results[i, j, k] = chi2_results[i, k, j] = stats.chi2.sf(diff,
                                                                              dim * (dim + 1) / 2)  # Vypocet p-hodnoty

    return {"Chi2": chi2_results}
