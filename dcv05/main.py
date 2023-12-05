from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from visuals import *
from eval import *
from utils import *


def main():
    """
    Hlavni funkce programu
    """
    # ------------------------------ NACTENI DAT -----------------------------------------------------------------------

    # Pocet "zbytecnych" sloupcu (sloupce, ktere nejsou soucasti dat)
    number_of_unimportant_columns = 4
    # Pomer rozdeleni dat na trenovaci a testovaci mnozinu
    split_ratio = 0.8

    # Nacteni dat
    data, classes_dict, classes_dict_inv = load_data(filepath="Rocks.xls", sheet_name="Data")

    # Permutace dat kvuli pozdejsi diskriminacni analyze (kdybych neudelal permutaci, tak vzdy oriznu prvnich x % dat
    # a zbytek pouziju jako testovaci mnozinu, coz zrovna v pripade struktury Rocks.xls neni idealni)
    perm = np.random.permutation(len(data["Class"]))
    for key in data.keys():
        data[key] = data[key][perm]

    # Nacteni dat a trid
    samples = np.array([data[key] for key in list(data.keys())[number_of_unimportant_columns:]]).T
    classes = data["Class"]

    # ------------------------------ PCA -------------------------------------------------------------------------------

    # Transformace dat na normalizovane normalni rozdeleni (standardizace)
    sc = StandardScaler()
    samples_sc = sc.fit_transform(samples)

    # PCA
    pca = PCA(n_components=len(data.keys()) - number_of_unimportant_columns)
    y = pca.fit_transform(samples_sc)

    # Vlastni cisla, resp. rozptyly
    eigenvalues = pca.explained_variance_
    print(f"Eigenvalues: {eigenvalues}")

    # Vykresleni PCA - pareto, 2D prostor prvnich dvou komponent, biplot
    plot_pareto(pca)
    plot_data_in_2D_space(y, classes, classes_dict)
    plot_biplot(pca, y, data, classes, classes_dict, number_of_unimportant_columns)

    # Z grafu je videt, ze abych zachoval 90 % informace, musim pouzit alespon 6 komponent

    # ------------------------------ DISKRIMINACNI ANALYZA -------------------------------------------------------------

    # Podvyber dat (zminovanych 6 komponent)
    new_samples = y[:, :6]

    # Rozdeleni dat na trenovaci a testovaci mnozinu
    train_x = new_samples[:int(len(samples) * split_ratio)]
    train_y = classes[:int(len(samples) * split_ratio)]
    test_x = new_samples[int(len(samples) * split_ratio):]
    test_y = classes[int(len(samples) * split_ratio):]

    # Diskriminacni analyza
    model = LinearDiscriminantAnalysis()

    # Transformace dat do pozadovaneho formatu od knihovny
    train_x = sc.fit_transform(train_x)
    test_x = sc.transform(test_x)

    # Trenovani modelu
    model.fit(train_x, train_y)

    # Evaluace klasifikace
    predicted_y = model.predict(test_x)
    acc = np.sum(predicted_y == test_y) / len(test_y)  # Accuracy
    print(f"Accuracy of classification (on test data): {acc * 100} %")

    conf_mat = confusion_matrix(test_y, predicted_y, classes_dict)  # Matice zamen
    plot_conf_mat(conf_mat, classes_dict.keys())

    print("Cross validation:")
    cross_validation(new_samples, classes)  # Cross validace pomoci k-fold a leave-one-out (KFCV a LOOCV)

    rocs = roc(test_x, test_y, classes_dict)  # ROC krivky
    plot_roc_curves(rocs, classes_dict)

    # Z matice zamen a ROC krivek je videt, ze nejvice problemovou je trida Breccia a Diorite, coz je logicke,
    # protoze tyto tridy jsou nejmene zastoupene v datech

    # Dokonce se casto stava (po opakovanem spousteni), ze se jedna z techto trid vubec nedostane do testovaci sady,
    # program by pak padal na chybu pri vypoctu ROC krivky, je nutne tuto tridu preskocit (viz. funkce roc())


if __name__ == "__main__":
    main()
