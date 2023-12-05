import numpy as np
import pandas as pd


def load_data(filepath="Rocks.xls", sheet_name="Data"):
    """
    Nacteni dat ze souboru
    :param filepath: Cesta k souboru
    :param sheet_name: Nazev listu v excelu
    :return: Data, slovnik trid (trida -> cislo), slovnik trid (cislo -> trida)
    """
    data = pd.read_excel(filepath, sheet_name=sheet_name)
    data = data.to_dict(orient="list")

    classes = data["Class"]
    unique_list = []  # Unikatni tridy
    for x in classes:
        if x not in unique_list:
            unique_list.append(x)
    classes_dict = {c: i for i, c in enumerate(unique_list)}  # Slovnik trid (trida -> cislo)
    classes_dict_inv = {v: k for k, v in classes_dict.items()}  # Slovnik trid (cislo -> trida)

    for key in data.keys():
        data[key] = np.array(data[key])

    data["Class"] = np.array([classes_dict[c] for c in classes]).T

    return data, classes_dict, classes_dict_inv

