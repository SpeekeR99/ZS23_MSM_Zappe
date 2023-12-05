import itertools
from visuals import *
from tests import *
from utils import *


def main():
    """
    Hlavni funkce programu
    :return:
    """
    # Vybrane sloupce, charakteristiky
    # Muj algoritmus nahodneho vyberu:
    #   D jako Dominik, D je ctvrte pismeno abecedy -> 4. sloupec = MVAP
    #   1999 rok narozeni, hodne devitek :) -> 9. sloupec = RCHQ
    #   Mam rad binarni soustavu -> 2. sloupec = RCSG
    my_choice = ["MVAP", "RCHQ", "RCSG"]
    if len(sys.argv) == 4:
        my_choice = sys.argv[1:]
    print(f"Selected columns: {my_choice}")

    # Nacteni dat
    samples, classes_dict, classes_dict_inv = load_data(my_choice, filepath="Rocks.xls", sheet_name="Data")

    # 1D
    dim = 1
    combinations = list(itertools.combinations(range(samples["Data"].shape[1]), dim))

    # Deskriptivni statistiky
    stats = descriptive_statistics(samples, my_choice, classes_dict_inv)
    print_nicely(stats, my_choice, classes_dict_inv)

    # Vykresleni dat
    plot_samples(samples, combinations, my_choice, classes_dict_inv, dim=dim)
    plot_samples(samples, combinations, my_choice, classes_dict_inv, codes=True, dim=dim)
    # Vykresleni boxplotu
    plot_boxplots(samples, my_choice, classes_dict_inv)
    plot_boxplots(samples, my_choice, classes_dict_inv, codes=True)

    # Testy
    one_dim_test_results = one_dim_tests(samples, my_choice, classes_dict)
    plot_test_results(one_dim_test_results, combinations, my_choice, classes_dict, classes_dict_inv, dim=dim,
                      num_of_tests=8)

    # nD (konkretne 2D, 3D)
    dims = [2, 3]
    for dim in dims:  # Pro kazdou dimenzi
        # Pripraveni nD dat
        combinations = list(itertools.combinations(range(samples["Data"].shape[1]), dim))
        n_dim_samples = prepare_n_dim_data(samples, combinations, dim=dim)

        # Vykresleni dat (funguje pouze pro 2D)
        plot_samples(samples, combinations, my_choice, classes_dict_inv, dim=dim)
        plot_samples(samples, combinations, my_choice, classes_dict_inv, codes=True, dim=dim)
        # Vykresleni elips (funguje pouze pro 2D a 3D)
        plot_elipses(n_dim_samples, combinations, my_choice, classes_dict_inv, dim=dim)
        plot_elipses(n_dim_samples, combinations, my_choice, classes_dict_inv, codes=True, dim=dim)

        # Testy
        n_dim_test_results = n_dim_tests(n_dim_samples, combinations, classes_dict, dim=dim)
        plot_test_results(n_dim_test_results, combinations, my_choice, classes_dict, classes_dict_inv, dim=dim,
                          num_of_tests=2)


if __name__ == "__main__":
    main()
