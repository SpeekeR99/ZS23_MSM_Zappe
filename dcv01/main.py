import time as t
import numpy as np
import scipy.integrate as integrate
from scipy.optimize import fsolve
import matplotlib.pyplot as plt

# Globalni konstanty
EX = 0.902778                   # Vypocitana stredni hodnota X (desetinny tvar, protoze Wolfram Alpha mi nedal zlomek)
VARX = 0.306024                 # Vypocitany rozptyl X (desetinny tvar, protoze Wolfram Alpha mi nedal zlomek)
EY = 115 / 162.                 # Vypocitana stredni hodnota Y
VARY = 24065 / 183708.          # Vypocitany rozptyl Y
COVXY = 1145 / 2268. - EX * EY  # Vypocitana kovariance X a Y


def three_dim_plot(func, x_space, y_space, top_view=False):
    """
    FUnkce pro vykresleni 3D grafu funkce dvou promennych
    :param func: Vykreslovana funkce (ocekavana lambda funkce)
    :param x_space: Prostor hodnot promenne x (hodnoty, ktere ma smysl vykreslit)
    :param y_space: Prostor hodnot promenne y (hodnoty, ktere ma smysl vykreslit)
    :param top_view: True, pokud se ma vykreslit pohled shora (default False)
    :return:
    """
    X, Y = np.meshgrid(x_space, y_space)  # Vytvoreni matic X a Y
    Z = np.array([func(x_space[i], y_space[j])  # Vypocet hodnot funkce
                  for i in range(len(x_space))  # pro kazdy bod v prostoru
                  for j in range(len(y_space))]
                 ).reshape(len(x_space), len(y_space))
    fig = plt.figure()
    fig.suptitle("Probability density function")
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(X, Y, Z, rstride=5, cstride=5, cmap='plasma', edgecolor='none')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('f(x, y)')
    if top_view:  # Pokud se ma vykreslit pohled shora, je nutne orotovat "kameru"
        ax.view_init(90, 270)
    plt.show(block=False)


def two_dim_plot(funcs, x_space, labels=None):
    """
    Funkce pro vykresleni grafu funkce jedne promenne
    :param funcs: Vykreslovane funkce (ocekavana lambda funkce)
    :param x_space: Prostor hodnot promenne x (hodnoty, ktere ma smysl vykreslit)
    :param labels: List obsahujici popisky os a grafu
    :return:
    """
    fig = plt.figure()
    fig.suptitle("Probability density function and distribution function")
    fig.subplots_adjust(hspace=0.5)  # Vzdalenost mezi grafy (vykresluji spolu vzdy funkci hustoty a distribucni funkci)
    for i, func in enumerate(funcs):
        ax = fig.add_subplot(len(funcs), 1, i + 1)
        ax.plot(x_space, [func(x) for x in x_space])
        ax.grid()
        ax.set_xlabel(labels[i] if labels else 'x')
        ax.set_ylabel(labels[i + len(funcs)] if labels else 'f(x)')
        ax.set_title(labels[i + len(funcs)] if labels else 'f(x)')
    plt.show(block=False)


def hist_plot(x, bins=100, labels=None):
    """
    Funkce pro vykresleni histogramu
    :param x: Vygenerovana data
    :param bins: Pocet intervalu (default 100) (pocet sloupcu histogramu)
    :param labels: List obsahujici popisky os a grafu
    :return:
    """
    fig = plt.figure()
    fig.suptitle("Histograms")  # Mozna jsem mohl vymyslet lepsi nadpis :)
    fig.subplots_adjust(hspace=0.5)
    ax = fig.add_subplot(211)
    ax.grid(linestyle='--')
    ax.hist(x, bins=bins, density=True)  # density=True - normalizace histogramu
    ax.set_xlabel(labels[0] if labels else 'x')
    ax.set_ylabel(labels[1] if labels else 'f(x)')
    ax.set_title(labels[1] if labels else 'f(x)')
    ax = fig.add_subplot(212)
    ax.grid(linestyle='--')
    ax.hist(x, bins=bins, density=True, cumulative=True)  # cumulative=True - distribucni funkce
    ax.set_xlabel(labels[2] if labels else 'x')
    ax.set_ylabel(labels[3] if labels else 'F(x)')
    ax.set_title(labels[3] if labels else 'F(x)')
    plt.show(block=False)


def empirical_plot(func, x_space, x, labels=None):
    """
    Funkce pro vykresleni empiricke distribucni funkce
    :param func: Vykreslovana funkce (ocekavana lambda funkce)
    :param x_space: Prostor hodnot promenne x (hodnoty, ktere ma smysl vykreslit)
    :param x: Vygenerovana data
    :param labels: List obsahujici popisky os a grafu
    :return:
    """
    fig = plt.figure()
    plt.plot(x_space, [func(x) for x in x_space])  # Vykresleni teoreticke distribucni funkce
    plt.plot(np.sort(x), np.linspace(0, 1, len(x)))  # Vykresleni empiricke distribucni funkce
    fig.suptitle("Empirical distribution function")
    plt.grid()
    plt.legend([labels[1], labels[1].replace('(', '*(')])  # * oznacuje empirickou distribucni funkci
    plt.xlabel(labels[0] if labels else 'x')
    plt.ylabel(labels[1] if labels else 'F(x)')
    plt.show(block=False)


def scatter_plot(x, y, method1=False):
    """
    Funkce pro vykresleni jednotlivych nagenerovanych bodu
    :param x: List obsahujici hodnoty promenne x (x-ove souradnice bodu)
    :param y: List obsahujici hodnoty promenne y (y-ove souradnice bodu)
    :param method1: True - byla pouzita metoda 1, False - byla pouzita metoda 2 (pro popis grafu)
    :return:
    """
    fig = plt.figure()
    if method1:
        fig.suptitle("Generated samples using method 1")
    else:
        fig.suptitle("Generated samples using method 2")
    plt.plot(x, y, "x")
    plt.grid()
    plt.show()


def print_statistics(x, y):
    """
    Funkce pro vypocet a vypis statistik na zaklade nagenerovanych dat
    :param x: List obsahujici hodnoty promenne x
    :param y: List obsahujici hodnoty promenne y
    :return: Vyberovy prumer x, vyberovy rozptyl x, vyberovy prumer y, vyberovy rozptyl y, kovariance x a y
    """
    print("E(X) = {0}".format(np.mean(x)))
    print("Var(X) = {0}".format(np.var(x)))
    print("E(Y) = {0}".format(np.mean(y)))
    print("Var(Y) = {0}".format(np.var(y)))
    print("Cov(X, Y) =\n", np.cov(x, y, bias=True))  # bias=True - aby vychazeli stejne hodnoty jako np.var()
    return np.mean(x), np.var(x), np.mean(y), np.var(y), np.cov(x, y, bias=True)


def first_method(f_join, samples=100, verbose=True, plot=True):
    """
    Prvni metoda pro generovani nahodnych bodu ze cviceni
    Funkce si na zaklade predane funkce f_join vypocita marginalni funkce hustoty pravdepodobnosti
    Pomoci integrace zjisti marginalni distribucni funkce
    Nasledne vygeneruje nahodne body odpovidajici teto distribucni funkci a vse v prubehu behu vykresli
    :param f_join: Zadana funkce f(x, y) (ocekavana lambda funkce)
    :param samples: Kolik bodu se ma vygenerovat (default 100)
    :param verbose: True - vypisovat cas, False - nevypisovat cas (pouzito z duvodu testovani konvergence)
    :param plot: True - vykreslit grafy, False - nevykreslovat grafy (pouzito z duvodu testovani konvergence)
    :return: Vygenerovane body (dvojice x, y)
    """
    # Vykresleni zadane funkce f(x, y) (i shora)
    if plot:
        three_dim_plot(f_join, np.linspace(0, 5/2., 100), np.linspace(0, 5/3., 100))
        three_dim_plot(f_join, np.linspace(0, 5/2., 100), np.linspace(0, 5/3., 100), top_view=True)

    # Vypocet marginalnich funkci hustoty pravdepodobnosti a distribucnich funkci
    f_x = lambda x: (0 <= x <= 5/2.) * (8 / 375.) * ((5 - 2 * x) ** 2) * (x * x + x + 1)
    F_x = lambda x: integrate.quad(f_x, 0, x)[0]  # integrate.quad() ze scipy je ekvivalent k Matlab integral()
    f_y = lambda y: (0 <= y <= 5/3.) * (-2 / 125.) * y * (27 * y * y * y - 162 * y * y + 351 * y - 260)
    F_y = lambda y: integrate.quad(f_y, 0, y)[0]

    # Vykresleni marginalnich funkci hustoty pravdepodobnosti a distribucnich funkci
    if plot:
        two_dim_plot([f_x, F_x], np.linspace(0, 5/2., 100), labels=["x", "x", "f(x)", "F(x)"])
        two_dim_plot([f_y, F_y], np.linspace(0, 5/3., 100), labels=["y", "y", "f(y)", "F(y)"])

    start = t.time()  # Tic
    u = np.random.uniform(0, 1, samples)
    v = np.random.uniform(0, 1, samples)
    x = np.zeros(samples)
    y = np.zeros(samples)

    # Vypocet v souladu s algoritmem ze cviceni
    for i in range(samples):
        SF = lambda x: F_x(x) - u[i]
        x[i] = fsolve(SF, 0.001)[0]  # fsolve() ze scipy je ekvivalent k Matlab fzero(); je nutno zadat pocatecni odhad
        fc_y = lambda y: f_join(x[i], y) / f_x(x[i])
        SFC = lambda y: (integrate.quad(fc_y, 0, y)[0] - v[i]) if 0 <= y <= (5/3. - (2/3.)*x[i]) else (1 if y > (5/3. - (2/3.)*x[i]) else 0)
        y[i] = fsolve(SFC, 0.001)[0]

    end = t.time()  # Toc
    if verbose:
        print("Time elapsed: {0:.3f} s".format(end - start))

    # Vykresleni histogramu, empirickych distribucnich funkci pro x a y a vykresleni vygenerovanych bodu (x, y)
    if plot:
        hist_plot(x, bins=int(samples/30), labels=["x", "f(x)", "x", "F(x)"])
        hist_plot(y, bins=int(samples/30), labels=["y", "f(y)", "y", "F(y)"])
        empirical_plot(F_x, np.linspace(0, 5/2., 100), x, labels=["x", "F(x)"])
        empirical_plot(F_y, np.linspace(0, 5/3., 100), y, labels=["y", "F(y)"])
        scatter_plot(x, y, method1=True)

    return x, y


def second_method(f_join, samples=1000, verbose=True, plot=True):
    """
    Druha metoda pro generovani nahodnych bodu ze cviceni (zamitaci metoda)
    Funkce generuje nahodne body na zaklade heuristiky, kde zname maximalni hodnotu f(x, y) a omezeni na oblast
    :param f_join: Zadana funkce f(x, y) (ocekavana lambda funkce)
    :param samples: Kolik bodu se ma vygenerovat (default 1000) (nevygeneruje se realne tolik; viz. ROA)
    :param verbose: True - vypisovat cas, roa, False - nevypisovat cas, roa (pouzito z duvodu testovani konvergence)
    :param plot: True - vykreslit grafy, False - nevykreslovat grafy (pouzito z duvodu testovani konvergence)
    :return: Vygenerovane body (dvojice x, y)
    """
    start = t.time()  # Tic
    x = np.zeros(0)
    y = np.zeros(0)

    max_f_value = 8/125. * (14 + 3 * np.sqrt(3))  # Maximalni hodnota f(x, y) v oblasti (zjisteno Wolframem Alpha)
    for i in range(samples):
        u = [5/2. * np.random.uniform(0, 1, 1), 5/3. * np.random.uniform(0, 1, 1), max_f_value * np.random.uniform(0, 1, 1)]
        found = np.where(u[0] + 2 * u[1] <= 4 and u[2] <= f_join(u[0], u[1]))[0]  # Podminka pro akceptaci bodu
        if len(found):
            x = np.append(x, u[0])
            y = np.append(y, u[1])

    end = t.time()  # Toc
    if verbose:
        print("Time elapsed: {0:.3f} s".format(end - start))

    # Vypocet rate of acceptance (pomer akceptovanych bodu k celkovemu poctu bodu)
    roa = len(x) / samples
    if verbose:
        print("Rate of acceptance: ", roa)

    # Vykresleni vygenerovanych bodu (x, y)
    if plot:
        scatter_plot(x, y)

    return x, y


def plot_concrete_convergence(samples, meassured, expected, method1, label="X", block=False):
    """
    Funkce vykresli zavislost mezi poctem vzorku a hodnotou dane charakteristiky
    :param samples: List s pocty vzorku (x-ova osa)
    :param meassured: Namerene hodnoty dane charakteristiky (y-ova osa)
    :param expected: Ocekavana hodnota dane charakteristiky (y-ova osa, zjisteno analyticky)
    :param method1: True - metoda 1, False - metoda 2 (pro titulek grafu)
    :param label: Label pro osu y (nazev charakteristiky)
    :param block: Blokovat vykresleni (True - blokovat, False - neblokovat)
    :return:
    """
    fig = plt.figure()
    if method1:
        fig.suptitle("Method 1: " + label + " convergence")
    else:
        fig.suptitle("Method 2: " + label + " convergence")
    plt.plot(samples, meassured, "x-", label="calculated " + label)
    plt.plot(samples, np.ones(len(samples)) * expected, label="expected " + label)
    plt.xlabel("# Samples")
    plt.ylabel(label)
    plt.grid()
    plt.legend()
    plt.show(block=block)


def plot_convergence(method, f_xy, method1=False):
    """
    Funkce pro vykresleni konvergence jednotlivych charakteristik (EX, VARX, EY, VARY, COVXY)
    Funkce prakticky pouze vola funkce pro generovani nahodnych bodu a pocita dane charakteristiky
    Nasledne vykresli grafy zavislosti dane charakteristiky na poctu generovanych bodu
    :param method: Funkce pro generovani nahodnych bodu (first_method, second_method)
    :param f_xy: funkce f(x, y) (ocekavana lambda funkce)
    :param method1: True - metoda 1, False - metoda 2 (kvuli nazvu grafu)
    :return:
    """
    samples = [  # Pocet generovanych bodu
        50, 100, 250, 500, 750, 1000, 1500, 2000, 2500, 3000, 5000, 7500, 10000, 15000, 20000, 25000, 50000, 100000
    ]
    if method1:  # Metoda 1 je pomalejsi, nechceme cekat dlouho, proto je omezen pocet bodu
        samples = samples[:8]
    EXs = []
    VARXs = []
    EYs = []
    VARYs = []
    COVXYs = []

    for sample in samples:
        x, y = method(f_xy, sample, verbose=False, plot=False)  # Generovani nahodnych bodu
        # Vypocet charakteristik
        EXs.append(np.mean(x))
        VARXs.append(np.var(x))
        EYs.append(np.mean(y))
        VARYs.append(np.var(y))
        COVXYs.append(np.cov(x, y, bias=True)[0][1])

    # Vykresleni konkretnich charakteristik v zavislosti na poctu generovanych bodu
    plot_concrete_convergence(samples, EXs, EX, method1, label="E(X)")
    plot_concrete_convergence(samples, VARXs, VARX, method1, label="Var(X)")
    plot_concrete_convergence(samples, EYs, EY, method1, label="E(Y)")
    plot_concrete_convergence(samples, VARYs, VARY, method1, label="Var(Y)")
    plot_concrete_convergence(samples, COVXYs, COVXY, method1, label="Cov(X,Y)", block=True)


if __name__ == '__main__':
    # Vypis ocekavanych hodnot (vypocitano analyticky na papir)
    print("Expected values:")
    print("E(X) = {0}".format(EX))
    print("Var(X) = {0}".format(VARX))
    print("E(Y) = {0}".format(EY))
    print("Var(Y) = {0}".format(VARY))
    print("Cov(X,Y) = {0}".format(COVXY))

    # Zvolena funkce hustoty pravdepodobnosti
    # f(x, y) = (48 / 125) * (x^2 * y + x * y + y) pro x >= 0, y >= 0, 2x + 3y <= 5
    f_xy = lambda x, y: (x >= 0 and y >= 0 and 2 * x + 3 * y <= 5) * (48 / 125.) * (x * x * y + x * y + y)

    # Generovani vzorku pomoci prvni ukazane metody na cviceni
    print("\nFirst method (1 000 samples):")
    x1, y1 = first_method(f_xy, samples=1000)  # Generuje 1 000 vzorku
    print_statistics(x1, y1)  # Vypis statistik ze vzorku (vyberovy prumer, vyberovy rozptyl, vyberova kovariance)

    # Generovani vzorku pomoci druhe ukazane metody na cviceni (zamitaci metoda)
    print("\nSecond method (10 000 samples):")
    x2, y2 = second_method(f_xy, samples=10000)  # Generuje "10 000" vzorku; rychlejsi, ale ne vsechny prijme (ROA)
    print_statistics(x2, y2)  # Vypis statistik ze vzorku (vyberovy prumer, vyberovy rozptyl, vyberova kovariance)

    # Vykresleni grafu konvergence jednotlivych vyberovych charakteristik v zavislosti na poctu vzorku
    print("\nCreating convergence plots for the first method...")
    plot_convergence(first_method, f_xy, method1=True)
    print("Creating convergence plots for the second method...")
    plot_convergence(second_method, f_xy)
    print("\nFinished!")
