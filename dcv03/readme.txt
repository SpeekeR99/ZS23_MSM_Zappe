run.bat (resp. main.py) v defaultním režimu nepřebírá z příkazové řádky žádné argumenty
(defaultní režim = můj výběr tří sloupců, ke kterým jsem napsal komentáře, viz dále)
jinak je možné zadat programu tři argumenty, které reprezentují názvy vybraných sloupců
(např. run.bat MVAP RCHQ RCSG)

pokud máte Python verze 3.8 a vyšší, tak by program měl vykreslovat kruhové grafy
v kódu je zajištěná kontrola, a pokud máte verzi Pythonu 3.7 a nižší, tak se budou
vykreslovat pouze p-hodnoty v maticové heatmap podobě
(ta verze 3.8 je nutná kvůli knihovně pycirclize, která je použita pro vykreslování kruhových grafů)

snad bude fungovat 3D vizualizace, měla by si založit server(y) na localhostu na různých portech
a automaticky otevřít webový prohlížeč na daných adresách (stejné jako minulý úkol ode mě)

adresar mimo jine obsahuje komentare.txt, tam se nachazi slovni komentar
k deskriptivni statistice, naslednym testum a jejich vysledkum + vizualizacim