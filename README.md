# Przetwarzanie języka naturalnego
# Laboratorium 4: analiza wydźwięku

1. Skopiować zawartość niniejszego repozytorium
na dysk lokalny. Pobrać korpus opinii `dataset_clarin.zip`
z linku na dole strony https://clarin-pl.eu/dspace/handle/11321/700
i rozpakować go.

    ```
    git clone https://github.com/PK-PJN-NS/laboratorium-4.git
    cd laboratorium-4
    unzip dataset_clarin.zip
    ```

2. Zainstalować `scikit-learn` — bibliotekę
z narzędziami do uczenia maszynowego.

    ```
    pip install sklearn
    ```

3. Dzisiejsze zadanie dotyczy analizy
wydźwięku (*sentiment analysis*), czyli
klasyfikowania opinii jako pozytywne
lub negatywne zależnie od ich treści.
Zostanie ono rozwiązane za pomocą
*regresji logistycznej*. W bibliotece
`scikit-learn` odpowiada za nią klasa
[`linear_model.LogisticRegression`](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html).

4. Skrypt `zadanie.py` działa następująco:

    * Wczytuje zbiór uczący
    (`dataset/*.train.txt`).

    * Zamienia znaki w każdym wierszu
    zbioru uczącego na małe.

    * Twórcy zbiorów danych z CLARIN-PL
    zadbali o tokenizację wierszy,
    więc się nią nia przejmujemy.

    * Zgodnie z książkami powinno się
    usunąć z wierszy zbioru uczącego
    wyrazy funkcyjne (*stop words*),
    np. *i*, *do*, *przez* oraz przeprowadzić
    *stemming* pozostałych wyrazów,
    czyli heurystycznie obciąć ich końcówki.
    My zamiast tego użyjemy *N-gramów
    znakowych* (*character N-grams*,
    dla odróżnienia od *word N-grams*,
    czyli *N-gramów wyrazowych*).
    Są to *N*-tki kolejnych znaków
    składających się na tokeny,
    do których z przodu i z tyłu
    dodano *wartowników*. Na przykład tokenowi
    `'polecam'` odpowiadają następujące
    5-gramy: `'#pole'`, `'polec'`, `'oleca'`,
    `'lecam'` i `'ecam#'`.
    Robimy tak, ponieważ:

        * dla języka polskiego nie ma dobrych
        stemmerów, a *lematyzatory* (np. plik
        `polski.sqlite3` z laboratorium 1)
        są duże i nieporęczne,

        * regresja logistyczna sama
        sobie wybierze ważne *N*-gramy.

    * Skrypt dodaje do zmiennej `X_train`
    kolejne wiersze, a do zmiennej
    `Y_train` — wartości 0 (opinia
    negatywna) lub 1 (opinia pozytywna).

    * Buduje model regresji logistycznej
    na podstawie zmiennych `X_train`
    i `Y_train`.

    * Wyświetla parametry modelu.

    * Wyświetla 20 *N*-gramów, które
    mają najbardziej dodatni wpływ
    na wynik regresji logistycznej,
    i 20 *N*-gramów, które mają
    na niego najbardziej ujemny wpływ.

    * Wyświetla raport o jakości
    klasyfikacji zbioru testowego
    (`dataset/*.test.txt + dataset/*.dev.txt`)
    za pomocą uzyskanego modelu.

    * Otwiera w przeglądarce stronę,
    która zawiera źle sklasyfikowane
    wiersze zbioru testowego
    (`dataset/*.test.txt + dataset/*.dev.txt`)
    z podświetleniem tych tokenów,
    które wpływają dodatnio
    lub ujemnie na wynik klasyfikacji.

5. Wybrać jeden ze zbiorów danych
z folderu `dataset`, np.:

    * `hotels.sentence.*.txt` — opinie o hotelach podzielone na zdania;

    * `medicine.sentence.*.txt` — opinie o lekarzach podzielone na zdania;

    * `products.text.*.txt` — opinie o zakupach — całe wypowiedzi;

    * `reviews.text.*.txt` — opinie o nauczycielach akademickich
    — całe wypowiedzi.

6. W pliku `zadanie.py` przypisać stałej
`DATASET` jedną z wartości `'hotels.sentence'`,
`'medicine.sentence'`, `'products.text'`,
`'reviews.text'` lub inną, odpowiadającą
wybranemu zbiorowi danych.

7. Uruchomić skrypt `zadanie.py`.
Skopiować do sprawozdania to, co zostanie
wypisane na ekranie, poczynając od
`{'dataset':`.

8. W razie kłopotów z odróżnieniem
w przeglądarce podświetlenia wyrazów
na zielono i na czerwono odszukać w pliku
`zadanie.py` stałe `GREEN` i `RED`
i je zmienić.

9. Zmieniając parametry `ngram_range`
i `penalty`, próbować znaleźć jak najlepszy
model. Za miarę jakości modelu przyjąć
wartość na przecięciu wiersza
`weighted avg` i kolumny `f1-score`
(im większa liczba, tym lepiej).
Uwagi:

    * Parametr `ngram_range` odpowiada
    za długości generowanych *N*-gramów.
    Przykładowe wartości do wypróbowania:
    `(4, 4)`, `(6, 6)`, `(7, 7)`, `(3, 7)`,
    `(100, 100)`. Ostatnia wartość oznacza:
    wcale nie dziel tokenów.

    * Wynikiem uczenia modelu zgodnie
    z regresją logistyczną jest słownik,
    który przypisuje wagi do *N*-gramów,
    oraz *wyraz wolny* (*bias*).
    Po zastosowaniu modelu do *N*-gramów,
    składających się na jakiś tekst,
    otrzymuje się sumę wag tych *N*-gramów
    i wyrazu wolnego. Tę sumę przekształca
    się na liczbę z przedziału (0, 1)
    zgodnie z [funkcją logistyczną](https://en.wikipedia.org/wiki/Logistic_function).
    Jeśli wynik tego przekształcenia
    jest większy niż 1/2 (co zachodzi,
    gdy suma jest dodatnia), model przewiduje,
    że tekst ma wydźwięk dodatni.
    Jeśli wynik tego przekształcenia
    jest mniejszy niż 1/2 (co zachodzi,
    gdy suma jest ujemna), model przewiduje,
    że tekst ma wydźwięk ujemny.

    * Parametr `penalty` może przyjmować
    wartości `l1` lub `l2`. Odpowiada im
    regresja logistyczna z regularyzacją L1
    (zwana *lasso regression*) i regresja
    logistyczna z regularyzacją L2 (zwana
    *ridge regression*). Modele otrzymane
    z regularyzacją L1 są *rzadkie*, czyli
    nie wszystkim *N*-gramom występującym
    w danych uczących przypisują niezerowe wagi.
    Modele otrzymane z regularyzacją L2
    są *gęste*: przypisują niezerowe wagi
    wszystkim *N*-gramom występującym w danych
    uczących.

10. Skopiować do sprawozdania dane
najlepszego uzyskanego modelu,
poczynając od `{'dataset':`.

11. Powyższe podejście jest raczej
prymitywne. Nie bierze ono pod uwagę
*polaryzacji* wyrazów, czyli tego,
że np. fragment 'nie polecam' ma wydźwięk
przeciwny do fragmentu 'polecam'.
Żeby temu zaradzić, proszę wykonać
następujące polecenia:

    * Dopisać do zbioru `NEGATION_START`
    wyrazy, które zmieniają wydźwięk
    swoich następników na przeciwny.
    Przykładowe wyrazy: `'nie'`, `'bez'`,
    `'oprócz'`, `'prócz'`, `'poza'`, `'brak'`,
    `'źle'`. Najlepiej byłoby dobrać ten zbiór
    tak, by zmaksymalizować jakość
    modeli. Nie przejmujemy się tym,
    że niektóre z tych wyrazów mają też
    znaczenia niezwiązane z negacją
    ('przez nie' itp.)

    * Zmodyfikować funkcję `preprocess_tokens()`
    tak, by zamieniała wyrazy o przeciwnym
    wydźwięku na wielkie litery. Konkretnie
    należy iterować zmienną `token` po liście
    `tokens` i wykonywać poniższe kroki:

        * jeśli zmienna `negate` jest fałszywa
        i `starts_with_negation(token)` jest fałszywe,
        to dodać `token` do listy `result`;

        * jeśli zmienna `negate` jest fałszywa,
        a `starts_with_negation(token)` jest prawdziwe,
        to dodać `token.upper()` do listy `result`;

        * jeśli zmienna `negate` jest prawdziwa,
        a `starts_with_negation(token)` jest fałszywe,
        to dodać `token.upper()` do listy `result`;

        * jeśli zmienna `negate` jest prawdziwa
        i `starts_with_negation(token)` jest prawdziwe,
        to dodać `token` do listy `result`;

        * jeśli `token` należy do zbioru `NEGATION_START`,
        to przypisać zmiennej `negate` wartość `True`;

        * jeśli `token` jest znakiem przestankowym,
        czyli `token.isalnum()` jest fałszywe,
        to przypisać zmiennej `negate` wartość `False`
        — dzięki tej heurystyce przeciwna polaryzacja
        wyrazów w liście `result` kończy się zazwyczaj
        w tym samym miejscu, co w rzeczywistości.

    * Skopiować do sprawozdania wartość
    stałej `NEGATION_START` i treść funkcji
    `preprocess_tokens()`.

12. Zmieniając parametry `ngram_range`
i `penalty`, próbować znaleźć jak najlepszy model.
Skopiować do sprawozdania dane
najlepszego uzyskanego modelu,
poczynając od `{'dataset':`.

13. Zadanie nadobowiązkowe: przyjrzeć
się kilku źle sklasyfikowanym opiniom,
wyświetlonym w przeglądarce, i jeśli
przyjdą nam do głowy jakieś wnioski
na ich temat, dopisać je do sprawozdania.

14. Zadanie nadobowiązkowe: spróbować
innych metod klasyfikacji, np.
[*Support Vector Classification*](https://scikit-learn.org/stable/modules/generated/sklearn.svm.LinearSVC.html).
Należy przy tym zakomentować wywołanie funkcji
`show_misclassifications_in_browser()`,
bo w `svm.LinearSVC` nie ma metody
`.predict_proba()`.
Skopiować do sprawozdania to,
co zostanie wypisane na ekranie,
poczynając od liczby *N*-gramów
o niezerowych wagach.
