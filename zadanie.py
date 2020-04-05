#!/usr/bin/python3

import os
import pprint
import webbrowser

from sklearn import linear_model
from sklearn import metrics
from sklearn.feature_extraction import text

PARAMS = {
    # TU(6): Wybrać zbiór danych.
    'dataset': 'reviews.text',
    # TU(9): Próbować różnych wartości.
    'ngram_range': (9, 10),
    # TU(9): Próbować różnych wartości.
    'penalty': 'l1',
    'solver': 'liblinear',
}

TRAIN_DATASET = f'dataset/{PARAMS["dataset"]}.train.txt'
TEST_DATASET = f'dataset/{PARAMS["dataset"]}.test.txt'
DEV_DATASET = f'dataset/{PARAMS["dataset"]}.dev.txt'

LABEL_TO_Y = {
    '__label__meta_minus_m': 0,
    '__label__meta_minus_s': 0,
    '__label__meta_amb': None,
    '__label__meta_zero': None,
    '__label__meta_plus_s': 1,
    '__label__meta_plus_m': 1,

    '__label__z_minus_m': 0,
    '__label__z_minus_s': 0,
    '__label__z_amb': None,
    '__label__z_zero': None,
    '__label__z_plus_s': 1,
    '__label__z_plus_m': 1,
}

Y_TO_CLASS = {
    0: 'negative',
    1: 'positive',
}

HTML_FILE = 'misclassifications.html'

# TU(8): Studenci-daltoniści proszeni są o zmianę
# poniższych wartości tak, by móc odróżnić podświetlenia
# wyrazów o wydźwięku dodatnim i ujemnym.
GREEN = [0x85, 0x99, 0x00]
RED = [0xdc, 0x32, 0x2f]


NEGATION_START = {
    # TU(11): Wpisać wyrazy, które zmieniają
    # wydźwięk swoich następników na przeciwny.
    'nie', 'bez', 'oprócz', 'prócz', 'poza', 'brak', 'źle'
}


def starts_with_negation(token):
    return token.startswith('nie') or token.startswith('bez')


def read_file(filename):
    with open(filename, 'rt', encoding='UTF-8') as file:
        for line in file:
            tokens = line.lower().split()
            y = LABEL_TO_Y[tokens[-1]]
            if y is not None:
                yield tokens[:-1], y


def preprocess_tokens(tokens):
    return tokens
    # TU(11): Usunąć powyższy wiersz i zaimplementować
    # wstępne przetwarzanie tokenów zgodnie z instrukcją.
    negate = False
    result = []
    return result


def read_from_file(filename, X, Y):
    for tokens, y in read_file(filename):
        X.append(' '.join(preprocess_tokens(tokens)))
        Y.append(y)


def print_weights(header, token_weights):
    print(header, ', '.join(t.replace(' ', '#') for w, t in token_weights))


def get_token_weights(count_vectorizer, model):
    token_weights = []
    for token, index in count_vectorizer.vocabulary_.items():
        v = model.coef_[0][index]
        if v:
            token_weights.append((v, token))
    token_weights.sort()
    return token_weights


def print_token_info(token_weights):
    print('{} N-gramów o niezerowych wagach'.format(len(token_weights)))
    print_weights('Dodatnie:', token_weights[-1:-20:-1])
    print_weights('Ujemne:', token_weights[:20])


def print_report(X_test, Y_test, count_vectorizer, model):
    X_counted = count_vectorizer.transform(X_test)
    print(metrics.classification_report(
        Y_test, model.predict(X_counted), digits=3))


def show_misclassifications_in_browser(
        X_test, Y_test, count_vectorizer, model):
    X_counted = count_vectorizer.transform(X_test)
    Y_predicted = model.predict(X_counted)
    html = open(HTML_FILE, 'wt')
    html.write('<html><body>\n')
    base_proba = model.predict_proba(count_vectorizer.transform(['']))[0][1]
    for x, y_real, y_predicted in zip(X_test, Y_test, Y_predicted):
        if y_real != y_predicted:
            html.write(f"""<p><b>Actual: {Y_TO_CLASS[not y_predicted]},
predicted: {Y_TO_CLASS[y_predicted]}</b><br>\n""")
            for word in x.split():
                word_counted = count_vectorizer.transform([word])
                word_score = model.predict_proba(word_counted)[0][1]
                if word_score < base_proba:
                    m = (base_proba - word_score) / base_proba
                    red = [255 - int((255 - c) * m) for c in RED]
                    rgb = ','.join(str(c) for c in red)
                    html.write(f'  <span title="{word_score:.2}" \
style="background-color:rgb({rgb});">{word}</span>\n')
                else:
                    m = (word_score - base_proba) / (1.0 - base_proba)
                    green = [255 - int((255 - c) * m) for c in GREEN]
                    rgb = ','.join(str(c) for c in green)
                    html.write(f'  <span title="{word_score:.2}" \
style="background-color:rgb({rgb});">{word}</span>\n')
            html.write('</p>\n')
    html.write('</body></html>')
    html.close()
    webbrowser.open_new_tab(f'file://{os.path.abspath(HTML_FILE)}')


def main():
    count_vectorizer = text.CountVectorizer(
        analyzer='char_wb',
        lowercase=False,
        ngram_range=PARAMS['ngram_range'])
    X_train = []
    Y_train = []
    read_from_file(TRAIN_DATASET, X_train, Y_train)
    X_train = count_vectorizer.fit_transform(X_train)
    model = linear_model.LogisticRegression(
        penalty=PARAMS['penalty'],
        solver=PARAMS['solver'],
        max_iter=1000,
        verbose=True
    )
    model.fit(X_train, Y_train)

    print('\n\n')
    pprint.pprint(PARAMS)
    token_weights = get_token_weights(count_vectorizer, model)
    print_token_info(token_weights)
    X_test = []
    Y_test = []
    read_from_file(TEST_DATASET, X_test, Y_test)
    read_from_file(DEV_DATASET, X_test, Y_test)
    print_report(X_test, Y_test, count_vectorizer, model)
    show_misclassifications_in_browser(
        X_test, Y_test, count_vectorizer, model)


if __name__ == '__main__':
    main()
