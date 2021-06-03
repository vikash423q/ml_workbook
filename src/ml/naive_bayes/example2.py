from sklearn.datasets import *
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from ml.naive_bayes.naive_bayes import MultiNomialNB as MultiNomialNBScratch

from utils import read_csv
import numpy as np
import pandas as pd


def prep_data():
    data = load_breast_cancer()
    feature_names = data['feature_names']
    x = data['data']
    y = data['target']
    return train_test_split(x, y, test_size=0.2)


def prep_spam_data():
    data = pd.read_csv("~/Downloads/spam_or_not_spam.csv")
    data = np.array(data.values)
    x_raw, y = data[:, 0], data[:, 1]
    y = np.array(y, dtype='float32')
    x_raw[2966] = 'null'

    tokens = set([word for seq in x_raw for word in seq.split()])
    idx_token_map = {tok: i for i, tok in enumerate(list(tokens))}
    x = np.zeros((x_raw.shape[0], len(tokens)))
    for idx, seq in enumerate(x_raw):
        feature_count = {}
        for word in seq.split():
            feature_count.setdefault(word, 0)
            feature_count[word] += 1
        for w, count in feature_count.items():
            x[idx, idx_token_map[w]] = count

    return train_test_split(x, y, test_size=0.2)


def start():
    x_train, x_test, y_train, y_test = prep_spam_data()
    model = MultiNomialNBScratch()
    model.fit(x_train, y_train)
    y_hat = model.predict(x_test)
    acc = (y_hat == y_test).mean()
    print(f"Final accuracy : {acc}")


def start_with_sklearn():
    x_train, x_test, y_train, y_test = prep_spam_data()
    model = MultinomialNB()
    model.fit(x_train, y_train)
    y_hat = model.predict(x_test)
    acc = (y_hat == y_test).mean()
    print(f"Final accuracy : {acc}")


if __name__ == '__main__':
    start()
    start_with_sklearn()

    # output
    # Final accuracy: 0.99
    # Final accuracy: 0.99
