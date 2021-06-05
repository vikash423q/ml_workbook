import numpy as np

from sklearn.datasets import load_breast_cancer, load_iris
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

from src.ml.decision_tree.decision_tree import DecisionTreeClassifier as DecisionTreeClassifierScratch


def prep_data():
    data = load_breast_cancer()

    feature_names = data['feature_names']
    x = data['data']
    y = data['target']
    x = x / np.max(x)
    y = y
    return train_test_split(x, y, test_size=0.2, random_state=23)


def start():
    x_train, x_test, y_train, y_test = prep_data()

    model = DecisionTreeClassifierScratch(max_depth=10, max_feature_split=10, method='entropy')
    model.fit(x_train, y_train)

    y_hat = model.predict(x_test)
    acc = (y_hat == y_test).mean()
    print(f"Final accuracy : {acc}")


def start_with_sklearn():
    x_train, x_test, y_train, y_test = prep_data()

    model = DecisionTreeClassifier(max_depth=10)
    model.fit(x_train, y_train)
    y_hat = model.predict(x_test)
    acc = (y_hat == y_test).mean()
    print(f"Final accuracy : {acc}")


if __name__ == '__main__':
    start()
    start_with_sklearn()

    # output:
    # Final accuracy: 0.9385964912280702
    # Final accuracy: 0.9122807017543859
