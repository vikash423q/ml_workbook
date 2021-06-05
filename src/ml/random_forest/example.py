import numpy as np

from sklearn.datasets import load_breast_cancer, load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

from src.ml.random_forest.random_forest import RandomForest


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

    model = RandomForest(max_depth=10, num_trees=20)
    model.fit(x_train, y_train)

    y_hat = model.predict(x_test)
    acc = (y_hat == y_test).mean()
    print(f"Final accuracy : {acc}")


def start_with_sklearn():
    x_train, x_test, y_train, y_test = prep_data()

    model = RandomForestClassifier(max_depth=10, n_estimators=20)
    model.fit(x_train, y_train)
    y_hat = model.predict(x_test)
    acc = (y_hat == y_test).mean()
    print(f"Final accuracy : {acc}")


if __name__ == '__main__':
    start()
    start_with_sklearn()

    # output:
    # Final accuracy: 0.956140350877193
    # Final accuracy: 0.9649122807017544
