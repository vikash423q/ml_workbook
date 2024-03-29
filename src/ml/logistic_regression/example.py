from sklearn.datasets import load_breast_cancer, load_iris
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC, SVR
from ml.logistic_regression.logistic_regression import LogisticRegresssion as LogisticRegressionScratch


def prep_data():
    data = load_iris()

    feature_names = data['feature_names']
    x = data['data']
    y = data['target']
    y = y.reshape(y.shape[0], 1)
    return train_test_split(x, y, test_size=0.2)


def start():
    x_train, x_test, y_train, y_test = prep_data()

    model = LogisticRegressionScratch()
    model.fit(x_train, y_train, x_test, y_test, epochs=5000, lr=0.01, l1_ratio=None,
              batch_size=64,
              output_dir="../../../temp/breast_cancer/")
    y_hat = model.predict(x_test)
    acc = (y_hat == y_test).mean()
    print(f"Final accuracy : {acc}")


def start_with_sklearn():
    x_train, x_test, y_train, y_test = prep_data()

    model = LogisticRegression(max_iter=10000)
    model.fit(x_train, y_train)
    y_hat = model.predict(x_test)
    acc = (y_hat == y_test).mean()
    print(f"Final accuracy : {acc}")


if __name__ == '__main__':
    start()