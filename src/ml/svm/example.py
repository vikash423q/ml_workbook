from sklearn.datasets import *
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC, SVR
from ml.svm.svm import SVM


def prep_data():
    data = load_breast_cancer()

    feature_names = data['feature_names']
    x = data['data']
    y = data['target']
    y = y.reshape(y.shape[0], 1)
    return train_test_split(x, y, test_size=0.2, random_state=21)


def start():
    x_train, x_test, y_train, y_test = prep_data()
    y_test[y_test <= 0] = -1

    model = SVM(lr=0.001, lambda_param=0.001, num_steps=1000, verbose=False)
    model.fit(x_train, y_train)
    y_hat = model.predict(x_test)
    acc = (y_hat == y_test).mean()
    print(f"Final accuracy : {acc}")


def start_with_sklearn():
    x_train, x_test, y_train, y_test = prep_data()
    y_train[y_train <= 0] = -1
    y_test[y_test <= 0] = -1

    model = SVC(C=1, kernel='rbf', max_iter=10000)
    model.fit(x_train, y_train)
    y_hat = model.predict(x_test)
    acc = (y_hat == y_test).mean()
    print(f"Final accuracy : {acc}")


if __name__ == '__main__':
    start()
    start_with_sklearn()