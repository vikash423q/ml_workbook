from sklearn.datasets import *
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from ml.naive_bayes.naive_bayes import GaussianNB as GaussianNBScratch


def prep_data():
    data = load_svmlight_file()
    feature_names = data['feature_names']
    x = data['data']
    y = data['target']
    return train_test_split(x, y, test_size=0.2, random_state=21)


def start():
    x_train, x_test, y_train, y_test = prep_data()

    model = GaussianNBScratch()
    model.fit(x_train, y_train)
    y_hat = model.predict(x_test)
    acc = (y_hat == y_test).mean()
    print(f"Final accuracy : {acc}")


def start_with_sklearn():
    x_train, x_test, y_train, y_test = prep_data()

    model = GaussianNB()
    model.fit(x_train, y_train)
    y_hat = model.predict(x_test)
    acc = (y_hat == y_test).mean()
    print(f"Final accuracy : {acc}")


if __name__ == '__main__':
    start()
    start_with_sklearn()

    # output
    # Final accuracy: 0.9666666666666667
    # Final accuracy: 0.966666666666666
