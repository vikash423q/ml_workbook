import numpy as np

from sklearn.naive_bayes import GaussianNB as SKGaussianNb, MultinomialNB as SKMultinomialNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier as SKDecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression as SKLogisticRegression
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.datasets import *
from matplotlib import pyplot as plt

from src.ml.knn.knn import KNNClassifier
from src.ml.decision_tree.decision_tree import DecisionTreeClassifier
from src.ml.naive_bayes.naive_bayes import GaussianNB, MultiNomialNB
from src.ml.logistic_regression.logistic_regression import LogisticRegresssion
from src.ml.random_forest.random_forest import RandomForest
from src.ml.svm.svm import SVM


def prep_data(normalise: bool = False):
    data = load_wine()
    x = data['data']
    if normalise:
        mean = np.mean(x, axis=0)
        std = np.std(x, axis=0)
        x = (x - mean) / std
    y = data['target']
    y = y.reshape(y.shape[0], 1)
    return train_test_split(x, y, test_size=0.2, random_state=21)


def predict_with_classifiers(normalise_input: bool = True):
    x_train, x_test, y_train, y_test = prep_data(normalise=normalise_input)
    y_train_raveled = y_train.reshape(-1)
    y_test_raveled = y_test.reshape(-1)

    labels, score, sk_score = [], [], []

    # logistic regression
    labels.append('Logistic Reg')
    model = LogisticRegresssion()
    model.fit(x_train, y_train, x_test, y_test, epochs=1000, lr=0.01, l1_ratio=None,
              batch_size=64,
              output_dir="../../../temp/breast_cancer/")
    y_hat = model.predict(x_test)
    score.append((y_hat == y_test).mean())

    model = SKLogisticRegression(max_iter=1000)
    model.fit(x_train, y_train)
    y_hat = model.predict(x_test)
    sk_score.append((y_hat == y_test).mean())

    # decision tree
    labels.append('Dec. Tree')
    model = SKDecisionTreeClassifier(max_depth=10, criterion='entropy')
    model.fit(x_train, y_train_raveled)

    y_hat = model.predict(x_test)
    score.append((y_hat == y_test_raveled).mean())

    model = DecisionTreeClassifier(max_depth=10, method='entropy')
    model.fit(x_train, y_train_raveled)
    y_hat = model.predict(x_test)
    sk_score.append((y_hat == y_test_raveled).mean())

    # naive bayes
    labels.append('Gaussian NB')
    model = GaussianNB()
    model.fit(x_train, y_train_raveled)
    y_hat = model.predict(x_test)
    score.append((y_hat == y_test_raveled).mean())

    model = SKGaussianNb()
    model.fit(x_train, y_train_raveled)
    y_hat = model.predict(x_test)
    sk_score.append((y_hat == y_test_raveled).mean())

    # Knn
    labels.append('KNN')
    model = KNNClassifier(n_neighbours=10)
    model.fit(x_train, y_train_raveled)
    y_hat = model.predict(x_test)
    score.append((y_hat == y_test_raveled).mean())

    model = KNeighborsClassifier(n_neighbors=10)
    model.fit(x_train, y_train_raveled)
    y_hat = model.predict(x_test)
    sk_score.append((y_hat == y_test_raveled).mean())

    # SVM
    labels.append('SVM')
    model = SVM(lr=0.001, lambda_param=0.001, num_steps=1000, verbose=False)
    model.fit(x_train, y_train)
    y_hat = model.predict(x_test)
    score.append((y_hat == y_test).mean())

    model = SVC(C=1, max_iter=1000)
    model.fit(x_train, y_train)
    y_hat = model.predict(x_test)
    sk_score.append((y_hat == y_test).mean())

    # Random Forest
    labels.append('Rand Forest')
    model = RandomForest(max_depth=10, num_trees=30)
    model.fit(x_train, y_train_raveled)
    y_hat = model.predict(x_test)
    score.append((y_hat == y_test_raveled).mean())

    model = RandomForestClassifier(max_depth=10, n_estimators=30)
    model.fit(x_train, y_train_raveled)
    y_hat = model.predict(x_test)
    sk_score.append((y_hat == y_test_raveled).mean())

    from utils import dump_json
    dump_json('../scribbles/accuracies.json', dict(labels=labels,
                                                   score=score,
                                                   sk_score=sk_score))

    return labels, score, sk_score


def plot_accuracy_bar(labels, score1, score2):
    width = 0.5
    plt.bar(np.arange(len(score1))*2, score1, width=width)
    plt.bar(np.arange(len(score2))*2 + width, score2, width=width)
    plt.xticks(np.arange(len(score1))*2, labels)

    colors = {'scratch': 'tab:blue', 'sklearn': 'tab:orange'}
    labels = list(colors.keys())
    handles = [plt.Rectangle((0, 0), 1, 1, color=colors[label]) for label in labels]
    plt.legend(handles, labels)

    plt.xlabel('Classifiers')
    plt.ylabel('Accuracy')

    plt.title('SkLearn Wine dataset')
    plt.savefig('wine_accuracy.png')
    plt.close()


if __name__ == '__main__':
    from utils import load_json
    # d =load_json('accuracies.json')
    # l, s1, s2 = d['labels'], d['score'], d['sk_score']
    # plot_accuracy_bar(l, s1, s2)

    l, s1, s2 = predict_with_classifiers(normalise_input=True)
    plot_accuracy_bar(l, s1, s2)