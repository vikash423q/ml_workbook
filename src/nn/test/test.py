import os
import numpy as np
import unittest

from src.utils import load_json
from src.nn.single_layer_train_utils import forward_propagation, calculate_loss, \
    backward_propagation, training, predict
from src.nn import deep_train_utils


class TestForwardPropagation(unittest.TestCase):
    def setUp(self) -> None:
        self.data = load_json(os.path.join(os.path.dirname(__file__),
                                           "../../../temp/test_data/forward_propagation.json"))

    def test_function(self):
        params = dict(W=[self.data['parameters']['W1'], self.data['parameters']['W2']],
                      B=[self.data['parameters']['b1'], self.data['parameters']['b2']])
        res = forward_propagation(X=self.data['X_assess'],
                                  parameters=params)

        m = (np.mean(res['Z1']), np.mean(res['A1']), np.mean(res['Z2']), np.mean(res['A2']))
        for m_, r_ in zip(m, self.data['result']):
            self.assertAlmostEqual(m_, r_)
        print(res)


class TestComputeCost(unittest.TestCase):
    def setUp(self) -> None:
        self.data = load_json(os.path.join(os.path.dirname(__file__),
                                           "../../../temp/test_data/compute_cost.json"))

    def test_function(self):
        res = calculate_loss(np.array(self.data['A2']), np.array(self.data['Y_assess']))
        print(res)
        self.assertAlmostEqual(res, self.data['cost'])


class TestBackPropagation(unittest.TestCase):
    def setUp(self) -> None:
        self.data = load_json(os.path.join(os.path.dirname(__file__),
                                           "../../../temp/test_data/backward_propagation.json"))

    def test_function(self):
        self.data['cache'] = {k: np.array(v) for k, v in self.data['cache'].items()}
        self.data['parameters'] = {k: np.array(v) for k, v in self.data['parameters'].items()}

        params = dict(W=[self.data['parameters']['W1'], self.data['parameters']['W2']],
                      B=[self.data['parameters']['b1'], self.data['parameters']['b2']])

        res = backward_propagation(self.data['cache'], np.array(self.data['X_assess']),
                                   np.array(self.data['Y_assess']), params)
        print(res, "\n")

    def test_deep_function(self):
        self.data['cache'] = {k: np.array(v) for k, v in self.data['cache'].items()}
        self.data['parameters'] = {k: np.array(v) for k, v in self.data['parameters'].items()}
        self.data['cache']['A0'] = np.array(self.data['X_assess'])

        params = dict(W1=self.data['parameters']['W1'], W2=self.data['parameters']['W2'],
                      b1=self.data['parameters']['b1'], b2=self.data['parameters']['b2'])

        res = deep_train_utils.backward_propagation(Y=np.array(self.data['Y_assess']),
                                                    A=self.data['cache']['A2'],
                                                    parameters=params,
                                                    cache=self.data['cache'],
                                                    layer_activations=['tanh', 'sigmoid'],
                                                    num_layers=2)
        print(res, "\n")


class TestNNTraining(unittest.TestCase):
    def setUp(self) -> None:
        self.data = load_json(os.path.join(os.path.dirname(__file__),
                                           "../../../temp/test_data/nn_model.json"))

        self.train_x_data = load_json(os.path.join(os.path.dirname(__file__),
                                                   "../../../temp/datasets/planar/X.json"))
        self.train_y_data = load_json(os.path.join(os.path.dirname(__file__),
                                                   "../../../temp/datasets/planar/Y.json"))

    def test_function(self):
        res = training(np.array(self.data['X_assess']), np.array(self.data['Y_assess']))
        print(res)

    def test_planar_train(self):
        res = training(np.array(self.train_x_data), np.array(self.train_y_data))
        print(res)

    def test_planar_with_deep(self):
        res = deep_train_utils.training(np.array(self.train_x_data), np.array(self.train_y_data),
                                        layer_dims=[4, 1],
                                        layer_activations=['tanh', 'sigmoid'],
                                        epochs=10000,
                                        learning_rate=1.2)
        print(res)


class TestPrediction(unittest.TestCase):
    def setUp(self) -> None:
        self.data = load_json(os.path.join(os.path.dirname(__file__),
                                           "../../../temp/test_data/predict.json"))

        self.train_x_data = load_json(os.path.join(os.path.dirname(__file__),
                                                   "../../../temp/datasets/planar/X.json"))
        self.train_y_data = load_json(os.path.join(os.path.dirname(__file__),
                                                   "../../../temp/datasets/planar/Y.json"))

    def test_function(self):
        self.data['parameters'] = {k: np.array(v) for k, v in self.data['parameters'].items()}
        print(self.data['parameters'])

        params = dict(W=[self.data['parameters']['W1'], self.data['parameters']['W2']],
                      B=[self.data['parameters']['b1'], self.data['parameters']['b2']])
        res = predict(np.array(self.data['X_assess']), params)
        self.assertAlmostEqual(np.mean(res), self.data['mean'])

    def test_deep(self):
        self.data['parameters'] = deep_train_utils.training(np.array(self.train_x_data), np.array(self.train_y_data),
                                                            layer_dims=[4, 1],
                                                            layer_activations=['tanh', 'sigmoid'],
                                                            epochs=10000,
                                                            learning_rate=1.2)
        print(self.data['parameters'])

        params = dict(W=[self.data['parameters']['W1'], self.data['parameters']['W2']],
                      B=[self.data['parameters']['b1'], self.data['parameters']['b2']])
        print(params)
        res = predict(np.array(self.data['X_assess']), params)
        self.assertAlmostEqual(np.mean(res), self.data['mean'])


if __name__ == '__main__':
    unittest.main()
