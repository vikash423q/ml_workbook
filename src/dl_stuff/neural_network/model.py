import os
import time
import datetime

from sklearn.model_selection import train_test_split

from src.dl_stuff.neural_network.cost import *
from src.dl_stuff.neural_network.prediction import *
from src.dl_stuff.neural_network.initialization import *
from src.dl_stuff.neural_network.update_parameters import *
from src.dl_stuff.neural_network.forward_propagation import *
from src.dl_stuff.neural_network.backward_propgation import *
from src.dl_stuff.evaluation.draw import plot
from src.utils import dump_pickle, dump_json


def fit(X: np.ndarray, Y: np.ndarray, num_layers: List[int], layer_activations: List[str], epochs: int,
        output_path: str,
        learning_rate: float = 0.01,
        dropout: float = None,
        mini_batch: int = None,
        parameters: dict = None,
        regularization: str = None,
        lamda: float = 0.001,
        optimizer: str = None,
        betav: float = 0.9,
        betas: float = 0.9,
        batch_norm: bool = False,
        X_test=None, Y_test=None,
        tag: str = 'train'):
    assert X.shape[1] == Y.shape[1], f"Input data {X.shape} and labeled data {Y.shape} shapes are invalid"

    n_x = X.shape[0]

    print(X.shape, Y.shape)
    X_train, X_valid, Y_train, Y_valid = train_test_split(X.T, Y.T, train_size=0.8, shuffle=True, random_state=21)

    X_train = X_train.T
    X_valid = X_valid.T
    Y_train = Y_train.T
    Y_valid = Y_valid.T
    m = X_train.shape[1]
    print(f"X train data: {X_train.shape}, X validation data: {X_valid.shape}")
    print(f"Y train data: {Y_train.shape}, Y validation data: {Y_valid.shape}")

    if not parameters:
        parameters = initialize_parameters(n_x, num_layers, batch_norm)

    if optimizer == 'momentum':
        v = initialize_velocity(parameters, num_layers)
    elif optimizer == 'rms':
        s = initialize_rmsprop(parameters, num_layers)
    elif optimizer == 'Adam':
        t = 0
        v, s = initialize_adam(parameters, num_layers)

    wd = os.path.join(output_path, str(datetime.datetime.now()) + '_' + tag)
    os.makedirs(wd, exist_ok=True)
    os.makedirs(os.path.join(wd, 'backup'), exist_ok=True)

    start_time = time.time()

    meta = dict(tag=tag, num_layers=num_layers, learning_rate=learning_rate,
                epochs=[], train_loss=[], test_loss=[], valid_acc=[], train_acc=[])
    if X_test is not None:
        meta['test_acc'] = []

    if mini_batch is None:
        mini_batch = m

    n_batches = math.ceil(m / mini_batch)
    print(n_batches, m)
    for i in range(epochs):
        epoch = i + 1

        cost = None
        for j in range(n_batches):
            X_batch = X_train[:, j * mini_batch:(j + 1) * mini_batch]
            Y_batch = Y_train[:, j * mini_batch:(j + 1) * mini_batch]

            A, caches = forward_propagation(X_batch, parameters, num_layers, layer_activations, dropout, batch_norm)

            if regularization == 'l1':
                cost = compute_cost_with_l1_regularization(A, Y_batch, lamda, parameters, num_layers, layer_activations[-1])
            elif regularization == 'l2':
                cost = compute_cost_with_l2_regularization(A, Y_batch, lamda, parameters, num_layers, layer_activations[-1])
            else:
                cost = compute_cost(A, Y_batch, layer_activations[-1])
            print(f"Epoch: {epoch} batch : {j + 1} cost : {cost}")

            grads = backward_propagation(A, Y_batch, caches, parameters, num_layers, layer_activations,
                                         regularization=regularization, batch_norm=batch_norm)

            if optimizer == 'momentum':
                parameters = update_parameters_with_momentum(grads, parameters, v, betav, num_layers, learning_rate)
            elif optimizer == 'rms':
                parameters = update_parameters_with_rmsprop(grads, parameters, s, betas, num_layers, learning_rate)
            elif optimizer == 'Adam':
                t = t + 1
                parameters = update_parameters_with_adam(grads, parameters, v, s, t, betav, betas, num_layers,
                                                         learning_rate)
            else:
                parameters = update_parameters(grads, parameters, num_layers, learning_rate)

        meta['train_loss'].append(cost)
        if epoch % 20 == 0:
            end_time = time.time()

            A_test, _ = forward_propagation(X_test, parameters, num_layers, layer_activations, dropout, batch_norm)
            if regularization == 'l1':
                test_cost = compute_cost_with_l1_regularization(A_test, Y_test, lamda, parameters, num_layers, layer_activations[-1])
            elif regularization == 'l2':
                test_cost = compute_cost_with_l2_regularization(A_test, Y_test, lamda, parameters, num_layers, layer_activations[-1])
            else:
                test_cost = compute_cost(A_test, Y_test, layer_activations[-1])
            meta['test_loss'].append(test_cost)

            meta['epochs'].append(epoch)
            _, acc = predict(X_train, Y_train, num_layers, layer_activations, parameters)
            meta['train_acc'].append(acc)
            print(f"Train Accuracy: {acc}")
            _, acc = predict(X_valid, Y_valid, num_layers, layer_activations, parameters)
            meta['valid_acc'].append(acc)
            print(f"Valid Accuracy: {acc}")
            if X_test is not None:
                _, acc = predict(X_test, Y_test, num_layers, layer_activations, parameters)
                meta['test_acc'].append(acc)
                print(f"Test Accuracy: {acc}")

            dur = round(float((end_time-start_time)/60), 2)
            tag_name = f"{tag}_epoch-{epoch}_dp-{dropout}_reg-{regularization}-{lamda}_lr-{learning_rate}_time-{dur}"
            if optimizer:
                tag_name += f'_opt-{optimizer}-{betav}' if optimizer != 'rms' else f'_opt-{optimizer}-{betav}-{betas}'

            loss_data = [(meta['train_loss'], [i+1 for i in range(epoch)]),
                         (meta['test_loss'], meta['epochs'])]
            acc_data = [(meta['train_acc'], meta['epochs']),
                        (meta['valid_acc'], meta['epochs']),
                        (meta['test_acc'], meta['epochs'])]

            plot(*loss_data, tag=tag_name, x_label='epochs', y_label='loss', path=os.path.join(wd, "loss.png"))
            plot(*acc_data, tag=tag_name, x_label='epochs', y_label='Acc.', path=os.path.join(wd, "accuracy.png"))

        if epoch < 1000 and epoch % 100 == 0 or epoch % 500 == 0:
            dump_pickle(os.path.join(wd, 'backup', f"backup_{epoch}.pkl"), parameters)
            dump_json(os.path.join(wd, 'meta.json'), meta)

    dump_pickle(os.path.join(wd, f"final_{epochs}.pkl"), parameters)
