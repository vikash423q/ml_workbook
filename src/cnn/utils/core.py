import pickle
from typing import Any


def handle_regularization(_func):
    def wrapper(*args, **kwargs):
        print('from wrapper')
        return _func(*args, **kwargs)


def load_pickle(file_name: str) -> Any:
    with open(file_name, 'rb') as f:
        return pickle.load(f)


def dump_pickle(file_name: str, data: Any) -> None:
    with open(file_name, 'wb') as f:
        pickle.dump(data, f)
