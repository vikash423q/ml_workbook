import pickle
from typing import Any


def load_pickle(file_name: str) -> Any:
    with open(file_name, 'rb') as f:
        return pickle.load(f)


def dump_pickle(file_name: str, data: Any) -> None:
    with open(file_name, 'wb') as f:
        pickle.dump(data, f)
