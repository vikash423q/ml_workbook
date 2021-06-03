import os
import re
import json
import pickle

from typing import Any, List
import matplotlib.pyplot as plt


def plot(args, x_label: str = None, y_label: str = None, tag: str = None, path: str = '.'):
    for arg in args:
        epochs = [i+1 for i in range(len(arg))]
        plt.plot(epochs, arg)

    if y_label:
        plt.ylabel = y_label
    if x_label:
        plt.xlabel = x_label

    if tag:
        plt.title(tag)

    if path:
        plt.savefig(path)
    plt.clf()


def find_files_with_extension(root_dir: str, extension: str) -> List[str]:
    return [os.path.join(root_dir, f) for f in os.listdir(root_dir) if os.path.isfile(os.path.join(root_dir, f))
            and f.endswith(extension)]


def get_filename_without_ext(fullpath: str) -> str:
    filename = fullpath.split('/')[-1]
    return re.sub(r'\.[^.]*$', '', filename)


def load_pickle(file_name: str) -> Any:
    with open(file_name, 'rb') as f:
        return pickle.load(f)


def dump_pickle(file_name: str, data: Any) -> None:
    with open(file_name, 'wb') as f:
        pickle.dump(data, f)


def dump_json(file_name: str, _dict: dict, indent=1) -> None:
    with open(file_name, "w", encoding="utf8", newline="\n", errors="ignore") as f:
        json.dump(_dict, f, indent=indent)


def load_json(file_name: str) -> dict:
    with open(file_name) as f:
        return json.load(f)


def create_csv(filename: str, data: List[List[Any]]) -> None:
    lines = []
    for row in data:
        line = ",".join([str(r) for r in row]) + '\n'
        lines.append(line)

    with open(filename, 'w') as f:
        f.writelines(lines)


def read_csv(filename: str) -> List[List[Any]]:
    with open(filename) as f:
        lines = f.readlines()

    data = [line.replace('\n', '').split(',') for line in lines]
    return data


def read_tsv(filename: str) -> List[List[Any]]:
    with open(filename) as f:
        lines = f.readlines()

    data = [line.replace('\n', '').split('\t') for line in lines]
    return data


def file_to_string(file_path: str, encoding: str = 'utf-8') -> str:
    with open(file_path, mode='r', encoding=encoding) as f:
        return f.read()


def create_tsv(filename: str, data: List[List[Any]]) -> None:
    lines = []
    for row in data:
        line = "\t".join([str(r) for r in row]) + '\n'
        lines.append(line)

    with open(filename, 'w') as f:
        f.writelines(lines)
