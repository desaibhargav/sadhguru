import os
import pickle

from typing import Any


def save_to_cache(filename: str, file: Any):
    with open(os.path.join(os.getcwd(), "cache", f"{filename}.pickle"), "wb") as output:
        pickle.dump(
            file,
            output,
            pickle.HIGHEST_PROTOCOL,
        )


def load_from_cache(filename: str):
    with open(os.path.join(os.getcwd(), "cache", f"{filename}.pickle"), "rb") as input:
        return pickle.load(input)