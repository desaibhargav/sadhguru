import sys
import pandas as pd
import pickle
import numpy as np
import torch

from typing import Union, Optional, Tuple, List
from sentence_transformers import SentenceTransformer, CrossEncoder, util
from backend.chunker import Chunker


def create_dataset(file: str) -> pd.DataFrame:
    """"""
    assert isinstance(
        file, str
    ), "please pass the path to a pickled pd.DataFrame object"
    try:
        dataset = pd.read_pickle(file)
        chunked = Chunker(
            chunk_by="length", expected_threshold=100, min_tolerable_threshold=75
        ).get_chunks(dataset)
        dataset_chunked = dataset.join(chunked).drop(
            columns=["subtitles", "timestamps"]
        )
        df = dataset_chunked.copy().dropna()
    except pickle.UnpicklingError:
        sys.exit("The passed file does not point to a pickled pd.DataFrame object")
