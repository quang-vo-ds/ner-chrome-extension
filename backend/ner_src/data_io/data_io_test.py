import datasets
from ..utils import *

class DataIOTest():
    """
    DataIOTest is an input/output data wrapper for test data.
    """
    def __init__(self, dataset_name='test', train_pct=100, val_pct=100, test_pct=100):
        pass
    
    @staticmethod
    def read_data(article, verbose=False):
        word_sequences = list()
        for i, word in enumerate(article.split()):
            word_sequences.append(word)
        if verbose:
            print(word_sequences)
        return word_sequences