import datasets
from ..utils import *

class DataIOHugFace():
    """
    DataIOCONLL is an input/output data wrapper for HuggingFace datasets.
    """
    def __init__(self, dataset_name='conll2003', train_pct=100, val_pct=100, test_pct=100):
        ## Load dataset
        self.train_ds = datasets.load_dataset(dataset_name, split=f"train[:{train_pct}%]")
        self.val_ds = datasets.load_dataset(dataset_name, split=f"validation[:{val_pct}%]")
        self.test_ds = datasets.load_dataset(dataset_name, split=f"test[:{test_pct}%]")
    
    
    def read_train(self, verbose=False):
        return self.read_data(self.train_ds, verbose)
    
    def read_val(self, verbose=False):
        return self.read_data(self.val_ds, verbose)
    
    def read_test(self, verbose=False):
        return self.read_data(self.test_ds, verbose)
    
    @staticmethod
    def read_data(dataset, verbose=False):
        word_sequences = list()
        tag_sequences = list()
        for i, row in enumerate(dataset):
            if len(row['tokens']) == 0 or len(row['ner_tags']) == 0:
                continue
            word_sequences.append(row['tokens'])
            tag_sequences.append([str(w) for w in row['ner_tags']])
            
        if verbose:
            print(f'Loading {len(word_sequences)} samples, {get_words_num(word_sequences)} words.')
        return word_sequences, tag_sequences