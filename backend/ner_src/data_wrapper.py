
import datasets
from .utils import *
from .data_io.data_io_hugface import DataIOHugFace
from .data_io.data_io_test import DataIOTest

class DataWrapper():
    """DataIO contains wrappers to create various data readers/writers."""
    @staticmethod
    def create(dataset_name, train_pct=100, val_pct=100, test_pct=100, verbose=True):
        if dataset_name in ['conll2003', 'test']:
            return DataIOHugFace(
                dataset_name = dataset_name,
                train_pct=train_pct, 
                val_pct=val_pct, 
                test_pct=test_pct
                )
        else:
            raise ValueError(f'Unknown DataIO {dataset_name}.')