import torch
import random
import re

def batch_iter(data, batch_size=32, shuffle=True):
    """ Yield batch of (sent, tag), by the reversed order of source length.
    Args:
        data: list of tuples, each tuple contains a sentence and corresponding tag.
        batch_size: batch size
        shuffle: bool value, whether to random shuffle the data
    """
    data_size = len(data)
    indices = list(range(data_size))
    if shuffle:
        random.shuffle(indices)
    batch_num = (data_size + batch_size - 1) // batch_size
    for i in range(batch_num):
        batch = [data[idx] for idx in indices[i * batch_size: (i + 1) * batch_size]]
        batch = sorted(batch, key=lambda x: len(x[0]), reverse=True)
        sentences = [x[0] for x in batch]
        tags = [x[1] for x in batch]
        yield sentences, tags

def get_words_num(word_sequences):
    return sum(len(word_seq) for word_seq in word_sequences)

def pad(data, padded_token, device):
    """ pad data so that each sentence has the same length as the longest sentence
    Args:
        data: list of sentences, List[List[word]]
        padded_token: padded token
        device: device to store data
    Returns:
        padded_data: padded data, a tensor of shape (max_len, b)
        lengths: lengths of batches, a list of length b.
    """
    lengths = [len(sent) for sent in data]
    max_len = lengths[0]
    padded_data = []
    for s in data:
        padded_data.append(s + [padded_token] * (max_len - len(s)))
    return torch.tensor(padded_data, device=device), lengths

def words2indices(origin, vocab):
    """ Transform a sentence or a list of sentences from str to int
    Args:
        origin: a sentence of type list[str], or a list of sentences of type list[list[str]]
        vocab: Vocab instance
    Returns:
        a sentence or a list of sentences represented with int
    """
    if isinstance(origin[0], list):
        result = [[vocab[w] for w in sent] for sent in origin]
    else:
        result = [vocab[w] for w in origin]
    return result

def print_var(**kwargs):
    for k, v in kwargs.items():
        print(k, v)

def get_test_sequence(input_sentence):
    input_sentence = re.findall(r"[\w]+|[^\s\w]", input_sentence)
    return [input_sentence]