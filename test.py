
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

import datasets
from itertools import chain
from collections import Counter
import random
import json
import click

from backend.ner_src.cfg import CFG
from backend.ner_src import utils
from backend.ner_src.data_wrapper import *
from backend.ner_src.model_wrapper import *
from backend.ner_src.vocab import Vocab

@click.group()
def cli():
    pass

@cli.command()
@click.option('--dataset_name', prompt = "dataset name: ", default=CFG.dataset_name, help='Name of dataset', show_default=True)
@click.option('--train_pct', prompt = "train pct: ", default=CFG.train_pct, help='Train percentage to load', show_default=True)
@click.option('--val_pct', prompt = "val pct: ", default=CFG.val_pct, help='Validation percentage to load', show_default=True)
@click.option('--test_pct', prompt = "test pct: ", default=CFG.test_pct, help='Test percentage to load', show_default=True)
def test_data_io(**args):
    data_io = DataWrapper().create(dataset_name=args['dataset_name'], train_pct=args['train_pct'], val_pct=args['val_pct'], test_pct=args['test_pct'])
    word_sequences_train, tag_sequences_train = data_io.read_train(True)
    word_sequences_dev, tag_sequences_dev = data_io.read_val(True)
    word_sequences_test, tag_sequences_test = data_io.read_test(True)
    click.echo(f"Train size: {len(word_sequences_train)} | {len(tag_sequences_train)}")
    click.echo(f"Val size: {len(word_sequences_dev)} | {len(tag_sequences_dev)}")
    click.echo(f"Test size: {len(word_sequences_test)} | {len(tag_sequences_test)}")
    click.echo(word_sequences_train[0:2])
    click.echo(tag_sequences_train[0:2])


@cli.command()
@click.option('--dataset_name', prompt = "dataset name: ", default=CFG.dataset_name, help='Name of dataset', show_default=True)
@click.option('--max_dict_size', prompt = "Max size: ", default=CFG.max_dict_size, help='The maximum size of dict', show_default=True)
@click.option('--freq_cutoff', prompt = "Frequency cutoff: ", default=CFG.freq_cutoff, help='If a word occurs less than freq_size times, it will be dropped.', show_default=True)
@click.option('--seq_vocab_path', prompt = "Sequence vocab path: ", default=CFG.seq_vocab_path, help='Where to save the dict', show_default=True)
@click.option('--tag_vocab_path', prompt = "Tag vocab path: ", default=CFG.tag_vocab_path, help='Where to save the dict', show_default=True)
def test_vocab(**args):
    data_io = DataWrapper().create(dataset_name=args['dataset_name'], train_pct=args['train_pct'], val_pct=args['val_pct'], test_pct=args['test_pct'])
    sequences, tags = data_io.read_train(True)
    sent_vocab = Vocab.build(sequences, int(args['max_dict_size']), int(args['freq_cutoff']), is_tags=False)
    tag_vocab = Vocab.build(tags, int(args['max_dict_size']), int(args['freq_cutoff']), is_tags=True)
    sent_vocab.save(args['seq_vocab_path'])
    tag_vocab.save(args['tag_vocab_path'])

@cli.command()
@click.option('--dataset_name', prompt = "dataset name: ", default=CFG.dataset_name, help='Name of dataset', show_default=True)
def test_batch_iter(**args):
    data_io = DataWrapper().create(dataset_name=args['dataset_name'], train_pct=1)
    sequences, tags = data_io.read_train(True)
    data = list(zip(sequences, tags))
    batch = utils.batch_iter(data, batch_size=3, shuffle=False)
    click.echo(next(batch))

@cli.command()
@click.option('--dataset_name', prompt = "dataset name: ", default=CFG.dataset_name, help='Name of dataset', show_default=True)
@click.option('--seq_vocab_path', prompt = "Sequence vocab path: ", default=CFG.seq_vocab_path, help='Where to save the dict', show_default=True)
@click.option('--tag_vocab_path', prompt = "Tag vocab path: ", default=CFG.tag_vocab_path, help='Where to save the dict', show_default=True)
def test_tokenizer(**args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    sent_vocab = Vocab.load(args['seq_vocab_path'])
    tag_vocab = Vocab.load(args['tag_vocab_path'])
    data_io = DataWrapper().create(dataset_name=args['dataset_name'], train_pct=1)
    sequences, tags = data_io.read_train(True)
    data = list(zip(sequences, tags))
    batch = utils.batch_iter(data, batch_size=5, shuffle=False)
    sequences, tags = next(batch)
    seq_tokens = utils.words2indices(sequences, sent_vocab)
    tag_tokens = utils.words2indices(tags, tag_vocab)
    seq_tokens, _ = utils.pad(seq_tokens, sent_vocab[sent_vocab.PAD], device)
    tag_tokens, _ = utils.pad(tag_tokens, tag_vocab[tag_vocab.PAD], device)
    click.echo(sequences)
    click.echo(seq_tokens)
    click.echo(tags)
    click.echo(tag_tokens)

@cli.command()
@click.option('--dataset_name', prompt = "dataset name: ", default=CFG.dataset_name, help='Name of dataset', show_default=True)
@click.option('--model_name', prompt = "Model name: ", default=CFG.model_name, help='Which architecture to use', show_default=True)
@click.option('--seq_vocab_path', prompt = "Sequence vocab path: ", default=CFG.seq_vocab_path, help='Where to save the dict', show_default=True)
@click.option('--tag_vocab_path', prompt = "Tag vocab path: ", default=CFG.tag_vocab_path, help='Where to save the dict', show_default=True)
@click.option('--state_dict_path', prompt = "State-dict path: ", default=CFG.state_dict_path, help='Where to save the dict', show_default=True)
def test_model(**args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    seq_vocab = Vocab.load(args['seq_vocab_path'])
    tag_vocab = Vocab.load(args['tag_vocab_path'])
    data_io = DataWrapper().create(dataset_name=args['dataset_name'], train_pct=1)
    sequences, tags = data_io.read_train(True)
    data = list(zip(sequences, tags))
    batch = utils.batch_iter(data, batch_size=2, shuffle=False)
    sequences, tags = next(batch)
    seq_tokens = utils.words2indices(sequences, seq_vocab)
    tag_tokens = utils.words2indices(tags, tag_vocab)
    seq_tokens, seq_lengths = utils.pad(seq_tokens, seq_vocab[seq_vocab.PAD], device)
    tag_tokens, _ = utils.pad(tag_tokens, tag_vocab[tag_vocab.PAD], device)
    model = ModelWrapper().create(args['model_name'], seq_vocab, tag_vocab)
    print(model(seq_tokens, tag_tokens, seq_lengths))
    model.save(args['state_dict_path'])
    model.load(args['state_dict_path'], device)

if __name__ == "__main__":
    cli()