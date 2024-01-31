import os
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

import datasets
from itertools import chain
from collections import Counter
import random
import json
import time
import click

from .ner_src.cfg import CFG
from .ner_src import utils
from .ner_src.data_wrapper import *
from .ner_src.model_wrapper import *
from .ner_src.vocab import Vocab


@click.group()
def cli():
    pass

@cli.command()
@click.option('--dataset_name', prompt = "dataset name: ", default=CFG.dataset_name, help='Name of dataset', show_default=True)
@click.option('--model_name', prompt = "model name: ", default=CFG.model_name, help='Which architecture to use', show_default=True)
@click.option('--seq_vocab_path', prompt = "sequence vocab path: ", default=CFG.seq_vocab_path, help='Where to save the dict', show_default=True)
@click.option('--tag_vocab_path', prompt = "tag vocab path: ", default=CFG.tag_vocab_path, help='Where to save the dict', show_default=True)
@click.option('--state_dict_path', prompt = "state-dict path: ", default=CFG.state_dict_path, help='Where to save the dict', show_default=True)
@click.option('--train_pct', prompt = "train pct: ", default=CFG.train_pct, help='Train percentage to load', show_default=True)
@click.option('--val_pct', prompt = "val pct: ", default=CFG.val_pct, help='Validation percentage to load', show_default=True)
@click.option('--test_pct', prompt = "test pct: ", default=CFG.test_pct, help='Test percentage to load', show_default=True)
@click.option('--verbose', prompt = "verbose: ", default=CFG.verbose, help='Show additional information', show_default=True)
def train(**args):
    """ Training BiLSTMCRF model
    Args:
        args: dict that contains options in command
    """
    ## Load data for training
    data_io = DataWrapper().create(dataset_name=args['dataset_name'], train_pct=args['train_pct'], val_pct=args['val_pct'], test_pct=args['test_pct'])
    seq_train, tag_train = data_io.read_train(args['verbose'])
    seq_val, tag_val = data_io.read_val(args['verbose'])
    data_train = list(zip(seq_train, tag_train))
    data_val = list(zip(seq_val, tag_val))
    print('num of training examples: %d' % (len(data_train)))
    print('num of validation examples: %d' % (len(data_val)))

    ## Create vocab or load from files
    if os.path.isfile(args['seq_vocab_path']) and os.path.isfile(args['tag_vocab_path']):
        seq_vocab = Vocab.load(args['seq_vocab_path'])
        tag_vocab = Vocab.load(args['tag_vocab_path'])
    else:
        seq_vocab = Vocab.build(seq_train, int(args['max_dict_size']), int(args['freq_cutoff']), is_tags=False)
        tag_vocab = Vocab.build(tag_train, int(args['max_dict_size']), int(args['freq_cutoff']), is_tags=True)
        seq_vocab.save(args['seq_vocab_path'])
        tag_vocab.save(args['tag_vocab_path'])

    max_epoch = int(CFG.max_epoch)
    log_every = int(CFG.log_every)
    validation_every = int(CFG.validation_every)
    # optimizer_save_path = args['--optimizer-save-path']
    min_dev_loss = float('inf')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    patience, decay_num = 0, 0

    ## Init model and its weight
    model = ModelWrapper().create(args['model_name'], seq_vocab, tag_vocab)
    for name, param in model.named_parameters():
        if 'weight' in name:
            nn.init.normal_(param.data, 0, 0.01)
        else:
            nn.init.constant_(param.data, 0)

    optimizer = torch.optim.Adam(model.parameters(), lr=float(CFG.lr))
    train_iter = 0  # train iter num
    record_loss_sum, record_tgt_word_sum, record_batch_size = 0, 0, 0  # sum in one training log
    cum_loss_sum, cum_tgt_word_sum, cum_batch_size = 0, 0, 0  # sum in one validation log
    record_start, cum_start = time.time(), time.time()

    print('start training...')
    for epoch in range(max_epoch):
        for sequences, tags in utils.batch_iter(data_train, batch_size=int(CFG.batch_size)):
            seq_tokens = utils.words2indices(sequences, seq_vocab)
            tag_tokens = utils.words2indices(tags, tag_vocab)
            train_iter += 1
            current_batch_size = len(seq_tokens)
            sequences, seq_lengths = utils.pad(seq_tokens, seq_vocab[seq_vocab.PAD], device)
            tags, _ = utils.pad(tag_tokens, tag_vocab[tag_vocab.PAD], device)

            # back propagation
            optimizer.zero_grad()
            batch_loss = model(sequences, tags, seq_lengths)  # shape: (b,)
            loss = batch_loss.mean()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=float(CFG.clip_max_norm))
            optimizer.step()

            record_loss_sum += batch_loss.sum().item()
            record_batch_size += current_batch_size
            record_tgt_word_sum += sum(seq_lengths)

            cum_loss_sum += batch_loss.sum().item()
            cum_batch_size += current_batch_size
            cum_tgt_word_sum += sum(seq_lengths)

            if train_iter % log_every == 0:
                print('log: epoch %d, iter %d, %.1f words/sec, avg_loss %f, time %.1f sec' %
                      (epoch + 1, train_iter, record_tgt_word_sum / (time.time() - record_start),
                       record_loss_sum / record_batch_size, time.time() - record_start))
                record_loss_sum, record_batch_size, record_tgt_word_sum = 0, 0, 0
                record_start = time.time()

            if train_iter % validation_every == 0:
                print('dev: epoch %d, iter %d, %.1f words/sec, avg_loss %f, time %.1f sec' %
                      (epoch + 1, train_iter, cum_tgt_word_sum / (time.time() - cum_start),
                       cum_loss_sum / cum_batch_size, time.time() - cum_start))
                cum_loss_sum, cum_batch_size, cum_tgt_word_sum = 0, 0, 0

                dev_loss = cal_dev_loss(model, data_val, 64, seq_vocab, tag_vocab, device)
                if dev_loss < min_dev_loss * float(CFG.patience_threshold):
                    min_dev_loss = dev_loss
                    model.save(args['state_dict_path'])
                    # torch.save(optimizer.state_dict(), optimizer_save_path)
                    patience = 0
                else:
                    patience += 1
                    if patience == int(CFG.max_patience):
                        decay_num += 1
                        if decay_num == int(CFG.max_decay):
                            print(f'Early stop. Save result model to {args["state_dict_path"]}')
                            return
                        lr = optimizer.param_groups[0]['lr'] * float(CFG.lr_decay)
                        model = ModelWrapper().create(args['model_name'], seq_vocab, tag_vocab)
                        model = model.load(args['state_dict_path'], device)
                        # optimizer.load_state_dict(torch.load(optimizer_save_path))
                        # for param_group in optimizer.param_groups:
                        #     param_group['lr'] = lr
                        patience = 0
                print('dev: epoch %d, iter %d, dev_loss %f, patience %d, decay_num %d' %
                      (epoch + 1, train_iter, dev_loss, patience, decay_num))
                cum_start = time.time()
                if train_iter % log_every == 0:
                    record_start = time.time()
    print(f'Reached {max_epoch} epochs, Save result model to {args["state_dict_path"]}')

@cli.command()
@click.option('--dataset_name', prompt = "dataset name: ", default=CFG.dataset_name, help='Name of dataset', show_default=True)
@click.option('--model_name', prompt = "model name: ", default=CFG.model_name, help='Which architecture to use', show_default=True)
@click.option('--seq_vocab_path', prompt = "sequence vocab path: ", default=CFG.seq_vocab_path, help='Where to save the dict', show_default=True)
@click.option('--tag_vocab_path', prompt = "tag vocab path: ", default=CFG.tag_vocab_path, help='Where to save the dict', show_default=True)
@click.option('--state_dict_path', prompt = "state-dict path: ", default=CFG.state_dict_path, help='Where to save the dict', show_default=True)
@click.option('--train_pct', prompt = "train pct: ", default=CFG.train_pct, help='Train percentage to load', show_default=True)
@click.option('--val_pct', prompt = "val pct: ", default=CFG.val_pct, help='Validation percentage to load', show_default=True)
@click.option('--test_pct', prompt = "test pct: ", default=CFG.test_pct, help='Test percentage to load', show_default=True)
@click.option('--verbose', prompt = "verbose: ", default=CFG.verbose, help='Show additional information', show_default=True)
def test(**args):
    """ Testing the model
    Args:
        args: dict that contains options in command
    """
    seq_vocab = Vocab.load(args['seq_vocab_path'])
    tag_vocab = Vocab.load(args['tag_vocab_path'])
    data_io = DataWrapper().create(dataset_name=args['dataset_name'], train_pct=args['train_pct'], val_pct=args['val_pct'], test_pct=args['test_pct'])
    seq_test, tag_test = data_io.read_test(args['verbose'])
    seq_test = utils.words2indices(seq_test, seq_vocab)
    tag_test = utils.words2indices(tag_test, tag_vocab)
    data_test = list(zip(seq_test, tag_test))
    print('num of test samples: %d' % (len(data_test)))

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = ModelWrapper().create(args['model_name'], seq_vocab, tag_vocab)
    model.load(CFG.state_dict_path, device)
    print('start testing...')
    print('using device', device)

    result_file = open(CFG.result_file, 'w')
    with torch.no_grad():
        for sentences, tags in utils.batch_iter(data_test, batch_size=int(CFG.batch_size), shuffle=False):
            padded_sentences, sent_lengths = utils.pad(sentences, seq_vocab[seq_vocab.PAD], device)
            predicted_tags = model.predict(padded_sentences, sent_lengths)
            for sent, true_tags, pred_tags in zip(sentences, tags, predicted_tags):
                sent, true_tags, pred_tags = sent[1: -1], true_tags[1: -1], pred_tags[1: -1]
                for token, true_tag, pred_tag in zip(sent, true_tags, pred_tags):
                    result_file.write(' '.join([seq_vocab.id2word(token), tag_vocab.id2word(true_tag),
                                                tag_vocab.id2word(pred_tag)]) + '\n')
                result_file.write('\n')
    result_file.close()

def cal_dev_loss(model, data_val, batch_size, seq_vocab, tag_vocab, device):
    """ Calculate loss on the development data
    Args:
        model: the model being trained
        dev_data: development data
        batch_size: batch size
        sent_vocab: sentence vocab
        tag_vocab: tag vocab
        device: torch.device on which the model is trained
    Returns:
        the average loss on the dev data
    """
    is_training = model.training
    model.eval()
    loss, n_sentences = 0, 0
    with torch.no_grad():
        for sequences, tags in utils.batch_iter(data_val, batch_size, shuffle=False):
            seq_tokens = utils.words2indices(sequences, seq_vocab)
            tag_tokens = utils.words2indices(tags, tag_vocab)
            seq_tokens, seq_lengths = utils.pad(seq_tokens, seq_vocab[seq_vocab.PAD], device)
            tag_tokens, _ = utils.pad(tag_tokens, tag_vocab[tag_vocab.PAD], device)
            batch_loss = model(seq_tokens, tag_tokens, seq_lengths)  # shape: (b,)
            loss += batch_loss.sum().item()
            n_sentences += len(seq_tokens)
    model.train(is_training)
    return loss / n_sentences

def predict_tags(input_sentence):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    seq_vocab = Vocab.load(CFG.seq_vocab_path)
    tag_vocab = Vocab.load(CFG.tag_vocab_path)
    word_sequences = utils.get_test_sequence(input_sentence)
    seq_test = utils.words2indices(word_sequences, seq_vocab)
    seq_test = torch.tensor(seq_test, device=device)
    model = ModelWrapper().create(CFG.model_name, seq_vocab, tag_vocab)
    model.load(CFG.state_dict_path, device)
    predicted_tags = model.predict(seq_test, [len(seq_test[0])])
    return predicted_tags


if __name__ == '__main__':
    cli()