class CFG:
    model_name = 'bilstm_crf'
    dataset_name = 'conll2003'
    seq_vocab_path = 'backend/ner_src/vocab/seq_vocab.json'
    tag_vocab_path = 'backend/ner_src/vocab/tag_vocab.json'
    state_dict_path = f'backend/ner_src/model/state_dict/{model_name}_state_dict.pth'
    verbose = True

    ## Data IO
    train_pct = 100
    val_pct = 100
    test_pct = 100

    ## Vocab
    max_dict_size = 5000
    freq_cutoff = 2

    ## Training hyperparameters
    max_epoch = 6
    batch_size = 10
    lr = 0.001
    lr_decay = 0.5
    max_decay = 0.4
    log_every = 10
    validation_every = 5
    clip_max_norm = 5.0
    max_patience = 4
    patience_threshold = 0.98
    result_file = 'backend/ner_src/result.txt'




    