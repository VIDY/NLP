class Hyperparams:
    '''Hyperparameters'''

    labels = ["is_tag", "not_tag"]
    n_eval = None

    # model
    num_vocab = 400000  # V
    max_len = 100  # T
    hidden_units = 100  # E, 50 | 100 | 200 | 300
    glove = 'data/glove.6B/glove.6B.{}d.txt'.format(hidden_units)
    num_blocks = 6  # number of encoder blocks
    dropout_rate = 0.2
    encoder_num_banks = 16
    num_highwaynet_blocks = 4

    # training
    num_epochs = 1000
    batch_size = 128  # N
    lr = 0.0005  # learning rate.
    logdir = 'data/log/keywords'  # log directory
