class Hyperparams:
    '''Hyperparameters'''

    # model
    num_vocab = 400000 # V
    max_len = 100 # T
    labels = [emoji.split("\t")[0] for emoji in open("data/log/emoji/emojis.txt", "r").read().splitlines()] # L
    max_labels = 3 # K. Maximum number of labels per sample
    hidden_units = 100  # E, 50 | 100 | 200 | 300
    glove = 'data/glove.6B/glove.6B.{}d.txt'.format(hidden_units)
    num_blocks = 6  # number of encoder blocks
    dropout_rate = 0.2
    encoder_num_banks = 16
    num_highwaynet_blocks = 4

    # inference
    num_predictions = None # None: all whose prob. is 0.5.
    # 2: top 2 predictions
    # 3: top 3 predictions

    # training
    num_epochs = 500
    batch_size = 19  # N
    lr = 0.0005  # learning rate.
    logdir = 'data/log/emoji'  # log directory
