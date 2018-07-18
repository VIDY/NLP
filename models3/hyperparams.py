class Hyperparams:
    '''Hyperparameters'''
    # task number
    task_num = 1 # 1: porno, 2: emojis

    if task_num == 1:
        # data paths
        train_p = '/mnt/extra/emoji/sentiment/porno/train.txt'
        dev_p = '/mnt/extra/emoji/sentiment/porno/dev.txt'
        test_p = '/mnt/extra/emoji/sentiment/porno/test.txt'

        train_c = '/mnt/extra/emoji/sentiment/data/train.txt'
        dev_c = '/mnt/extra/emoji/sentiment/data/dev.txt'
        test_c = '/mnt/extra/emoji/sentiment/data/test.txt'

        # model
        num_vocab = 400000 # V
        max_len = 100 # T
        labels = ["porno", "clean"] # L
        hidden_units = 100  # alias = E, 50 | 100 | 200 | 300
        glove = '/mnt/extra/emoji/sentiment/glove.6B/glove.6B.{}d.txt'.format(hidden_units)
        num_blocks = 6  # number of encoder blocks
        dropout_rate = 0.2
        encoder_num_banks = 16
        num_highwaynet_blocks = 4
    elif task_num == 2:
        # data paths
        train = '/mnt/extra/emoji/sentiment/tweets/train.txt'
        dev = '/var/www/html/nlp/input/emoji.txt'
        #dev = 'tweets/dev.txt'
        test = '/mnt/extra/emoji/sentiment/tweets/test.txt'
        
        # model
        num_vocab = 400000 # V
        max_len = 100 # T
        labels = [emoji.split("\t")[0] for emoji in open("/mnt/extra/emoji/sentiment/tweets/emojis.txt", "r").read().splitlines()] # L
        max_labels = 3 # K. Maximum number of labels per sample
        hidden_units = 100  # E, 50 | 100 | 200 | 300
        glove = '/mnt/extra/emoji/sentiment/glove.6B/glove.6B.{}d.txt'.format(hidden_units)
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
    logdir = '/mnt/extra/emoji/sentiment/log/{}'.format(task_num)  # log directory
