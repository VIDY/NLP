class Hyperparams:
    '''Hyperparameters'''
    glove = 'data/glove.6B/glove.6B.300d.txt'


    # model
    num_vocab = 50000
    max_len = 100
    categories = ["arts", "automobiles", "books", "business", "corrections", "education", "fashion", "health", "insider", "jobs", "magazine", "movies", "multimedia", "nyregion", "obituaries", "opinion", "politics", "reader-center", "realestate", "science", "smarter-living", "sports", "t-magazine", "technology", "theater", "travel", "upshot", "us", "well", "world"]
    hidden_units = 300  # alias = E
    num_blocks = 6  # number of encoder blocks
    dropout_rate = 0.2
    encoder_num_banks = 16
    num_highwaynet_blocks = 4

    # training
    num_epochs = 500
    batch_size = 32  # alias = N
    lr = 0.0005  # learning rate.
    logdir = 'data/log/category'  # log directory





