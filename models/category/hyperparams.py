# def embed(tensor):
#     # load pretrained word vectors
#     vectors = np.zeros((10000, 300), np.float32)
#     f = '/Users/ryan/git/sentvec/glove.6B/glove.6B.300d.txt'
#     for i, line in enumerate(open(f, 'r')):
#         if i == 10000: break
#         vector = line.split(" ")[1:]
#         vector = np.array(vector, np.float32)
#         vectors[i] = vector
#     lookup_table = tf.convert_to_tensor(vectors)
#
#     # associate
#     embedded = tf.nn.embedding_lookup(lookup_table, tensor)
#     return embedded

class Hyperparams:
    '''Hyperparameters'''
    # data paths
    train = 'data/train.txt'
    #dev = 'data/test.txt'
    dev = '/var/www/html/nlp/input/category.txt'
    test = 'data/test.txt'
    glove = '../glove.6B/glove.6B.300d.txt'


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
    logdir = '../log/01'  # log directory
    results = "results" # results





