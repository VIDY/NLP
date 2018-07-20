from __future__ import print_function
from models.category.hyperparams import Hyperparams as hp
import tensorflow as tf
import numpy as np
import re, regex
import codecs
import unicodedata
from tqdm import tqdm
import json

def load_vocab(num_vocab=hp.num_vocab):
    vocab = ["<PAD>", "<UNK>", "<EOS>"]
    for i, line in enumerate(open(hp.glove, 'r')):
        if i == num_vocab - 3: break
        word = line.split()[0]
        vocab.append(word)

    word2idx = {word: idx for idx, word in enumerate(vocab)}
    idx2word = {idx: word for idx, word in enumerate(vocab)}

    return word2idx, idx2word

def load_labels():
    cat2idx = {cat: idx for idx, cat in enumerate(hp.categories)}
    idx2cat = {idx: cat for idx, cat in enumerate(hp.categories)}
    return cat2idx, idx2cat

def load_data(mode="train",data=[]):
    word2idx, idx2word = load_vocab()
    cat2idx, idx2cat = load_labels()

    # Parse
    text_lengths, texts, categories = [], [], []

    for entry in data:
        print(entry)
        text, category = entry["text"],"theater"

        text = text.lower()
        text = ''.join(char for char in unicodedata.normalize('NFD', text)
                       if unicodedata.category(char) != 'Mn')  # Strip accents
        text = regex.sub("[^ A-Za-z0-9\-']", "", text)
        text = [word2idx.get(word, 1) for word in text.split() + ["<EOS>"]]
        text = text[-hp.max_len:]
        text_lengths.append(len(text))
        texts.append(np.array(text, np.int32).tostring())

        category = cat2idx[category]
        categories.append(category)

    # Monitor

    print("text lengths look like", text_lengths[:10])
    print("texts look like", " ".join(idx2word[t] for t in np.fromstring(texts[0], np.int32)))
    print("categories look like", categories[:10])
    #print("test_data", json.dumps(test_data))

    return text_lengths, texts, categories

def get_batch_data():
    with tf.device('/cpu:0'):
        # Load data
        text_lengths, texts, categories = load_data()

        # calc total batch count
        num_batch = len(text_lengths) // hp.batch_size

        # Create Queues
        text_length, text, category = tf.train.slice_input_producer([text_lengths, texts, categories])

        # str to int
        text = tf.decode_raw(text, tf.int32)
        text = tf.pad(text, [(hp.max_len, 0)])[-hp.max_len:] # prepadding

        # Batching
        texts, categories = tf.train.batch([text, category],
                                           num_threads=8,
                                           shapes=([hp.max_len,], []),
                                           batch_size=hp.batch_size,
                                           capacity=hp.batch_size * 4,
                                           allow_smaller_final_batch=False)

        # _, (texts, categories) = tf.contrib.training.bucket_by_sequence_length(
        #                                                 input_length=text_length,
        #                                                 tensors=[text, category],
        #                                                 batch_size=hp.batch_size,
        #                                                 bucket_boundaries=[i for i in range(minlen + 1, maxlen - 1, 10)],
        #                                                 num_threads=8,
        #                                                 capacity=hp.batch_size * 4,
        #                                                 dynamic_pad=True)

    return texts, categories, num_batch  # (N, T), (N,), ()

