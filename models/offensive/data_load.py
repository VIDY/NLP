from __future__ import print_function
from models.offensive.hyperparams import Hyperparams as hp
import tensorflow as tf
import numpy as np
import re
import codecs
import ast

def load_vocab(num_vocab=hp.num_vocab):
    vocab = ["<PAD>", "<UNK>"]
    for i, line in enumerate(open(hp.glove, 'r')):
        if i == num_vocab - 2: break
        word = line.split()[0]
        vocab.append(word)

    word2idx = {word: idx for idx, word in enumerate(vocab)}
    idx2word = {idx: word for idx, word in enumerate(vocab)}

    return word2idx, idx2word

def load_labels():
    label2idx = {label: idx for idx, label in enumerate(hp.labels)}
    idx2label = {idx: label for idx, label in enumerate(hp.labels)}
    return label2idx, idx2label

word2idx, idx2word = load_vocab()
label2idx, idx2label = load_labels()

def refine(text):
    text = text.lower()
    text = re.sub("[^ A-Za-z0-9\-']", "", text)

    text = [word2idx.get(word, 1) for word in text.split()]  # 1: <UNK>
    text = text[-hp.max_len:]
    return text

def refine2(text):
    indices, labels = [], []
    words = text.split()
    for word in words:
        word = word.lower()
        word = re.sub("[^ #A-Za-z0-9\-']", "", word)
        if word.startswith("#"): # tag
            labels.append(0)
        else:
            labels.append(1)

        indices.append(word2idx.get(word.replace("#", ""), 1)) # 1: <UNK>

    indices = indices[-hp.max_len:]
    labels = labels[-hp.max_len:]

    return indices, labels

def load_data(mode="train", data=[]):

    texts = []
    labels = []
    for entry in data:
        texts.append(np.array(refine(entry["text"]), np.int32).tostring())
        labels.append(0)

    return texts, labels

def get_batch_data():
    with tf.device('/cpu:0'):
        # Load data
        texts, labels = load_data(max_num=5000000)

        # calc total batch count
        num_batch = len(texts) // hp.batch_size

        # Create Queues
        text, label = tf.train.slice_input_producer([texts, labels])

        # str to int
        text = tf.decode_raw(text, tf.int32)
        text = tf.pad(text, [(hp.max_len, 0)])[-hp.max_len:] # prepadding

        label_shape = []


        # Batching
        texts, labels = tf.train.batch([text, label],
                                           num_threads=8,
                                           shapes=([hp.max_len,], label_shape),
                                           batch_size=hp.batch_size,
                                           capacity=hp.batch_size * 4,
                                           allow_smaller_final_batch=False)

    return texts, labels, num_batch  # (N, T), (N,), ()

