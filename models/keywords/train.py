from __future__ import print_function
from models.keywords.hyperparams import Hyperparams as hp
import tensorflow as tf
from models.keywords.data_load import get_batch_data, load_vocab
from models.keywords.modules import *
from tqdm import tqdm

class Graph:
    def __init__(self, mode="train"):
        # Load vocab
        self.word2idx, self.idx2word = load_vocab()

        # Set phase
        is_training = True if mode=="train" else False

        # Load data
        if is_training: # x: text. (N, T), y: label(s). (N,) / (N, T)
            self.x, self.y, self.num_batch = get_batch_data()
        else:
            self.x = tf.placeholder(tf.int32, shape=(None, hp.max_len))

        # Embedding
        enc = embed(self.x,
                    vocab_size=hp.num_vocab,
                    num_units=hp.hidden_units,
                    scope="enc_embed")

        # Residual bi-GRUs 
        enc = gru(enc, num_units=hp.hidden_units//2, bidirection=True, scope="gru1")
        enc += gru(enc, num_units=hp.hidden_units//2, bidirection=True, scope="gru2")


        # Readout
        self.logits = tf.layers.dense(enc, len(hp.labels))  # (N, L) / (N, T, L)
        self.preds = tf.argmax(self.logits, -1) # (N,) / (N, T)

        self.global_step = tf.Variable(0, name='global_step', trainable=False)
        if is_training:
            # Loss
            self.loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=self.logits, labels=self.y) # (N,) / (N, T)
            self.masks = tf.to_float(tf.not_equal(self.x, 0))  # masking
            self.loss *= self.masks
            self.loss = tf.reduce_mean(self.loss)

            # Training Scheme
            self.optimizer = tf.train.AdamOptimizer(learning_rate=hp.lr)
            self.train_op = self.optimizer.minimize(self.loss, global_step=self.global_step)

            # Summary
            tf.summary.scalar('loss', self.loss)
            tf.summary.merge_all()

if __name__ == '__main__':
    # Construct graph
    g = Graph(); print("Graph loaded")

    # Session
    sv = tf.train.Supervisor(logdir=hp.logdir)
    with sv.managed_session() as sess:
        for epoch in range(1, hp.num_epochs + 1):
            if sv.should_stop(): break
            for step in tqdm(range(g.num_batch), total=g.num_batch, ncols=70, leave=False, unit='b'):
                sess.run(g.train_op)
    print("Done")


