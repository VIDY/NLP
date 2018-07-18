from __future__ import print_function
from models3.hyperparams import Hyperparams as hp
import tensorflow as tf
from models3.data_load import get_batch_data, load_vocab, load_labels
from models3.modules import *
from tqdm import tqdm

class Graph:
    def __init__(self, mode="train"):
        # Set Phase Flag
        is_training = True if mode=="train" else False

        # Load data
        if is_training: # x: text. (N, T), y: label(s). (N, (K))
            self.x, self.y, self.num_batch = get_batch_data()
        else:
            self.x = tf.placeholder(tf.int32, shape=(None, hp.max_len))
            # self.y = tf.placeholder(tf.int32)

        # Encoder
        ## Embedding
        enc = embed(self.x,
                        vocab_size=hp.num_vocab,
                        num_units=hp.hidden_units,
                        scope="enc_embed")


        # Encoder pre-net
        prenet_out = prenet(enc,
                            num_units=[hp.hidden_units, hp.hidden_units//2],
                            dropout_rate=hp.dropout_rate,
                            is_training=is_training)  # (N, T, E/2)

        # Encoder CBHG
        ## Conv1D bank
        enc = conv1d_banks(prenet_out,
                           K=hp.encoder_num_banks,
                           is_training=is_training)  # (N, T, K * E / 2)

        ### Max pooling
        enc = tf.layers.max_pooling1d(enc, 2, 1, padding="same")  # (N, T, K * E / 2)

        ### Conv1D projections
        enc = conv1d(enc, hp.hidden_units//2, 3, scope="conv1d_1")  # (N, T, E/2)
        enc = bn(enc, is_training=is_training, activation_fn=tf.nn.relu, scope="norm1")
        enc = conv1d(enc, hp.hidden_units//2, 3, scope="conv1d_2")  # (N, T, E/2)
        enc = bn(enc, is_training=is_training, activation_fn=tf.nn.relu, scope="norm2")
        enc += prenet_out  # (N, T, E/2) # residual connections

        ### Highway Nets
        for i in range(hp.num_highwaynet_blocks):
            enc = highwaynet(enc, num_units=hp.hidden_units//2,
                             scope='highwaynet_{}'.format(i))  # (N, T, E/2)

        # GRU
        enc = gru(enc, num_units=hp.hidden_units//2) # (N, T, E/2)

        # last hidden vectors
        enc = enc[:, -1, :] # (N, E/2)

        # Readout
        self.logits = tf.layers.dense(enc, len(hp.labels))  # (N, L)
        if hp.task_num == 1:
            self.preds = tf.argmax(self.logits, -1) # (N,)
        elif hp.task_num == 2:
            self.val, self.preds = tf.nn.top_k(self.logits, k=hp.max_labels) # (N, K), (N, K)
            self.seqlens = tf.count_nonzero(tf.greater(self.val, 0.), -1) # (N,). x > 0.

        self.global_step = tf.Variable(0, name='global_step', trainable=False)
        if is_training:
            # Loss
            if hp.task_num == 1:
                self.loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=self.logits, labels=self.y)
            elif hp.task_num == 2:
                self.labels = tf.one_hot(indices=self.y, depth=len(hp.labels)) # (N, K, L)
                self.labels = tf.reduce_sum(self.labels, 1) # (N, L)
                self.loss = tf.nn.sigmoid_cross_entropy_with_logits(logits=self.logits, labels=self.labels)
            self.loss = tf.reduce_mean(self.loss)

            # Training Scheme
            self.optimizer = tf.train.AdamOptimizer(learning_rate=hp.lr)
            self.train_op = self.optimizer.minimize(self.loss, global_step=self.global_step)

            # Summary
            tf.summary.scalar('loss', self.loss)
            tf.summary.merge_all()

if __name__ == '__main__':
    # Construct graph
    g = Graph()
    print("Graph loaded")

    # Load vocabulary
    word2idx, idx2word = load_vocab()
    label2idx, idx2label = load_labels()

    # Start a session
    sv = tf.train.Supervisor(logdir=hp.logdir)
    with sv.managed_session() as sess:
        for epoch in range(1, hp.num_epochs + 1):
            if sv.should_stop(): break
            for step in tqdm(range(g.num_batch), total=g.num_batch, ncols=70, leave=False, unit='b'):
                sess.run(g.train_op)

                # # Monitor
                # if step % 1000 == 0:
                #     x, y, preds = sess.run([g.x, g.y, g.preds])
                #     x0, y0, pred = x[0], y[0], preds[0]
                #     print("input:", " ".join(idx2word[idx] for idx in x0))
                #     print("label:", idx2label[y0])
                #     print("pred:", idx2label[pred])

    print("Done")


