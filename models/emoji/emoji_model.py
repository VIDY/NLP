from __future__ import print_function


import os
import json
from models.emoji.hyperparams import Hyperparams as hp
import numpy as np
import tensorflow as tf
from models.emoji.train import Graph
from models.emoji.data_load import get_batch_data, load_vocab, load_data
from scipy.stats import spearmanr
import sys
import json
from tqdm import tqdm

class EmojiModel:
    def __init__(self):


        self.graph = Graph(mode="dev");
        print("Graph loaded")

        self.sess = tf.Session()
        self.saver = tf.train.Saver()
        self.saver.restore(self.sess, tf.train.latest_checkpoint(hp.logdir)); print("Model restored!")


    def eval(self,test_data):

        '''
        Get evaluation accuray.
        '''

        # Load data
        texts, Y = load_data(mode="dev",data=test_data)

        hp.batch_size=len(texts)

        # Parse
        X = np.zeros((len(texts), hp.max_len), np.int32)
        for i, text in enumerate(texts):
            text = np.fromstring(text, np.int32)
            X[i, -len(text):] = text

        Y = [np.fromstring(label, np.int32) for label in Y]


        # Feed-forward
        preds, seqlens = [], []
        for step in tqdm(range(len(X) // hp.batch_size)):
            # batch
            x = X[step * hp.batch_size: (step + 1) * hp.batch_size]
            # y = labels[step * hp.batch_size: (step + 1) * hp.batch_size]
            #
            # ys.extend(y)

            # predict

            preds_, seqlens_, gs = self.sess.run([self.graph.preds, self.graph.seqlens, self.graph.global_step], {self.graph.x: x})  # (N, K)
            seqlens.extend(seqlens_.tolist())
            preds.extend(preds_.tolist())

        # calculation

        hits, predictions, labels = 0, 0, 0

        final_results=[]
        for y, pred, seqlen in zip(Y, preds, seqlens):
            # y: ?, pred: K, seqlen: scalar
            seqlen = max(1, seqlen)


            pred = pred[:seqlen] # -> pred <= K

            # pred = [0, 1, 2]
            labeled_pred=[]
            for unlabeled_pred in pred:
              labeled_pred.append(hp.labels[unlabeled_pred])

            final_results.append(labeled_pred)

            hits += len(np.intersect1d(y, pred))
            predictions += len(pred)
            labels += len(y)

        return final_results
