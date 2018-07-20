from __future__ import print_function

import os

from models.offensive.hyperparams import Hyperparams as hp
import numpy as np
import tensorflow as tf
from models.offensive.train import Graph
from models.offensive.data_load import load_data
from tqdm import tqdm
from models.offensive.data_load import load_vocab


class OffensiveModel:
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

        texts, Y = load_data(mode="dev",data=test_data)
        hp.batch_size=len(texts)

        # Load vocab
        word2idx, idx2word = self.graph.word2idx, self.graph.idx2word

        # Parse
        X = np.zeros((len(texts), hp.max_len), np.int32)
        for i, text in enumerate(texts):
            text = np.fromstring(text, np.int32)
            X[i, -len(text):] = text

        # Feed-forward
        preds = []
        for step in tqdm(range(len(X) // hp.batch_size)):
            # batch
            x = X[step * hp.batch_size: (step + 1) * hp.batch_size]

            # predict
            preds_, gs = self.sess.run([self.graph.preds, self.graph.global_step], {self.graph.x: x}) # (N,)
            preds_ = preds_.tolist()
            preds.extend(preds_)


        final_results=[]
        for x, y, pred in zip(X, Y, preds):
          final_results.append(pred)

        return final_results

