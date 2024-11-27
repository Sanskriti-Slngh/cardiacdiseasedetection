from pydicom import dcmread
from my_lib import process_xml
import tensorflow as tf
import math
import os
import numpy as np
from matplotlib.path import Path
from tensorflow.keras.utils import to_categorical
import sys
import random
import csv
import pickle

class dataGenerator_ng(tf.keras.utils.Sequence):
    def __init__(self, pids, batch_size, ddir="../dataset",
                 limit_pids=None, shuffle=False, data = "train"
                 ):
        ddir_scans = {"N": "deidentified_nongated",
                      "G": "Gated_release_final"}
        if limit_pids:
            self.pids = pids[0:limit_pids]
        else:
            self.pids = pids
        self.ddir = ddir + "/" + ddir_scans['N']

        # Load all the images across pids
        self.X = []
        self.scores = []
        self.shuffle = shuffle

        # Read the data from file
        with open(ddir + "/train_dev_inputs.dump", 'rb') as fin:
            print(f"Loading test from {fin}")
            train_inputs, dev_inputs = pickle.load(fin)

        with open(ddir + "/Ntrain_dev_pids.dump", 'rb') as fin:
            print(f"Loading test from {fin}")
            train_pids, dev_pids = pickle.load(fin)

        train_inputs = np.transpose(train_inputs, (0, 2, 3, 1))
        dev_inputs = np.transpose(dev_inputs, (0, 2, 3, 1))

        if data == 'train':
            self.X = train_inputs
            self.pids = train_pids
        elif data == 'dev':
            self.X = dev_inputs
            self.pids = dev_pids
        elif data == 'test':
            with open(ddir + "/test_inputs.dump", 'rb') as fin:
                print(f"Loading test from {fin}")
                test_inputs = pickle.load(fin)
            with open(ddir + "/Ntest_pids.dump", 'rb') as fin:
                print(f"Loading test from {fin}")
                test_pids = pickle.load(fin)
            self.X = test_inputs
            self.pids = test_pids

        if limit_pids:
            self.pids = self.pids[0:limit_pids]
            self.X = self.X[0:limit_pids]

        score = {}
        with open(self.ddir + "/scores.csv") as fin:
            csvreader = csv.reader(fin)
            is_header = True
            for row in csvreader:
                if is_header:
                    is_header = False
                    continue

                pid, lca, lad, lcx, rca, total = row
                pid = pid.rstrip("A")
                total = float(total)
                score[pid] = total

        self.scores = []
        for pid in self.pids:
            self.scores.append(score[str(pid)])

        self.batch_size = batch_size

        # Reshuffle
        if self.shuffle:
            indices = [i for i in range(len(self.X))]
            random.shuffle(indices)
            xxx = [self.X[i] for i in indices]
            yyy = [self.scores[i] for i in indices]
            self.X = xxx
            self.scores = yyy

    def __len__(self):
        return math.ceil(self.X.shape[0] / self.batch_size)

    def __getitem__(self, idx):
        if ((idx +1 ) * self.batch_size) > self.X.shape[0]:
            Xs = self.X[idx * self.batch_size : self.X.shape[0]]
            Ys = self.scores[idx * self.batch_size: len(self.scores)]
           # pids = self.pids[idx * self.batch_size : len(self.pids)]
        else:
            Xs = self.X[idx * self.batch_size:(idx + 1) * self.batch_size]
            Ys = self.scores[idx * self.batch_size:(idx + 1) * self.batch_size]
           # pids = self.pids[idx * self.batch_size:(idx + 1) * self.batch_size]

        Ys = np.array(Ys)
        Ys = Ys.reshape(Ys.shape[0], 1)
        return Xs, Ys

    def on_epoch_end(self):
        # Reshuffle
        if self.shuffle:
            indices = [i for i in range(len(self.X))]
            random.shuffle(indices)
            xxx = [self.X[i] for i in indices]
            yyy = [self.mdata[i] for i in indices]
            self.X = xxx
            self.mdata = yyy
