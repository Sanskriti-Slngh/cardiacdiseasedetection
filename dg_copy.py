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

class dataGenerator(tf.keras.utils.Sequence):
    def __init__(self, pids, batch_size, ddir="../dataset",
                 upsample_ps=0, limit_pids=None, shuffle=True,
                 only_use_pos_images=False, data_aug_enable=False,
                 num_neg_images_per_batch=0, scan_type="N"
                 ):
        ddir_scans = {"N": "deidentified_nongated",
                      "G": "Gated_release_final"}
        self.scan_type = scan_type
        if limit_pids:
            self.pids = pids[0:limit_pids]
        else:
            self.pids = pids
        self.ddir = ddir + "/" + ddir_scans[self.scan_type]
        # Load all the images across pids
        self.X = []
        self.scores = []
        self.mdata = []
        self.upsample_ps = upsample_ps
        self.cache = {}
        self.shuffle = shuffle
        self.only_use_pos_images = only_use_pos_images
        self.data_aug_enable = data_aug_enable
        # Use num_neg_images_per_batch variable to define number of negative images to use per batch.
        # It has to be used along with only_use_pos_images. In every batch, these many positive images
        # will be replaced by randomly selected negative images.
        self.num_neg_images_per_batch = num_neg_images_per_batch
        # These two neg_X/mdata variables contain negative images
        self.neg_X = []
        self.neg_mdata = []
        self.fixed_normalization = False
        self.just_segmentation = False

        # Estimate total work
        total_work = 0
        progress_count = 0
        for pid in self.pids:
            if self.scan_type == "G":
                directory_name = self.ddir + "/patient/" + str(pid) + '/'
            else:
                directory_name = self.ddir + "/" + str(pid) + '/'
            for subdir, dirs, files in os.walk(directory_name):
                total_work += len(files)

        min_value = 0
        #print (f"Loading dataset from {self.ddir}")
        for i, pid in enumerate(self.pids):
            print(pid)
            if self.scan_type == "G":
                directory_name = self.ddir + "/patient/" + str(pid) + '/'
            else:
                directory_name = self.ddir + "/" + str(pid) + '/'
            #print (directory_name)
            for subdir, dirs, files in os.walk(directory_name):
                for iidx, filename in enumerate(sorted(files, reverse=True)):
                    filepath = subdir + os.sep + filename
                    if filepath.endswith(".dcm"):
                        img = dcmread(filepath).pixel_array
                        if scan_type == "N":
                            img = img * (img > 0)
                        self.X.append(img)
                        self.mdata.append((pid, iidx))
        self.batch_size = batch_size
        sys.stdout.write("\n")

    def __len__(self):
        return math.ceil(len(self.X) / self.batch_size)

    def __getitem__(self, idx):
        if ((idx +1 ) * self.batch_size) > len(self.X):
            Xs = self.X[idx * self.batch_size : len(self.X)]
            mdatas = self.mdata[idx * self.batch_size : len(self.X)]
        else:
            Xs = self.X[idx * self.batch_size:(idx + 1) * self.batch_size]
            mdatas = self.mdata[idx * self.batch_size:(idx + 1) * self.batch_size]

        assert(self.num_neg_images_per_batch == 0), f"num_nag_images_per_batch ({self.num_neg_images_per_batch}) should only be used when self.only_use_pos_images is 1"

        # Normalize here
        Xs = np.array(Xs)
        norm_const = np.array(2 ** 16 - 1).astype('float32')
        if self.fixed_normalization:
            Xs = Xs / norm_const
        else:
            min_value = np.min(Xs)
            max_value = np.max(Xs)
            range = max_value - min_value
            Xs = (Xs - min_value)/range

        height, width = Xs[0].shape
        m = len(Xs)

        Ys = np.zeros((m, height, width, 1))

        # load XML and prepare Ys
        if self.scan_type == "N":
            for index, (pid, iidx) in enumerate(mdatas):
                fname = self.ddir + "/calcium_xml/" + str(pid) + (".xml")
                if not os.path.exists(fname):
                    continue
                if fname not in self.cache:
                    mdata = process_xml(fname)
                    self.cache[fname] = mdata
                else:
                    mdata = self.cache[fname]
                if iidx not in mdata:
                    continue
                for _ in mdata[iidx]:
                    poly_path = Path(_['pixels'])
                    y, x = np.mgrid[:height, : width]
                    coors = np.hstack((x.reshape(-1,1), y.reshape(-1,1)))
                    mask = poly_path.contains_points(coors).reshape(height, width)
                    Ys[index, :, :, 0] += mask

                    segmentations_fn = "D:/tiya2022/data/ngseg.csv"
                    s = csv.reader(open(segmentations_fn))
                    for row in s:
                        if row[0] == pid:
                            xadd = int(row[1])
                            yadd = int(row[2])
                            sliceg = int(row[3])
                            slicen = int(row[4])

                    fn = "D:/tiya2022/dataset/deidentified_nongated/" + str(pid)


            return np.array(Xs).reshape(m, height, width, 1), Ys


    def on_epoch_end(self):
        # Reshuffle
        if self.shuffle:
            indices = [i for i in range(len(self.X))]
            random.shuffle(indices)
            xxx = [self.X[i] for i in indices]
            yyy = [self.mdata[i] for i in indices]
            self.X = xxx
            self.mdata = yyy
