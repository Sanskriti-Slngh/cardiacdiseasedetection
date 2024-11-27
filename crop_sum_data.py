import numpy as np
from pydicom import dcmread
import pickle
import os
import matplotlib.pyplot as plt
import csv
from my_lib import process_xml
import argparse
from matplotlib.path import Path
import skimage

import tensorflow as tf
import gc

import models.unet as unet

# Model parameters
batch_size = 8
model_name = "unet"
#params['loss'] = "dice"


params = {}
params['models_dir'] = "../trained_models/unet"

generate_gated_train_dev_test_set = True
sdir_id = "N"
train_set_size = 0.8
dev_set_size = 0.1

# data path directory
ddir = "../dataset"
sdirs = {"G" : "Gated_release_final", "N" : "deidentified_nongated"}

# set the random seed to create the same train/val/test split
np.random.seed(10015321)
debug = 2

# process non-gated CTs
k = "N"
sdir = f"{ddir}/{sdirs[k]}"

pids_final = []

count = 0
for subdir, dirs, files in os.walk(sdir):
    for filename in sorted(files, reverse=True):
        filepath = str.replace(subdir, "\\", "/") + "/" + filename
        pid = filepath.split("/")[-3]
    count += 1
    if count%2 == 0:
        pids_final.append(pid)

    else:
        continue

pids_final = pids_final[2:]
print(pids_final)

#pids_final = ["1"]
data = {}
for subdir, dirs, files in os.walk(sdir):
    for filename in sorted(files, reverse=True):
        filepath = str.replace(subdir, "\\", "/") + "/" + filename
        pid = filepath.split("/")[-3]
        print(pid)
        if filepath.endswith(".dcm") and pid in pids_final:
            if (not pid in data):
                data[pid] = {}
                data[pid]['images'] = []
            else:
                filename_path_g = "../dataset" + "/Gated_release_final" + "/patient/" + str(pid) + ''
                for s, d, f in os.walk(filename_path_g):
                    for fl in sorted(f, reverse=True):
                        fp = s + os.sep + fl
                        if fp.endswith(".dcm"):
                            ds = dcmread(fp)
                            try:
                                data_Centers_g = (ds[0x0018, 0x9313].value)
                                recons_g = (ds[0x0018, 0x9318].value)
                                diameter = ds[0x0018, 0x1100].value
                            except:
                                data_Centers_g = [0.203125, -155.296875, -119.25]
                                recons_g = [14.203125, -175.296875, -119.25]
                                diameter = 208

                ds = dcmread(filepath)

                try:
                    pixel_length = ds[0x0028, 0x0030].value[0]
                    recon = ds[0x0018, 0x9318].value
                    subtract_x = ds[0x0018, 0x9313].value[0] - data_Centers_g[0]
                    subtract_y = ds[0x0018, 0x9313].value[1] - data_Centers_g[1]
                except:
                    pixel_length = 0.751953125
                    recon = [-4.6240234375, -181.6240234375, 896.4]
                    subtract_x = 0.3759765625 - data_Centers_g[0]
                    subtract_y = -181.6240234375 - data_Centers_g[1]

                r = int((diameter / 2) / pixel_length)
                sub_pixel_x = (recons_g[0] + subtract_x - recon[0]) / pixel_length
                sub_pixel_y = (recons_g[1] + subtract_y - recon[1]) / pixel_length

                x = 256 + sub_pixel_x - r
                y = 256 + sub_pixel_y - r
                p = ds.pixel_array

                if x < 0 or y < 0:
                    data_Centers_g = [0.203125, -155.296875, -119.25]
                    recons_g = [14.203125, -175.296875, -119.25]
                    diameter = 208
                    pixel_length = 0.751953125
                    recon = [-4.6240234375, -181.6240234375, 896.4]
                    subtract_x = 0.3759765625 - data_Centers_g[0]
                    subtract_y = -181.6240234375 - data_Centers_g[1]
                    r = int((diameter / 2) / pixel_length)
                    sub_pixel_x = (recons_g[0] + subtract_x - recon[0]) / pixel_length
                    sub_pixel_y = (recons_g[1] + subtract_y - recon[1]) / pixel_length
                    x = 256 + sub_pixel_x - r
                    y = 256 + sub_pixel_y - r

                q = p[int(y):int(y + r * 2), int(x):int(x + r * 2)]
                try:
                    w = skimage.transform.resize(q, (512, 512), order=0, preserve_range=True)
                    data[pid]['images'].append(w)
                except:
                    print("count")
                    print(pid)
                    print(np.sum(q))
                    exit()
                    data[pid]['images'].append(np.zeros((512,512)))


# process non-gated score file
with open(sdir + "/scores.csv") as fin:
    csvreader = csv.reader(fin)
    is_header = True
    for row in csvreader:
        if is_header:
            is_header = False
            continue
        pid, lca, lad, lcx, rca, total = row
        total = float(total)
        pid = pid.rstrip("A")
        print(pid, total)
        if (str(pid) in data):
            data[pid]['mdata'] = total
        else:
            print(f"WARNING: {pid}.xml found but no matching images")

# Load the trained unet model
# Load Model
if model_name == 'unet':
    model = unet.Model(None, params)
else:
    model = None
    exit("Something went wrong, model not defined")

model.is_train = False
# We may not need this?
optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
model.compile(optimizer)


xs = []
ys = []

for pid in pids_final:
    print(pid)
    if True:
        try:
            ys.append(data[pid]['mdata'])
            a = np.array(data[pid]['images'])
            print(a.shape)

            if a.shape[0] > 60:
                a = a[:60, :, :]
            else:
                b = np.zeros((60, 512, 512), dtype=np.uint16)
                b[0:a.shape[0], :, :] = a[:, :, :]
                a = b
            a = np.reshape(a, (60, 512, 512, 1))
            min_value = np.min(a)
            max_value = np.max(a)
            diff = max_value - min_value
            a = (a - min_value) / diff
            binary_mask = model.model.predict(a)
            binary_mask = np.sum(binary_mask, axis=0)
            xs.append(binary_mask)
        except:
            print(pid)
            print("^ doesnt have mdata")



xs = np.array(xs)
ys = np.array(ys)
print(xs.shape, ys.shape)

if generate_gated_train_dev_test_set:
    tidx = int(xs.shape[0] * train_set_size)
    didx = int(xs.shape[0] * (train_set_size + dev_set_size))
    print(tidx)
    print(didx)
    train_x = xs[0:tidx, :, :, :]
    dev_x = xs[tidx:didx, :, :, :]
    test_x = xs[didx:, : , :, :]

    train_xr = xs[0:tidx, :, :, :]
    dev_xr = xs[tidx:didx, :, :, :]
    test_xr = xs[didx:, :, :, :]
    train_y = ys[0:tidx]
    dev_y = ys[tidx:didx]
    test_y = ys[didx:]
    train_p = pids_final[0:tidx]
    dev_p = pids_final[tidx:didx]
    test_p = pids_final[didx:]


    print(train_x.shape, dev_x.shape, test_x.shape)
    print(train_y.shape, dev_y.shape, test_y.shape)
    print(train_p)
    print(dev_p)
    print(test_p)

    # dump into train file
    fname = "../dataset" + "/ntrain_ngsum_crop.dump"
    with open(fname, 'wb') as fout:
        print (f"Saving train into {fname}")
        pickle.dump((train_x, train_y, train_p), fout, protocol=4)
#        pickle.dump((train_xr, train_x, train_fxr, train_fx, train_y, train_p), fout, protocol=4)

    #  dump into dev file
    fname = "../dataset" + "/ndev_ngsum_crop.dump"
    with open(fname, 'wb') as fout:
        print (f"Saving dev into {fname}")
        pickle.dump((dev_x, dev_y, dev_p), fout, protocol=4)
      #  pickle.dump((dev_xr, dev_x, dev_fxr, dev_fx, dev_y, dev_p), fout, protocol=4)

    #  dump into test file
    fname = "../dataset" + "/ntest_ngsum_crop.dump"
    with open(fname, 'wb') as fout:
        print(f"Saving test into {fname}")
        pickle.dump((test_x, test_y, test_p), fout, protocol=4)
      #  pickle.dump((test_xr, test_x, test_fxr, test_xr, test_y, test_p), fout, protocol=4)