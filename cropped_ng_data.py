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


# User options
loss_choices = ("bce", "dice", "focal", "dice_n_bce", "score")
scan_type_choices = ("N", "G")
parser = argparse.ArgumentParser()
parser.add_argument("-ddir", default="../dataset", type=str, help="Data set directory. Don't change sub-directories of the dataset")
parser.add_argument("-pid", default=0, type=int, help="pid to plot")
parser.add_argument("-scan_type", default=0, type=str, help="scan type (G|N)")
parser.add_argument("-mdir", default="../trained_models/unet", type=str, help="Model's directory")
parser.add_argument("-mname", default="unet", type=str, help="Model's name")
parser.add_argument("-loss", type=str, choices=loss_choices, default='dice', help=f"Pick loss from {loss_choices}")
args = parser.parse_args()

batch_size = 8
# Model parameters
model_name = args.mname

params = {}
params['reset_history'] = False ; # Keep this false
params['models_dir'] = args.mdir
params['loss'] = args.loss
params['print_summary'] = False

generate_gated_train_dev_test_set = True
sdir_id = "N"
train_set_size = 0.8
dev_set_size = 0.1

# data path directory
ddir = args.ddir
sdirs = {"G" : "Gated_release_final",
           "N" : "deidentified_nongated"}

# output data directory where pickle objects will be dumped.
odir = "../dataset"

# set the random seed to create the same train/val/test split
np.random.seed(10015321)
debug = 2

data = {}
full_image = {}

images = {}
seg_info = {}
# process non-gated CTs
k = "N"
data[k] = {}
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

#print(pids_final[0:int(xs.shape[0] * train_set_size)])

#pids_final = ["1"]
exceptions = [102, 103, 106, 108, 109, 4, 41, 111, 113, 114, 65, 66, 67, 68, 7, 70, 72, 74, 76, 77, 78,79, 8,80, 82, 84, 85, 87, 88, 9, 92,98,99, 115, 117, 118, 120, 39, 123, 125, 127,42, 45,47,52, 55,60,61,63,64, 128, 13, 130, 131, 134, 135, 137, 141, 143, 144, 146, 147, 148, 15, 152, 153, 154, 159, 16, 161, 163, 164, 19, 190, 195, 203, 211, 214, 22, 23, 28, 32, 33, 35, 36, 167, 168, 170, 172, 174, 178, 179, 181, 186, 182, 183, 184, 185, 192]
print(sdir)
for subdir, dirs, files in os.walk(sdir):
    for filename in sorted(files, reverse=True):
        filepath = str.replace(subdir, "\\", "/") + "/" + filename
        if filepath.endswith(".dcm"):
            pid = filepath.split("/")[-3]
            if (not pid in data[k]):
                data[k][pid] = {}
                data[k][pid]['images'] = []
                full_image[pid] = []
            if int(pid) in exceptions:
                ds = dcmread(filepath)
                data[k][pid]['images'].append(ds.pixel_array)
                full_image[pid].append(ds.pixel_array)
                continue
            else:
                filename_path_g = args.ddir + "/Gated_release_final" + "/patient/" + str(pid) + ''
                for s, d, f in os.walk(filename_path_g):
                    for fl in sorted(f, reverse=True):
                        fp = s + os.sep + fl
                        ds = dcmread(fp)
                        data_Centers_g = (ds[0x0018, 0x9313].value)
                        recons_g = (ds[0x0018, 0x9318].value)
                        diameter = ds[0x0018, 0x1100].value

                ds = dcmread(filepath)

                pixel_length = ds[0x0028, 0x0030].value[0]
                recon = ds[0x0018, 0x9318].value
                subtract_x = ds[0x0018, 0x9313].value[0] - data_Centers_g[0]
                subtract_y = ds[0x0018, 0x9313].value[1] - data_Centers_g[1]
                r = int((diameter / 2) / pixel_length)

                sub_pixel_x = (recons_g[0] + subtract_x - recon[0]) / pixel_length
                sub_pixel_y = (recons_g[1] + subtract_y - recon[1]) / pixel_length
                x = 256 + sub_pixel_x - r
                y = 256 + sub_pixel_y - r
                p = ds.pixel_array

                q = p[int(y):int(y + r * 2), int(x):int(x + r * 2)]
                w = skimage.transform.resize(q, (512, 512), order=0, preserve_range=True)
                data[k][pid]['images'].append(w)
                q = p[:, :]
                w = skimage.transform.resize(q, (512, 512), order=0, preserve_range=True)
                full_image[pid].append(w)


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
        if (str(pid) in data[k]):
            data[k][pid]['mdata'] = total
        else:
            print(f"WARNING: {pid}.xml found but no matching images")

final_segs = []
xs = []
ys = []
xs_full = []

for pid in pids_final:
    if int(pid) == 103:
        continue
    else:
        a = np.array(data[k][pid]['images'])
        c = np.array(full_image[pid])
        print(pid)
        print(a.shape)
        print(c.shape)

        if a.shape[0] > 60:
            a = a[:60, :, :]
            c = c[:60, :, :]
        else:
            b = np.zeros((60, 512, 512), dtype=np.uint16)
            d = np.zeros((60, 512, 512), dtype=np.uint16)
            b[0:a.shape[0], :, :] = a[:,:,:]
            d[0:c.shape[0], :, :] = c[:,:,:]
            a = b
            c = d
        a = np.reshape(a, (60, 512, 512, 1))
        c = np.reshape(c, (60, 512, 512, 1))
        xs.append(a)
        xs_full.append(c)
#        print(data[k][pid]['mdata'])
        ys.append(data[k][pid]['mdata'])

xs = np.array(xs)
ys = np.array(ys)
xs_full = np.array(xs_full)
print(xs.shape, ys.shape, xs_full)


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

x_masks = []
x_masks_full = []
p = []
for i, pid in enumerate(pids_final):
    print (f"Predicting on pid = {pid}")
    try:
        a = model.model.predict(xs[i])
        b = model.model.predict(xs_full[i])
        x_masks.append(a[:,:,:,0])
        x_masks_full.append(b[:,:,:,0])
        if pid in p:
            continue
        else:
            p.append(pid)
    except:
        continue

# cropped image
xs_real = xs
# cropped image mask
xs = x_masks
# full image
xs_full = xs_full
# full image mask
xs_full_mask = x_masks_full

del x_masks, i, pid, data[k], final_segs, a, b,c,pids_final, full_image, data, images, x_masks_full, model
xs = np.array(xs)
xs_full_mask = np.array(xs_full_mask)
print(xs.shape)
print(xs_full_mask.shape)
xs = np.transpose(xs, (0, 2, 3, 1))
xs_full_mask = np.transpose(xs_full_mask, (0, 2, 3, 1))

print("Shapes of final")
print(xs_real.shape, xs.shape, xs_full.shape, xs_full_mask.shape, ys.shape)

if generate_gated_train_dev_test_set:
    tidx = int(xs.shape[0] * train_set_size)
    didx = int(xs.shape[0] * (train_set_size + dev_set_size))
    print(tidx)
    print(didx)
    train_x = xs[0:tidx, :, :, :]
    dev_x = xs[tidx:didx, :, :, :]
    test_x = xs[didx:, : , :, :]

    train_xr = xs_real[0:tidx, :, :, :]
    dev_xr = xs_real[tidx:didx, :, :, :]
    test_xr = xs_real[didx:, :, :, :]
    train_fxr = xs_full[0:tidx, :, :, :]
    dev_fxr = xs_full[tidx:didx, :, :, :]
    test_fxr = xs_full[didx:, :, :, :]
    train_fx = xs_full_mask[0:tidx, :, :, :]
    dev_fx = xs_full_mask[tidx:didx, :, :, :]
    test_fx = xs_full_mask[didx:, :, :, :]
    train_y = ys[0:tidx]
    dev_y = ys[tidx:didx]
    test_y = ys[didx:]
    train_p = p[0:tidx]
    dev_p = p[tidx:didx]
    test_p = p[didx:]


    print(train_x.shape, dev_x.shape, test_x.shape)
    print(train_y.shape, dev_y.shape, test_y.shape)
    print(train_p)
    print(dev_p)
    print(test_p)

    # dump into train file
    fname = odir + "/train_ng_crop.dump"
    with open(fname, 'wb') as fout:
        print (f"Saving train into {fname}")
        pickle.dump((train_xr, train_x, train_y, train_p), fout, protocol=4)
#        pickle.dump((train_xr, train_x, train_fxr, train_fx, train_y, train_p), fout, protocol=4)

    #  dump into dev file
    fname = odir + "/dev_ng_crop.dump"
    with open(fname, 'wb') as fout:
        print (f"Saving dev into {fname}")
        pickle.dump((dev_xr, dev_x, dev_y, dev_p), fout, protocol=4)
      #  pickle.dump((dev_xr, dev_x, dev_fxr, dev_fx, dev_y, dev_p), fout, protocol=4)

    #  dump into test file
    fname = odir + "/test_ng_crop.dump"
    with open(fname, 'wb') as fout:
        print(f"Saving test into {fname}")
        pickle.dump((test_xr, test_x, test_y, test_p), fout, protocol=4)
      #  pickle.dump((test_xr, test_x, test_fxr, test_xr, test_y, test_p), fout, protocol=4)