import numpy as np
import pydicom
from pydicom.data import get_testdata_file
from pydicom import dcmread
import pickle
import tensorflow as tf
import random
import os
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import ImageGrid
import csv
from statistics import mean, median, mode, variance
import sys
import xml.etree.ElementTree as et
import plistlib
import matplotlib.patches as patches
import bz2
import gc
from my_lib import process_xml, compute_agatston_for_slice
import argparse
from dataGenerator import dataGenerator
import skimage

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

# import models
import models.unet as unet
import models.unetfreeze as unetfreeze


loss_choices = ("bce", "dice", "focal", "dice_n_bce", "score")
scan_type_choices = ("N", "G")
parser = argparse.ArgumentParser()
parser.add_argument("-batch_size", type=int, action='append', help="List of batch sizes")
parser.add_argument("-ddir", default="../dataset", type=str, help="Data set directory. Don't change sub-directories of the dataset")
parser.add_argument("-mdir", default="../trained_models/unet", type=str, help="Model's directory")
parser.add_argument("-mname", default="unetfreeze", type=str, help="Model's name")
parser.add_argument("-pid", default=13, type=int, help="pid to plot")
parser.add_argument("-loss", type=str, choices=loss_choices, default='dice', help=f"Pick loss from {loss_choices}")
parser.add_argument("-scan_type", type=str, choices=scan_type_choices, default='N', help=f"Pick scan type from {scan_type_choices}")
parser.add_argument("-dice_loss_fraction", default=1.0, type=float, help="Total loss is sum of dice loss and cross entropy loss. This controls fraction of dice loss to consider. Set it to 1.0 to ignore class loss")
parser.add_argument("--evaluate", action="store_true", default=False, help="Evaluate the model")
parser.add_argument("-set", type=str, choices=("train", "dev", "test"), default='dev', help="Specify set (train|dev|test) to evaluate or predict on")
parser.add_argument("--print_stats", action="store_true", default=False, help="Predict on all patients and print number of predicted calcified pixels")
parser.add_argument("--only_use_pos_images", action="store_true", default=False, help="Evaluate with positive images only")
parser.add_argument("-pmask_threshold", default=0, type=int, help="A non-zero number will filter lesion less than this area")
parser.add_argument("--print_agatston_score", action="store_true", default=False, help="Print Agaston score for actual and predicted")
args = parser.parse_args()
# User options
#os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
model_name = args.mname
batch_size = 8

# Model parameters
params = {}
params['reset_history'] = False ; # Keep this false
params['models_dir'] = args.mdir
params['loss'] = args.loss
params['print_summary'] = False
params['alpha'] = args.dice_loss_fraction ; # fraction of dice loss
params['scan_type'] = args.scan_type
ddir_scans = {"N": "deidentified_nongated",
                      "G": "Gated_release_final"}

# data set directory
ddir = args.ddir

plot_3d = True
plot_2d = False
create_100_worst_preds = False

#nongated_img = dcmread(ddir + "/" + ddir_scans[params['scan_type'] + "/" + ])
#ds = dcmread(filepath)
 #               images.append(ds.pixel_array)

# Read train, dev and test set Ids
fname = ddir + "/Ntrain_dev_pids.dump"
train_pids = []
dev_pids = []
test_pids = []
if os.path.isfile(fname):
    with open(fname, 'rb') as fin:
        print(f"Loading train/dev from {fname}")
        train_pids, dev_pids = pickle.load(fin)

fname = ddir + "/Ntest_pids.dump"
if os.path.isfile(fname):
    with open(fname, 'rb') as fin:
        print(f"Loading test from {fname}")
        test_pids = pickle.load(fin)

print (f"Total train samples {len(train_pids)}")
print (f"Total dev samples {len(dev_pids)}")
print (f"Total test samples {len(test_pids)}")

if (ddir == "../mini_dataset"):
    ddir = ddir + "/" + ddir_scans[params["scan_type"]]
elif params['scan_type'] == "G":
    ddir = ddir + "/Gated_release_final"
else:
    ddir_copy = ddir
    ddir = ddir + "/deidentified_nongated"


#model_name = 'unetfreeze'
# Load Model
if model_name == 'unet':
    model = unet.Model(None, params)
elif model_name == 'unetfreeze':
    print("here")
    model = unetfreeze.M(None, params)
else:
    model = None
    exit("Something went wrong, model not defined")

model.is_train = False
# We may not need this?
optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
model.compile(optimizer)
print (train_pids)
print (dev_pids)
print (test_pids)

if args.evaluate:
    if args.set == "train":
        print ("Evaluating on train set")
        Y_hat = model.my_evaluate(train_pids, batch_size, only_use_pos_images=args.only_use_pos_images)
    elif args.set == "dev":
        print("Evaluating on dev set")
        Y_hat = model.my_evaluate(dev_pids, batch_size, only_use_pos_images=args.only_use_pos_images)
    elif args.set == "test":
        print("Evaluating on test set")
        Y_hat = model.my_evaluate(test_pids, batch_size, only_use_pos_images=args.only_use_pos_images)
    else:
        exit(f"ERROR: Unknown {args.eval_set}")
    exit()

if False:
    def get_y(pid):
        y_hat = np.zeros((60, 512, 512))
        aaa = model.my_predict([pid], batch_size)
        if aaa.shape[0] > 60:
            y_hat[0:aaa.shape[0], :, :] = aaa[:60, :, :, 0]
        else:
            y_hat[0:aaa.shape[0], :, :] = aaa[:, :, :, 0]
        return(y_hat)
    train_inputs = []
    dev_inputs = []
    test_inputs = []
    for pid in train_pids:
        y_hat = get_y(pid)
        train_inputs.append(y_hat)
    for pid in dev_pids:
        y_hat = get_y(pid)
        dev_inputs.append(y_hat)
    for pid in test_pids:
        y_hat = get_y(pid)
        test_inputs.append(y_hat)
    train_inputs = np.array(train_inputs)
    dev_inputs = np.array(dev_inputs)
    test_inputs = np.array(test_inputs)
    print(train_inputs.shape)
    print(dev_inputs.shape)
    print(test_inputs.shape)

    fname = "../dataset/train_dev_inputs.dump"
    with open(fname, 'wb') as fout:
        print(f"Saving train/dev into {fname}")
        pickle.dump((train_inputs, dev_inputs), fout, protocol=4)

    fname = "../dataset/test_inputs.dump"
    with open(fname, 'wb') as fout:
        print(f"Saving train/dev into {fname}")
        pickle.dump((test_inputs), fout, protocol=4)

    exit()

if create_100_worst_preds:
    score = {}
    with open("D:/tiya2022/dataset/deidentified_nongated/scores.csv") as fin:
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


    train_inputs = []
    y_hats = model.my_predict_ng(train_pids, batch_size)
    for key, pid in enumerate(train_pids):
        real_score = score[str(pid)]
        train_inputs.append(abs(y_hats[key][0]-real_score))

    inputs = [(score, pid) for score, pid in zip(train_inputs, train_pids)]
    inputs = sorted(inputs)
    inputs = inputs[-100:]
    print(inputs)
    q = [_[1] for _ in inputs]
    print(q)

    fname = "../dataset/train_inputs.dump"
    with open(fname, 'wb') as fout:
        print(f"Saving train/dev into {fname}")
        pickle.dump((q), fout, protocol=4)




    exit()

if args.print_stats:
    stats = []
    for pid in dev_pids:
        Y_hat = model.my_predict([pid], batch_size)
        stats.append((pid, np.sum(Y_hat > 0.5)))

    for _ in stats:
        print (_)
    exit()


ag_scores = []
pids = [args.pid]
with open('../dataset/train_ngsum_crop.dump', 'rb') as fin:
    _, o, pids = pickle.load(fin)

pids.remove("103")
pids.remove("12")
pids.remove("192")

#pids = ["192"]
if args.print_agatston_score:
    if args.set == "train":
        pids = train_pids
    elif args.set == "dev":
        pids = dev_pids
    elif args.set == "test":
        pids = test_pids
    else:
        exit(f"ERROR: Unknown {args.eval_set}")

for pid in pids:
    # Plot original and prediction for given pid
    Y_hat = model.my_predict([pid], batch_size)
 #   print (np.sum(Y_hat[:, :, :, 0]))


    if params["scan_type"] == "G":
        filename_path = ddir + "/patient/" + str(pid) + '/'
    else:

        filename_path = ddir_copy + "/deidentified_nongated" + "/" + str(pid) + '/'
    images = []
    data_Centers_g = {}
    recons_g = {}

    for subdir, dirs, files in os.walk(filename_path):
        for filename in sorted(files, reverse=True):
            filepath = subdir + os.sep + filename
            count = 0
            if filepath.endswith(".dcm"):
                if params["scan_type"] == "N":

                    filename_path_g = ddir_copy + "/Gated_release_final" + "/patient/" + str(pid) + ''
                    #print(filename_path_g)
                    for s, d, f in os.walk(filename_path_g):
                        for fl in sorted(f, reverse=True):
                            fp = s + os.sep + fl
                            count = 0
                            if fp.endswith(".dcm"):
                                ds = dcmread(fp)
                                count += 1
                                try:
                                    #print("over here")
                                    data_Centers_g = (ds[0x0018, 0x9313].value)
                                    recons_g = (ds[0x0018, 0x9318].value)
                                    diameter = ds[0x0018, 0x1100].value
                                except:
                                    #print("here")
                                    data_Centers_g = [0.203125, -155.296875, -119.25]
                                    recons_g = [14.203125, -175.296875, -119.25]
                                    diameter = 208

                    ds = dcmread(filepath)
                    count += 1
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

                    try:
                        q = p[int(y):int(y + r * 2), int(x):int(x + r * 2)]
                        w = skimage.transform.resize(q, (512, 512), order=0, preserve_range=True)
                        images.append(w)
                    except:
                        print(pid)
                        try:
                            pids.remove(pid)
                        except:
                            continue
                        print("^ THIS DIDNT WORK")
                        continue
                    #images.append(p)

                else:
                    ds = dcmread(filepath)
                    count += 1
                    images.append(ds.pixel_array)



                 # python predict.py -mname unet -pid 14 -scan_type Gc
            #print(ds[0x0020, 0x1041].value)

    if pid in pids:
        # python predict.py -pid 14 -mname unet
        imgs = np.array(images)
        print("Radius of nongated: " + str(int(r * 2)))
        #print(imgs.shape)

        # normalization
        min_value = np.min(imgs)
        max_value = np.max(imgs)
        diff = max_value - min_value
        imgs = (imgs - min_value) / diff

        Y_hat = model.model.predict(imgs)
        #print(np.sum(Y_hat))
        #print(np.sum(Y_hat>0.5))
        images = np.array(images)
        if params['scan_type'] == "N":
            images *= (images > 0)

        # read original mdata
        fname = ddir + "/calcium_xml/" + str(pid) + (".xml")
        if os.path.exists(fname):
            mdata = process_xml(fname)
        else:
            mdata = None

        # Get True Y
        my_dg = dataGenerator([pid], batch_size, shuffle=False, ddir=args.ddir, scan_type=params["scan_type"])
        m, height, width, _ = Y_hat.shape
        if params['scan_type'] == 'G':
            Y_true = np.zeros(Y_hat.shape)
        else:
            Y_true = np.zeros((m, 1))
        X_all = np.zeros((m, height, width))
        for i in range(len(my_dg)):
            if ((i + 1) * batch_size) < Y_true.shape[0]:
                Y_true[i * batch_size : (i+1) * batch_size] = my_dg[i][1]
                X_all[i * batch_size : (i+1) * batch_size] = my_dg[i][0].reshape(-1, height, width)
            else:
                Y_true[i * batch_size : ] = my_dg[i][1]
                X_all[i * batch_size:] = my_dg[i][0].reshape(-1, height, width)

        pixel_spacing = pixel_length
        thickness = ds[0x0018, 0x0050].value

        # Compute Agatson score from true mask
        if params["scan_type"] == "G":
            ag_score_true = 0
            for X, Y in zip(X_all, Y_true):
                ag_score_true += compute_agatston_for_slice(X, Y, thickness, pixel_spacing)
        else:
            with open(ddir + "/scores.csv") as fin:
                csvreader = csv.reader(fin)
                is_header = True
                for row in csvreader:
                    if is_header:
                        is_header = False
                        continue

                    pid, lca, lad, lcx, rca, total = row
                    pid = pid.rstrip("A")
                    lca = float(lca)
                    lad = float(lad)
                    lcx = float(lcx)
                    rca = float(rca)
                    total = float(total)
                    ag_score_true = total

        # Create mdata from prediction
        #  {<image_index>: [{cid: <integer>, pixels: [(x1,y1), (x2,y2)..]},..]
        pmdata = {}
        ymdata = {}
        ## FIXME, extract the predicted cid
        ag_score_hat = 0
        for id in range(Y_hat.shape[0]):
            pred = Y_hat[id][:, :, 0] > 0.5
            ag_score_hat += compute_agatston_for_slice(images[id], pred.reshape(height, width, 1), thickness, pixel_spacing)
            Y, X = np.where(pred)
            #print(Y,X)
            if len(Y) > 0:
                pmdata[id] = []
                ttt = {'cid': 0, 'pixels': []}
                for y, x in zip(Y, X):
                    ttt['pixels'].append((x,y))
                pmdata[id].append(ttt)

        print("AG SCORE: ")
        #print(ag_score_hat)
        print(ag_score_hat*((r*2)**2)/(512**2))
        ag_scores.append((pid, ag_score_true, ag_score_hat))
        if args.print_agatston_score:
            continue


        if params["scan_type"] == "G":
            for id in range(Y_hat.shape[0]):
                Y, X = np.where(Y_true[id][:, :, 0] > 0)
                if len(Y) > 0:
                    ymdata[id] = []
                    ttt = {'cid': 0, 'pixels': []}
                    for y, x in zip(Y, X):
                        ttt['pixels'].append((x,y))
                    ymdata[id].append(ttt)
        else:
            mdata = None


        # plot
        pixel_colors = {0: 'red',
                        1: 'blue',
                        2: 'green',
                        3: 'yellow'}

        def add_patches(ax, mdata, is_predict=False):
            ax.patches = []
            if mdata and plot_cid in mdata:
                for roi in mdata[plot_cid]:
                    if (is_predict):
                        for p in roi['pixels']:
                            ax.add_patch(patches.Circle(p, radius=1, color=pixel_colors[roi['cid']]))
                    else:
                        ax.add_patch(patches.Polygon(roi['pixels'], closed=True, color=pixel_colors[roi['cid']]))
        #print(images[0].shape)
        if plot_3d:
            def previous_slice(ax):
                """Go to the previous slice."""
                global plot_cid
                volume = ax[0].volume
                n = volume.shape[0]
                plot_cid = (plot_cid - 1) % n  # wrap around using %
                for i in range(2):
                    ax[i].images[0].set_array(volume[plot_cid])
                    ax[i].set_title(f"Image {plot_cid}")
                add_patches(ax[0], mdata)
                add_patches(ax[1], pmdata, is_predict=True)


            def next_slice(ax):
                """Go to the next slice."""
                global plot_cid
                volume = ax[0].volume
                n = volume.shape[0]
                plot_cid = (plot_cid + 1) % n
                for i in range(2):
                    ax[i].images[0].set_array(volume[plot_cid])
                    ax[i].set_title(f"Image {plot_cid}")
                add_patches(ax[0], mdata)
                add_patches(ax[1], pmdata, is_predict=True)


            def process_key(event):
                fig = event.canvas.figure
                ax = fig.axes
                if event.key == 'p':
                    previous_slice(ax)
                elif event.key == 'n':
                    next_slice(ax)
                fig.canvas.draw()


            def multi_slice_viewer(volume):
                global plot_cid
                fig, ax = plt.subplots(1, 2)
                ax[0].volume = volume
                plot_cid = volume.shape[0] // 2
                img = volume[plot_cid]
                for i in range(2):
                    ax[i].imshow(img, cmap="bone")
                    ax[i].set_title(f"Image {plot_cid}")
                add_patches(ax[0], mdata)
                add_patches(ax[1], pmdata, is_predict=True)
                fig.canvas.mpl_connect('key_press_event', process_key)


            multi_slice_viewer(images)
            plt.show()
        elif plot_2d:
            fig = plt.figure(figsize=(512, 512))
            s = int(images.shape[0] ** 0.5 + 0.99)
            grid = ImageGrid(fig, 111, nrows_ncols=(s, s))
            #print(images.shape[0])
            for i in range(images.shape[0]):
                ax = grid[i]
                ax.imshow(images[i, :, :], cmap=plt.cm.bone)

            plt.show()


if args.print_agatston_score:
    for ag_score in ag_scores:
        print (",".join([str(_) for _ in ag_score]))
exit()
