
import numpy as np
import pydicom
from pydicom.data import get_testdata_file
from pydicom import dcmread
import pickle
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
from my_lib import process_xml
import argparse

# get model
import models.unet as unet

# User options
parser = argparse.ArgumentParser()
parser.add_argument("-ddir", default="../dataset", type=str, help="Data set directory. Don't change sub-directories of the dataset")
parser.add_argument("-pid", default=0, type=int, help="pid to plot")
parser.add_argument("-scan_type", default="N", type=str, help="scan type (G|N)")
args = parser.parse_args()

doPlot = False
plot3D = False
pid = random.choice([i for i in range(100)]) # specify the patient id
pid = 1 # args.pid
batch_size = 8
print (pid)
sdir_id = args.scan_type ;# G for gated, N for non-gated
generate_gated_train_dev_test_set = True
train_set_size = 0.8
dev_set_size = 0.1
test_set_size = 0.1

# data path directory
#ddir = "../dataset/cocacoronarycalciumandchestcts-2"
ddir = args.ddir
sdirs = {"G" : "Gated_release_final",
           "N" : "deidentified_nongated"}

# output data directory where pickle objects will be dumped.
odir = "../dataset"

# set the random seed to create the same train/val/test split
np.random.seed(10015321)
debug = 2

# myprint function
def myprint(x, level):
    if (level < debug):
        print (x)

# directory structure
# G/calcium_xml/<id>
# G/patient/<id>/*/*.dcm
# NG/<id>/<id>/*.dcm
# NG/scores (csv)

# read all images of a given patient
# this is to plot CT for a given patient
# for 3D plots:
    # p -> to go to previous slice
    # n -> to go to next slice

images = []

# pixel colors
pixel_colors = {0: 'red',
                1: 'blue',
                2: 'green',
                3: 'yellow'}

# function to add patches from mdata on given matplot Axes
plot_cid = 0  # global variable to remember image index that is currently plotted




data = {}
# {G: {<pid>: {images: [], mdata: {}},
# {N: {<pid>: {images: [], mdata: {}}}

# to track progress
total_work = 0
progress_count = 0

def tick():
    global progress_count, total_work
    progress_count += 1
    if (progress_count%64 == 0):
        p = int(progress_count*100/total_work)
        sys.stdout.write(f"\r{p}%")
        sys.stdout.flush()

if sdir_id == "G":
    # Process gated CTs
    k = "G"
    data[k] = {}
    sdir = f"{ddir}/{sdirs[k]}/Patients"
    myprint(f"Processing {sdir} folder", 1)

    # estimate total work
    total_work = 0
    for subdir, dirs, files in os.walk(sdir):
        total_work += len(files)

    progress_count = 0
    for subdir, dirs, files in os.walk(sdir):
        images_indices = []
        for filename in sorted(files, reverse=True):
            tick()
            filepath = str.replace(subdir, "\\", "/") + "/" + filename
            image_index = filename.split(".")[0].split("-")[-1]
            if image_index in images_indices:
                print (f" duplicate images in {filepath}")
                break
            images_indices.append(image_index)
            myprint(f"Processing {filepath}", 4)
            if filepath.endswith(".dcm"):
                pid = filepath.split("/")[-3]
                if (not pid in data[k]):
                    data[k][pid] = {}
                    data[k][pid]['images'] = []
                data[k][pid]['images'].append(dcmread(filepath))

    sdir = f"{ddir}/{sdirs[k]}/calcium_xml"
    myprint(f"Processing {sdir} folder", 1)

    # estimate total work
    total_work = 0
    for subdir, dirs, files in os.walk(sdir):
        total_work += len(files)

    progress_count = 0
    for subdir, dirs, files in os.walk(sdir):
        for filename in files:
            tick()
            filepath = str.replace(subdir, "\\", "/") + "/" + filename
            myprint(f"Processing {filepath}", 4)
            if filepath.endswith(".xml"):
                pid = filepath.split("/")[-1].split(".")[0]
                if (pid in data[k]):
                    data[k][pid]['mdata'] = process_xml(filepath)
                else:
                    print (f"WARNING: {pid}.xml found but no matching images")
else:

    # process non-gated CTs
    k = "N"
    data[k] = {}
    sdir = f"{ddir}/{sdirs[k]}"
    myprint(f"Processing {sdir}/{sdir} folder", 1)

    # estimate total work
    total_work = 0
    for subdir, dirs, files in os.walk(sdir):
        total_work += len(files)

    progress_count = 0
    for subdir, dirs, files in os.walk(sdir):
         for filename in sorted(files, reverse=True):
            tick()
            filepath = str.replace(subdir, "\\", "/") + "/" + filename
            myprint(f"Processing {filepath}", 4)
            if filepath.endswith(".dcm"):
                pid = filepath.split("/")[-3]
                if (not pid in data[k]):
                    data[k][pid] = {}
                    data[k][pid]['images'] = []
                data[k][pid]['images'].append(dcmread(filepath))

    # process non-gated score file
    with open(sdir + "/scores.csv") as fin:
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
            # FIXME: hmm, what is total?
            #sum = lca + lad + lcx + rca
            #assert(total > sum - 1 and total < sum + 1), f"TOTAL doesn't match ({total} != {lca} + {lad} + {lcx} + {rca})"
            if (pid in data[k]):
                data[k][pid]['mdata'] = [lca, lad, lcx, rca, total]
            else:
                print(f"WARNING: {pid}.xml found but no matching images")


# train/test/dev split
all_pids = []
for pid in data[sdir_id].keys():
    if sdir_id == "G" and (pid == "159" or pid == "238" or pid == "398" or pid == "415" or pid == "421"):
        continue
    elif sdir_id == "N" and (pid == "103"):
        continue
    else:
        all_pids.append(pid)

if generate_gated_train_dev_test_set:
    # reshuffle
    print (f"Splitting gated data set into train/dev/test as {train_set_size}/{dev_set_size}/{test_set_size}")
    random.shuffle(all_pids)
    total_pids = len(all_pids)
    tidx = int(total_pids*train_set_size)
    didx = int(total_pids*(train_set_size + dev_set_size))
    train_pids = all_pids[0:tidx]
    dev_pids = all_pids[tidx:didx]
    test_pids = all_pids[didx:]

    # dump into train/dev file
    fname = odir + "/" + sdir_id + "train_dev_pids.dump"
    with open(fname, 'wb') as fout:
        print (f"Saving train/dev into {fname}")
        pickle.dump((train_pids, dev_pids), fout, protocol=4)

    fname = odir + "/" + sdir_id + "test_pids.dump"
    with open(fname, 'wb') as fout:
        print(f"Saving test into {fname}")
        pickle.dump(test_pids, fout, protocol=4)

    # print stats
    for pids, name in zip((train_pids, dev_pids, test_pids), ("train", "dev", "test")):
        num_positive_samples = 0
        num_positive_slices = 0
        num_total_slices = 0
        for pid in pids:
            num_total_slices += len(data[sdir_id][pid]['images'])
            if ('mdata' in data[sdir_id][pid]):
                if len(data[sdir_id][pid].keys()) > 0:
                    num_positive_samples += 1
                for key in data[sdir_id][pid].keys():
                    num_positive_slices += 1
        print(f"{name} statistics: ")
        print(f"  Number of patients = {len(pids)}")
        print(f"  Positive samples = {num_positive_samples}")
        print(f"  Total slices = {num_total_slices}")
        print(f"  Total positive slices = {num_positive_slices}")