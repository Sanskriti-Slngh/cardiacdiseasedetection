import numpy as np
from pydicom import dcmread
import pickle
import os
import matplotlib.pyplot as plt
import csv
from my_lib import process_xml
import argparse
import skimage

# get model
import models.unet as unet

# User options
parser = argparse.ArgumentParser()
parser.add_argument("-ddir", default="../dataset", type=str, help="Data set directory. Don't change sub-directories of the dataset")
parser.add_argument("-pid", default=0, type=int, help="pid to plot")
parser.add_argument("-scan_type", default=0, type=str, help="scan type (G|N)")
args = parser.parse_args()

generate_gated_train_dev_set = True
sdir_id = "N"
generate_gated_train_dev_test_set = True
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

# myprint function
def myprint(x, level):
    if (level < debug):
        print (x)

from matplotlib.path import Path

segmentations_fn = "D:/tiya2022/data/ngseg.csv"
pids = []
segs = {}
s = csv.reader(open(segmentations_fn))
count = 0
for row in s:
    if count == 0:
        count = 1
        continue
    else:
        if row[0] in pids:
            print("repeat")
        else:
            pids.append(row[0])
            segs[row[0] + "x"] = []
            segs[row[0] + "y"] = []
            segs[row[0] + "slice_g"] = []
            segs[row[0] + "slice_ng"] = []

        segs[row[0] + 'x'].append(int(row[1]))
        segs[row[0] + 'y'].append(int(row[2]))
        segs[row[0] + 'slice_g'].append(int(row[3]))
        segs[row[0] + 'slice_ng'].append(int(row[4])+1)

# OVERIDE DELETE LATER


data = {}

# process non-gated CTs
k = "N"
data[k] = {}
sdir = f"{ddir}/{sdirs[k]}"
myprint(f"Processing {sdir}/{sdir} folder", 1)

print(pids)
#pids = ["3"]
images = {}
seg_info = {}
for pid in pids:
    print("PID: " + pid)
    count = 0

    filename_path = sdir +"/" + str(pid) + '/' + str(pid) + '/'
    for subdir, dirs, files in os.walk(filename_path):
        for filename in sorted(files, reverse=True):
            filepath = subdir + os.sep + filename
            count += 1

            if filepath.endswith(".dcm"):
                if (not pid in data[k]):
                    data[k][str(pid)] = {}
                if True:
                    count2 = 0
                    filename_path_g = args.ddir + "/Gated_release_final" + "/patient/" + str(pid) + ''
                    for s, d, f in os.walk(filename_path_g):

                        for fl in sorted(f, reverse=True):
                            count2 +=1
                            fp = s + os.sep + fl
                            if fp.endswith(".dcm"):
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

                    w = np.zeros((512,512))
                    if count in segs[pid + "slice_ng"]:
                        print(count, segs[pid + "slice_ng"])
                        a = segs[pid + "slice_ng"].index(count)
                        #q = p[int(y + segs[pid + "y"][a]*r/512):int(y + r * 2+ segs[pid + "y"][a]*r/512), int(x + segs[pid + "x"][a]*r/512):int(x + r * 2 + segs[pid + "x"][a]*r/512)]
                        q = p[int(y):int(y + r * 2),
                            int(x):int(x + r * 2)]
                        w = skimage.transform.resize(q, (512, 512), order=0, preserve_range=True)

                    data[k][pid][count] = (w)
            else:
                print(filepath)
                exit()
      #  print(count, count2)
        seg_info[pid] = [count, count2]


datas = {}
sdir = f"{ddir}/{sdirs['G']}/calcium_xml"
progress_count = 0
count = 0
for subdir, dirs, files in os.walk(sdir):
    for filename in files:
        filepath = str.replace(subdir, "\\", "/") + "/" + filename
        if filepath.endswith(".xml"):
            pid = filepath.split("/")[-1].split(".")[0]

            if (not pid in datas):
                datas[pid] = {}

            if pid in data[k]:
                _ = process_xml(filepath)
                for slice in range(seg_info[pid][1]):
                    mask = np.zeros((512, 512))
                    print(pid, slice)
                    #y_mask = np.zeros((seg_info[int(pid)][0], 512, 512, 1))
                    if slice in _:
                        for aaa in _[slice]:
                            pixels = aaa['pixels']
                            if slice in segs[str(pid)+"slice_g"]:
                                a = segs[str(pid) + "slice_g"].index(slice)
                                new = []
                                for x, y in pixels:
                                    print("x" + str(x) +
                                    ", Y: " + str(y))
                                    x = x + segs[str(pid) + "x"][a]
                                    y = y + segs[str(pid) + "y"][a]
                                    new.append((x,y))
                                    print("x" + str(x) +
                                    ", Y: " + str(y))
                                pixels = new


                            poly_path = Path(pixels)
                            y, x = np.mgrid[:512, :512]
                            coors = np.hstack((x.reshape(-1, 1), y.reshape(-1, 1)))
                            mask += poly_path.contains_points(coors).reshape(512, 512)
                            for x, y in pixels:
                                mask[y, x] = 1

                    datas[pid][slice] = mask

final_segs = []
xs = []
ys = []
for pid in pids:
    for i in segs[str(pid) + "slice_g"]:
        a = segs[str(pid) + "slice_g"].index(i)
        xs.append(data[k][str(pid)][segs[str(pid) + "slice_ng"][a]])

        #move y_mask
        new_mask = np.zeros((512,512))
        img = datas[str(pid)][segs[str(pid) + "slice_g"][a]]
        y,x = np.where(img)
        for x,y in zip(x,y):
            new_mask[x-segs[str(pid) + "x"][a], y-segs[str(pid) + "y"][a]] = 1

        ys.append(datas[str(pid)][segs[str(pid) + "slice_g"][a]])

xs = np.array(xs)
ys = np.array(ys)
print(xs.shape, ys.shape)
xs = np.reshape(xs, (xs.shape[0], 512, 512, 1))
ys = np.reshape(ys, (ys.shape[0], 512, 512, 1))
print(xs.shape, ys.shape)

p = 0
plot = False
if plot:
    for i in range(10):

        img = xs[p, :, :, :]
        img_mask = ys[p, :, :, :]
        imgplot = plt.imshow(img, cmap='gray')
        plt.imshow(img_mask, cmap='jet', alpha=0.5)
        plt.show()
        print(img.sum())
        print(img_mask.sum())
        p = p + 1


if generate_gated_train_dev_set:
    print(f"goign here = {len(pids)}")
    tidx = int(xs.shape[0] * train_set_size)
    print(tidx)
    train_x = xs[0:tidx, :, :, :]
    dev_x = xs[tidx:, :, :, :]
    train_y = ys[0:tidx, :, :, :]
    dev_y = ys[tidx:, :, :, :]

    print(train_x.shape, dev_x.shape)
    print(train_y.shape, dev_x.shape)

    # dump into train/dev file
    fname = odir + "/train_ng.dump"
    with open(fname, 'wb') as fout:
        print (f"Saving train into {fname}")
        pickle.dump((train_x, train_y), fout, protocol=4)

    #  dump into train/dev file
    fname = odir + "/dev_ng.dump"
    with open(fname, 'wb') as fout:
        print (f"Saving dev into {fname}")
        pickle.dump((dev_x, dev_y), fout, protocol=4)