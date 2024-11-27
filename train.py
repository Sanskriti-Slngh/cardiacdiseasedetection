import argparse
import datetime
import numpy as np
import pydicom
from pydicom.data import get_testdata_file
from pydicom import dcmread
import pickle
import tensorflow as tf
import random
import os
import pytz
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
from dataGenerator_ng import dataGenerator_ng

# import models
import models.unet as unet
import models.unetfreeze as unetfreeze
import models.ng_unet_retrain as ngunet
import models.unetcrop as unetcrop
import models.unetfreezecroptwo as unetfreezecroptwo
import models.deepunetfreeze as deepunetfreeze
import models.unetscore as unetsum

loss_choices = ("bce", "dice", "focal", "dice_n_bce", "score")
scan_type_choices = ("N", "G")

parser = argparse.ArgumentParser()
parser.add_argument("-batch_size", type=int, action='append', help="List of batch sizes")
parser.add_argument("-epochs", default=100, type=int)
parser.add_argument("-max_train_patients", default=None, type=int, help="To limit number of training examples")
parser.add_argument("-dice_loss_fraction", default=1.0, type=float, help="Total loss is sum of dice loss and cross entropy loss. This controls fraction of dice loss to consider. Set it to 1.0 to ignore class loss")
parser.add_argument("-upsample_ps", default=None, type=int, help="Non zero value to enable up-sampling positive samples during training")
parser.add_argument("-ddir", default="../dataset", type=str, help="Data set directory. Don't change sub-directories of the dataset")
parser.add_argument("-patient_splits_dir", type=str, help="Directory in which patient splits are located.", default=None)
parser.add_argument("-mdir", default="../trained_models/unet", type=str, help="Model's directory")
parser.add_argument("-mname", default="unetfreeze", type=str, help="Model's name")
parser.add_argument("--plot", action="store_true", default=False, help="Plot the metric/loss")
parser.add_argument("--train", action="store_true", default=False, help="Train the model")
parser.add_argument("--hsen", action="store_true", default=False, help="Generate random hyper parameters")
parser.add_argument("-lr", help="List of learning rates", action='append', type=float)
parser.add_argument("-steps_per_epoch", default=None, type=int, help="Number of steps per epoch. Set this to increase the frequency at which Tensorboard reports eval metrics. If None, it will report eval once per epoch.")
parser.add_argument("-model_save_freq_steps", default=40, type=int,
                    help="Save the model at the end of this many batches. If low,"
                    "can slow down training. If none, save after each epoch.")
parser.add_argument("-loss", type=str, choices=loss_choices, default='score', help=f"Pick loss from {loss_choices}")
parser.add_argument("--reset", default=False, action="store_true", help="To reset model")
parser.add_argument("--only_use_pos_images", action="store_true", default=False, help="Train with positive images only")
parser.add_argument("--use_dev_pos_images", action="store_true", default=False, help="Evaluate only on positive samples on dev set")
parser.add_argument("--den", action="store_true", default=False, help="Enable data augmentation")
parser.add_argument("-num_neg_images_per_batch", default=0, type=int, help="Number of positive images to be replaced with neg images per batch. Use with --only_use_pos_images")
parser.add_argument("-scan_type", type=str, choices=scan_type_choices, default='N', help=f"Pick scan type from {scan_type_choices}")
parser.add_argument("-ng_seg", type=bool, default=False, help="Are we performing unet training on nongated scans?")
args = parser.parse_args()

TIME_FORMAT = "%Y-%m-%d-%H-%M"

def get_time():
  return datetime.datetime.now(pytz.timezone('US/Pacific'))
start_time = get_time()
print(f'Launched at {start_time}')

# User options
#os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
#os.environ["TF_GPU_ALLOCATOR"]='cuda_malloc_async'
model_name = args.mname
use_adam = True
if args.lr:
    learning_rates = args.lr
else:
    learning_rates = [0.01]
momentum = 0.9
batch_sizes = args.batch_size
epochs = args.epochs
plot = args.plot
train = args.train

# Model parameters
params = {}
params['resetHistory'] = args.reset
params['print_summary'] = False
params['dropout'] = 0
params['data_aug_enable'] = args.den
params['models_dir'] = args.mdir
params['upsample_ps'] = args.upsample_ps ; # set non-zero integer to up-sample positive samples
params['limit_pids'] = args.max_train_patients
params['alpha'] = args.dice_loss_fraction ; # fraction of dice loss
params['ddir'] = args.ddir
params['steps_per_epoch'] = args.steps_per_epoch
params['model_save_freq_steps'] = args.model_save_freq_steps
params['loss'] = args.loss
params['only_use_pos_images'] = args.only_use_pos_images
params['use_dev_pos_images'] = args.use_dev_pos_images
params['num_neg_images_per_batch'] = args.num_neg_images_per_batch
params["scan_type"] = args.scan_type
params["ng_seg"] = args.ng_seg

# Hyper parameter search
if args.hsen:
    #learning_rates = [10**random.uniform(-2,-5)]
    #print (learning_rates)
    params['alpha'] = random.uniform(0.8, 1.0)
    print (f"Using alpha as {params['alpha']}")

if args.patient_splits_dir is None:
  patient_splits_dir = args.ddir
else:
  patient_splits_dir = args.patient_splits_dir

# Read train, dev and test set Ids

fname = os.path.join(patient_splits_dir, params["scan_type"] + "train_dev_pids.dump")
with open(fname, 'rb') as fin:
    print(f"Loading train/dev from {fname}")
    train_pids, dev_pids = pickle.load(fin)

fname = os.path.join(patient_splits_dir, params["scan_type"] + "test_pids.dump")
with open(fname, 'rb') as fin:
    print(f"Loading test from {fname}")
    test_pids = pickle.load(fin)

#print (train_pids)
#print (dev_pids)

print (f"Total train samples {len(train_pids)}")
print (f"Total dev samples {len(dev_pids)}")
print (f"Total test samples {len(test_pids)}")

# LossHistory Class
class LossHistory(tf.keras.callbacks.Callback):
    def __init__(self):
        self.train_losses = []
        self.val_losses = []
        self.train_seg_f1 = []
        self.val_seg_f1 = []
        self.train_class_acc = []
        self.val_class_acc = []
        self.acc_epochs = 0
        super(LossHistory, self).__init__()

    def on_epoch_end(self, epoch, logs):
        self.train_losses.append(logs.get('loss'))
        self.val_losses.append(logs.get('val_loss'))
        self.train_seg_f1.append(logs.get('seg_f1'))
        self.val_seg_f1.append(logs.get('val_seg_f1'))
        self.train_class_acc.append(logs.get('class_acc'))
        self.val_class_acc.append(logs.get('val_class_acc'))

        # Re-shuffle
        #indices = [i for i in range(len(model.xseg_train))]
        #random.shuffle(indices)
        #xxx = [model.xseg_train[i] for i in indices]
        #yyy = [model.yseg_train[i] for i in indices]
        #model.xseg_train = np.array(xxx)
        #model.yseg_train = np.array(yyy)
        #del xxx, yyy

        gc.collect()
        if args.model_save_freq_steps and epoch % args.model_save_freq_steps == 0:
            # Save model
            print ("Saving the model in ../experiments/current/m_" + str(epoch))
            model.save('../experiments/current/m_' + str(epoch))
            #print ("Full evaluation on dev set")
            #model.my_evaluate(dev_pids, batch_size, only_use_pos_images=False)

# learning rate scheduler
def lr_scheduler(epoch, lr):
    if epoch < 50:
        return lr
    return lr * tf.math.exp(-0.1)

## Load Model and run training
for batch_size in batch_sizes:
    for lr in learning_rates:
        if train or 'models_dir' not in params:
            params['models_dir'] = params['models_dir']
        history = LossHistory()
        if model_name == 'unet':
            model = unet.Model(history, params)
        elif model_name == "unetfreeze":
            model = unetfreeze.M(history, params)
        elif model_name == "ngunet":
            model = ngunet.Model(history, params)
        elif model_name == "unetcrop":
            model = unetcrop.Model(history, params)
        elif model_name == "unetcroptwo":
            model = unetfreezecroptwo.M(history, params)
        elif model_name == "deepunetfreeze":
            model = deepunetfreeze.M(history, params)
        elif model_name == "unetsum":
            model = unetsum.Model(history, params)
        else:
            model = None
            exit("Something went wrong, model not defined")

        if not train:
            model.is_train = False

        ## training
        if use_adam:
            optimizer = tf.keras.optimizers.Adam(learning_rate=lr)
        else:
            optimizer = tf.keras.optimizers.SGD(learning_rate=lr, momentum=momentum)

        model.compile(optimizer)

        #Set the data
        model.train_pids = train_pids
        model.dev_pids = dev_pids
        model.test_pids = test_pids

        # instantiate model
        if train:
            if params["ng_seg"]:
                model.train_ng_seg(batch_size, epochs, lr_scheduler)
                model.model.evaluate(model.xseg_train, model.yseg_train, batch_size)
            elif model_name == "unetfreeze" or model_name == "unetcroptwo" or model_name == "deepunetfreeze":
                print("LOL")
                model.train_crop(batch_size, epochs)
            elif model_name == "unetsum":
                model.train_ng_sum(batch_size, epochs, lr_scheduler)
            else:
                print("HERE")
                model.train_notcrop(batch_size, epochs)
                model.model.evaluate(model.xseg_train, batch_size)
            #model.save("cropped_unet_freeze")
            model.save()
            #y_hat = model.my_evaluate(train_pids, batch_size, only_use_pos_images=args.only_use_pos_images)
        elif not plot:
            pids = ['40', '41'] # '42', '43', '44', '45', '46', '47', '48', '49', '5', '51', '52', '53', '54', '55', '56', '57', '58']
                #['10', '100', '101', '102', '103', '104', '105', '106', '107', '108', '109', '11', '110', '111', '112', '113', '114', '115', '116', '117', '118', '119', '12', '120', '121', '122', '123', '124', '125', '126', '127', '128', '129', '13', '130', '131', '132', '133', '134', '135', '136', '137', '138', '139', '14', '140', '141', '142', '143', '144', '145', '146', '147', '148', '149', '15', '150', '151', '152', '153', '154', '155', '156', '157', '158', '159', '16', '160', '161', '162', '163', '164', '165', '166', '167', '168', '169', '17', '170', '171', '172', '173', '174', '175', '176', '177', '178', '179', '18', '180', '181', '182', '183', '184', '185', '186', '187', '188', '189', '19', '190', '191', '192', '193', '194', '195', '196', '197', '198', '199', '2', '20', '200', '201', '202', '203', '204', '205', '206', '207', '208', '209', '21', '210', '211', '212', '213', '214', '22', '23', '24', '25', '26', '27', '28', '29', '3', '30', '31', '32', '33', '34', '35', '36', '37', '38', '39', '4']
            y_hats = model.get_prediction_data(pids, batch_size, "G")
#            model.new_predict()

       # my_dg = dataGenerator_ng([131], 8,
        #                            data="train"
         #                           )
        #X = my_dg[0][0]
        #print (model.model.predict(X))

        #print (model.model.weights)
        if plot:
            fig, ax = plt.subplots(nrows=1, ncols=3)
            model.train_plot(fig, ax, show_plot=False)

end_time = get_time()
elapsed_time = (end_time - start_time).total_seconds() / 3600
print(f'Completed at {end_time}. Elapsed hours: {elapsed_time}')
if plot or train:
    plt.show()

