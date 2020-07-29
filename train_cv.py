import glob
import numpy as np
import tensorflow as tf
import datetime
import os
import os.path
from tensorflow.keras.layers import *
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.optimizers import Adam
import random
import json
import shutil
import argparse
import logging

from code.preprocessing import *
from code.postprocessing import *
from code.utils import *
from code.models import *
from code.loss import *
from code.metrics import *
import train as train

import code.baseline_regression.regression as baseline_regression
import code.baseline_patch_based.patch_based as baseline_patch_based

def get_dataset_cv(config, autotune):
    """
    Loads training and validation globs according to the current fold.

    :param config: config dictionary
    :param autotune: tensorflow autotune
    :return: nothing
    """
    if config['dataset'] == 'original':
        all_data_root = "data/original/all/images/"
    elif config['dataset'] == 'maps1800':
        all_data_root = "data/maps1800/all/images/"
    else:
        raise Exception('Unrecognised dataset')

    all_data_glob = np.array(glob.glob(all_data_root + "*.png"))
    all_data_glob = sorted(all_data_glob)

    k_done = config['tmp']['cv_k_done']
    all_idxs = config['tmp']['cv_shuffled_idxs']
    N_fold = int(len(all_idxs)/config['cv_k'])
    train_idxs = np.concatenate([all_idxs[ :N_fold*k_done ], all_idxs[ N_fold*(k_done+1): ]])
    val_idxs = all_idxs[ N_fold*k_done:N_fold*(k_done+1) ]

    training_data_glob = list(np.array(all_data_glob)[train_idxs])
    val_data_glob = list(np.array(all_data_glob)[val_idxs])

    config['tmp']['val_data_glob'] = val_data_glob
    train_dataset, val_dataset, val_dataset_numpy = train.get_dataset_from_path(training_data_glob, val_data_glob, config, autotune)

    return train_dataset, val_dataset, val_dataset_numpy, training_data_glob, val_data_glob


def cv_setup(config):
    """
    Prepares CV by shuffling indices and creating a temporary entry
    :param config: config dictionary
    :return: nothing
    """

    if config['dataset'] == 'original':
        all_data_root = "data/original/all/images/"
    elif config['dataset'] == 'maps1800':
        all_data_root = "data/maps1800/all/images/"
    else:
        raise Exception('Unrecognised dataset')

    all_data_glob = glob.glob(all_data_root + "*.png")
    allset_size = len(all_data_glob)
    
    all_shuffled_idx = np.arange(allset_size)
    random.seed(config['seed'])

    random.shuffle(all_shuffled_idx)
    config['tmp'] = {}
    config['tmp']['cv_shuffled_idxs'] = all_shuffled_idx
    config['tmp']['cv_k_done'] = 0
    config['tmp']['top_log_folder'] = config['log_folder']

    return config


def run_experiment_cv(config):
    """
    Prepares and runs a cv experiment by calling external methods

    :param config: config dictionary
    :return: nothing
    """

    config = cv_setup(config)
    for i in range(config['cv_k_todo']):
        print("train_cv.run_experiment_cv: doing fold no. %d"%i)
        config['tmp']['cv_log_folder'] = os.path.join(config['tmp']['top_log_folder'],'split'+str(i))
        config['log_folder'] = config['tmp']['cv_log_folder']
        os.mkdir(config['tmp']['cv_log_folder'])
        if config['use_baseline_code1']:
            baseline_regression.run_experiment(config, get_dataset_cv)
        elif config['use_baseline_code2']:
            baseline_patch_based.run_experiment(config, get_dataset_cv)
        elif config['use_ensemble']:
            train.run_experiment_ensemble(config, get_dataset_cv) 
        else:
            train.run_experiment(config, get_dataset_cv)
        config['tmp']['cv_k_done'] += 1


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-c','--config_dir',default='config')
    args = parser.parse_args()
    argsdict = vars(args)

    # load each config file and run the experiment
    for config_file in glob.glob(os.path.join(argsdict['config_dir'], "*.json")):
        config = json.loads(open(config_file, 'r').read())
        name = config['name'] + '_' + datetime.datetime.now().strftime("%m%d_%H_%M_%S")
        config['log_folder'] = 'experiments/'+name
        os.makedirs(config['log_folder'])

        cmd = "cp %s %s" % (config_file, config['log_folder'])
        os.system(cmd)

        logging_path = os.path.join(config['log_folder'], 'train_cv.log')
        logging.basicConfig(filename=logging_path, filemode='w', format='%(asctime)s - %(name)s - %(message)s', level=logging.INFO)
        logging.info("Begin logging for CV")

        run_experiment_cv(config)
