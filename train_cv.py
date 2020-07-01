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

from code.preprocessing import *
from code.postprocessing import *
from code.utils import *
from code.models import *
from code.loss import *
from code.metrics import *
from train import *


def get_dataset_cv(config, autotune):
    if config['dataset'] == 'original_all':
        all_data_root = "data/original/all/images/"
    else:
        raise Exception('Unrecognised dataset')

    print("get_dataset_cv():\nconfig dataset: %s\npath: %s\n"%(config['dataset'],all_data_root))
    all_data_glob = np.array(glob.glob(all_data_root + "*.png"))
    all_size = len(all_data_glob)

    k_done = config['tmp']['cv_k_done']
    all_idxs = np.array(config['tmp']['cv_shuffled_idxs'])
    train_idxs = np.append( all_idxs[ :config['cv_k']*k_done ],
                            all_idxs[ config['cv_k']*(k_done+1): ] )
    val_idxs = all_idxs[ config['cv_k']*k_done:config['cv_k']*(k_done+1) ]

    training_data_root = os.path.join(*[config['tmp']['tmp_cv_data_folder'],'training','images'])
    training_gt_root = training_data_root.replace('images', 'groundtruth')
    val_data_root = training_data_root.replace('training', 'validation')
    val_gt_root = val_data_root.replace('images', 'groundtruth')

    if os.path.exists( config['tmp']['tmp_cv_data_folder'] ):
        shutil.rmtree( config['tmp']['tmp_cv_data_folder'] )
    os.makedirs(training_data_root)
    #os.makedirs(os.path.join( os.path.join(*[config['tmp']['tmp_cv_data_folder'],'training','groundtruth']) ))
    os.makedirs(training_gt_root)
    os.makedirs(val_data_root)
    #os.makedirs(os.path.join( os.path.join(*[config['tmp']['tmp_cv_data_folder'],'validation','groundtruth']) ))
    os.makedirs(val_gt_root)


    for source in all_data_glob[train_idxs]:
        name = source.split('/')[-1]
        dest = os.path.join(training_data_root,name)
        shutil.copyfile(source,dest)
        
        source = source.replace('images','groundtruth')
        dest = dest.replace('images','groundtruth')
        shutil.copyfile(source,dest)


    print(training_data_root)
    training_data_glob = glob.glob( os.path.join( training_data_root,"*.png") )
    trainset_size = len(training_data_glob)

    for source in all_data_glob[val_idxs]:
        name =  source.split('/')[-1]
        dest = os.path.join(val_data_root,name)
        shutil.copyfile(source,dest)

        source = source.replace('images','groundtruth')
        dest = dest.replace('images','groundtruth')
        shutil.copyfile(source,dest)

    val_data_glob = glob.glob( os.path.join( val_data_root,"*.png") )
    valset_size = len(val_data_glob)

    print(trainset_size)
    print(valset_size)
    train_dataset, val_dataset = get_dataset_from_path(training_data_glob, val_data_glob, config, autotune)

    return train_dataset, val_dataset, trainset_size, valset_size, training_data_root, val_data_root


def prep_experiment_cv(config,autotune):
    if config['use_cv']:
        return get_dataset_cv(config,autotune)
    else:
        return get_dataset(config,autotune)



def cv_setup(config):
        training_data_root = "data/original/all/images/"
        training_data_glob = glob.glob(training_data_root + "*.png")
        trainset_size = len(training_data_glob)

        train_shuffled_idx = np.arange(trainset_size)
        random.seed(config['seed'])
        random.shuffle(train_shuffled_idx)
        print(config)
        print(type(config))
        config['tmp'] = {}
        config['tmp']['cv_shuffled_idxs'] = train_shuffled_idx
        config['tmp']['cv_k_done'] = 0
        config['tmp']['top_log_folder'] = config['log_folder']
        config['tmp']['tmp_cv_data_folder'] = 'data/tmp_cv'
        print(config['tmp'])

        return(config)


def run_experiment_cv(config,prep_function):
    if config['use_cv']:
        config = cv_setup(config)
        for i in range(config['cv_k_todo']):
            print("train_cv.run_experiment_cv: doing fold no. %d"%i)
            config['tmp']['cv_log_folder'] = os.path.join(config['tmp']['top_log_folder'],'split'+str(i))
            config['log_folder'] = config['tmp']['cv_log_folder']
            os.mkdir(config['tmp']['cv_log_folder'])
            run_experiment(config,prep_function)
            config['tmp']['cv_k_done'] += 1
    else:
        run_experiment(config,prep_function)


if __name__ == '__main__':
    # load each config file and run the experiment
    for config_file in glob.glob('config/' + "*.json"):
        print('main running config file %s'%config_file)
        config = json.loads(open(config_file, 'r').read())
        name = config['name'] + '_' + datetime.datetime.now().strftime("%m%d_%H_%M_%S")
        config['log_folder'] = 'experiments/'+name
        os.makedirs(config['log_folder'])
        run_experiment_cv(config,prep_experiment_cv)
