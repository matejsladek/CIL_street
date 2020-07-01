import glob
import numpy as np
import tensorflow as tf
import datetime, os
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

    print("line67 get_dataset():\nconfig dataset: %s\npath: %s\n"%(config['dataset'],all_data_root))
    all_data_glob = np.array(glob.glob(all_data_root + "*.png"))
    all_size = len(all_data_glob)

    k_done = config['tmp']['cv_k_done']
    all_idxs = np.array(config['tmp']['cv_shuffled_idxs'])
    train_idxs = np.append( all_idxs[ :config['cv_k']*k_done ],
                            all_idxs[ config['cv_k']*(k_done+1): ] )
    val_idxs = all_idxs[ config['cv_k']*k_done:config['cv_k']*(k_done+1) ]

    training_data_root = os.path.join(*[config['tmp']['tmp_cv_data_folder'],'training','images'])
    if os.path.exists(training_data_root):
        shutil.rmtree(training_data_root)
    os.mkdir(training_data_root)
    for source in all_data_glob[train_idxs]:
        name = os,path.split(source)[-1]
        dest = os.path.join(training_data_root,name)
        shutil.copyfile(source,dest)
    training_data_glob = glob.glob(training_data_root + "*.png")
    trainset_size = len(training_data_glob)

    val_data_root = training_data_root.replace('training', 'validation')
    if os.path.exists(val_data_root):
        shutil.rmtree(val_data_root)
    os.mkdir(val_data_root)
    for source in all_data_glob[val_idxs]:
        name = os,path.split(source)[-1]
        dest = os.path.join(val_data_root,name)
        shutil.copyfile(source,dest)
    val_data_glob = glob.glob(val_data_root + "*.png")
    valset_size = len(val_data_glob)

    train_dataset, val_dataset = get_dataset_from_path(training_data_glob, val_data_glob, config, autotune)

    return train_dataset, val_dataset, trainset_size, valset_size, training_data_root, val_data_root


def run_single_split(config,autotune):

    if config['use_cv']:
        train_dataset, val_dataset, trainset_size, valset_size, training_data_root, val_data_root = get_dataset_cv(config,autotune)
    else:
        train_dataset, val_dataset, trainset_size, valset_size, training_data_root, val_data_root = get_dataset(config,autotune)
    
    val_dataset_original = val_dataset
    val_dataset_2 = list(val_dataset)
    val_dataset_numpy_x = np.concatenate([a.numpy()[:, ...] for a,b in val_dataset_2])
    val_dataset_numpy_y = np.concatenate([b.numpy()[:, ...] for a,b in val_dataset_2])
    val_dataset_numpy = (val_dataset_numpy_x, val_dataset_numpy_y)

    print(f"Training dataset contains {trainset_size} images.")
    print(f"Validation dataset contains {valset_size} images.")
    steps_per_epoch = max(trainset_size // config['batch_size'], 1)

    model = create_and_train_model(train_dataset, val_dataset_original, val_dataset_numpy, steps_per_epoch, config)

    postprocess = get_postprocess(config)

    in_val_path = val_data_root
    gt_val_path = val_data_root.replace('images', 'groundtruth')
    pred_val_path = os.path.join(config['log_folder'], "pred_val")
    os.mkdir(pred_val_path)
    postprocess_val_path = os.path.join(config['log_folder'], "postprocess_val")
    os.mkdir(postprocess_val_path)

    in_test_path = 'data/test_images'
    pred_test_path = os.path.join(config['log_folder'], "pred_test")
    os.mkdir(pred_test_path)
    postprocess_test_path = os.path.join(config['log_folder'], "postprocess_test")
    os.mkdir(postprocess_test_path)

    print(np_kaggle_metric(tf.image.resize(model.predict(val_dataset_numpy_x), [400, 400]), tf.image.resize(val_dataset_numpy_y, [400, 400])))
    print(model.evaluate(x=val_dataset_numpy_x, y=val_dataset_numpy_y))

    print('Begin validation for ' + config['name'])
    save_predictions(model=model,
                     model_size=config['img_resize'],
                     output_size=config['img_size'],
                     normalize=config['normalize'],
                     crop=False,
                     input_path=in_val_path,
                     output_path=pred_val_path,
                     config=config)
    out_file = open(config['log_folder'] + "/validation_score.txt", "w")
    out_file.write("Validation through evaluate():\n" + str(model.evaluate(x=val_dataset_numpy_x, y=val_dataset_numpy_y, verbose=0)) + '\n')
    out_file.write("Validation before post processing:\n")
    val_score = compute_metric(np_kaggle_metric, pred_val_path, gt_val_path)
    print(val_score)
    out_file.write(str(val_score) + "\nValidation after post processing:\n")
    postprocess(pred_val_path, postprocess_val_path)
    val_score = compute_metric(np_kaggle_metric, postprocess_val_path, gt_val_path)
    out_file.write(str(val_score) + '\n')
    out_file.close()

    print('Begin test for ' + config['name'])
    save_predictions(model=model,
                     model_size=config['img_resize'],
                     output_size=config['img_size_test'],
                     normalize=config['normalize'],
                     crop=True,
                     input_path=in_test_path,
                     output_path=pred_test_path,
                     config=config)
    to_csv(pred_test_path, os.path.join(config['log_folder'], 'pred_submission.csv'))
    postprocess(pred_test_path, postprocess_test_path)
    to_csv(postprocess_test_path, os.path.join(config['log_folder'],'postprocess_submission.csv'))

    print('Finished ' + config['name'])


def cv_setup(config,autotune):
        training_data_root = "data/original/all/images/"
        training_data_glob = glob.glob(training_data_root + "*.png")
        trainset_size = len(training_data_glob)

        train_shuffled_idx = np.arange(trainset_size)
        random.seed(config['seed'])
        random.shuffle(train_shuffled_idx)
        config['tmp']['cv_shuffled_idx'] = train_shuffled_idx
        config['tmp']['cv_k_done'] = 0
        config['tmp']['top_log_folder'] = config['log_folder']
        config['tmp']['tmp_cv_data_folder'] = 'data/tmp_cv'

        return(config)


def run_experiment_cv(config):
    autotune = tf.data.experimental.AUTOTUNE
    prepare_gpus()

    if config['use_cv']:
        config = cv_setup(config,autotune)
        for i in range(config['cv_k_todo']):
            config['tmp']['cv_log_folder'] = os.path.join(config['tmp']['top_log_folder'],'split'+str(i))
            config['log_folder'] = config['tmp']['cv_log_folder']
            os.mkdir(config['tmp']['cv_log_folder'])

            run_single_split(config,autotune)
            config['tmp']['cv_k_done'] += 1
    else:
        run_single_split(config,autotune)


if __name__ == '__main__':
    # load each config file and run the experiment
    for config_file in glob.glob('config/' + "*.json"):
        config = json.loads(open(config_file, 'r').read())
        name = config['name'] + '_' + datetime.datetime.now().strftime("%m%d_%H_%M_%S")
        config['log_folder'] = 'experiments/'+name
        os.makedirs(config['log_folder'])
        run_experiment_cv(config)
