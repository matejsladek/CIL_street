# -----------------------------------------------------------
# Ensemble training script. Requires a config folder to be passed as a parameter. For each .json file in that folder
# an experiment is run.
# CIL 2020 - Team NaN
# -----------------------------------------------------------
from glob import glob
import json
import numpy as np
import tensorflow as tf
import datetime, os
from tensorflow.keras.layers import *
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.optimizers import Adam

from code.preprocessing import *
from code.postprocessing import *
from code.utils import *
from code.models import *
from code.loss import *
from code.metrics import *
from train import prepare_gpus, get_dataset, prep_experiment, create_and_train_model


def bag(in_path, out_path, config):
    """
    Crawls through prediction folders in order to average them.

    :param in_path: common root to prediction folders
    :param out_path: path where to save predictions
    :param config: config dictionary
    :return: nothing
    """
    # TODO: rewrite more nicely
    # dictionary containing images indicized by the last letters in their paths
    images = {}
    # dictionary associating the last letter of paths to image name
    name = {}
    # prepare dictionaries
    for img in glob.glob(in_path + '0/*.png'):
        print(img)
        images[img[-7:]] = []
        name[img[-7:]] = img[len(in_path)+1:]

    # read images
    for i in range(config['n_ensemble']):
        for img in glob.glob(in_path + str(i) + '/*.png'):
            images[img[-7:]].append(cv2.imread(img, 0))

    # average predictions
    for img in glob.glob(in_path + '0/*.png'):
        arr = images[img[-7:]]
        arr = np.array(arr)
        m = np.mean(arr, axis=0)
        # additionally, one can select a threshold for dropping all pixels below a certain intensity
        m[m < config['bagging_threshold']] = 0
        # save averaged predictions
        cv2.imwrite(out_path + '/' + name[img[-7:]], m)


def run_experiment(config,prep_function):

    # train ensemble of models
    for i in range(config['n_ensemble']):
        autotune = tf.data.experimental.AUTOTUNE
        prepare_gpus()
        train_dataset, val_dataset, val_dataset_numpy, \
        trainset_size, valset_size, training_data_root, val_data_root = prep_function(config, autotune)
        val_dataset_numpy_x, val_dataset_numpy_y = val_dataset_numpy
        steps_per_epoch = max(trainset_size // config['batch_size'], 1)
        print('Begin training for ' + config['name'] + '/' + str(i))
        model = create_and_train_model(train_dataset, val_dataset, val_dataset_numpy, steps_per_epoch, config)

        in_test_path = 'data/test_images'
        pred_test_path = os.path.join(config['log_folder'], "pred_test") + str(i)
        os.mkdir(pred_test_path)
        in_val_path = val_data_root
        pred_val_path = os.path.join(config['log_folder'], "pred_val") + str(i)
        os.mkdir(pred_val_path)

        print('Saving predictions')
        save_predictions(model=model,
                         crop=True,
                         input_path=in_test_path,
                         output_path=pred_test_path,
                         postprocessed_output_path=None,
                         config=config,
                         postprocess=None)
        save_predictions(model=model,
                         crop=False,
                         input_path=in_val_path,
                         output_path=pred_val_path,
                         postprocessed_output_path=None,
                         config=config,
                         postprocess=None)

        del model
        print('Finished ' + config['name'] + '/' + str(i))
        config['seed'] += 1

    pred_test_ensemble_path = os.path.join(config['log_folder'], "pred_test_ensemble")
    os.mkdir(pred_test_ensemble_path)
    pred_val_ensemble_path = os.path.join(config['log_folder'], "pred_val_ensemble")
    os.mkdir(pred_val_ensemble_path)

    # aggregate predictions
    bag(os.path.join(config['log_folder'], "pred_val"), pred_val_ensemble_path, config)
    bag(os.path.join(config['log_folder'], "pred_test"), pred_test_ensemble_path, config)

    # apply postprocessing to validation predictions
    postprocess = get_postprocess(config['postprocess'])
    out_file = open(config['log_folder'] + "/validation_score.txt", "w")
    out_file.write("Validation results\n")
    out_file.write("\nKaggle metric on predictions: \n")

    val_predictions = []
    for img in glob.glob(pred_val_ensemble_path + '/*.png'):
        val_predictions.append(cv2.imread(img, 0))
    val_predictions = np.array(val_predictions)
    # TODO: do we really need this?
    if config['normalize']:
        val_predictions = np.where(val_predictions > 128, 1, 0)
    val_predictions = np.expand_dims(val_predictions, axis=-1).astype(np.uint8)

    out_file.write(str(kaggle_metric(val_predictions, val_dataset_numpy_y)))
    out_file.write("\nKaggle metric on predictions after post processing: \n")
    val_postprocessed_predictions = postprocess(val_predictions)
    out_file.write(str(kaggle_metric(val_postprocessed_predictions, val_dataset_numpy_y)))
    out_file.write('\n')
    out_file.close()

    # apply post processing to test predictions
    postprocess_test_ensemble_path = os.path.join(config['log_folder'], "postprocess_test_ensemble")
    os.mkdir(postprocess_test_ensemble_path)

    for img in glob.glob(pred_test_ensemble_path + '/*.png'):
        im = cv2.imread(img, 0)
        # TODO: do we really need this?
        if config['normalize']:
            im = np.where(im > 128, 1, 0)
        im = im.astype(np.uint8)
        im = postprocess(im)
        im = np.squeeze(im, -1)
        out_path = postprocess_test_ensemble_path + img[len(pred_test_ensemble_path):]
        cv2.imwrite(out_path, im)

    to_csv(pred_test_ensemble_path, os.path.join(config['log_folder'], 'pred_submission.csv'))
    to_csv(postprocess_test_ensemble_path, os.path.join(config['log_folder'], 'postprocess_submission.csv'))


if __name__ == '__main__':
    # load each config file and run the experiment
    for config_file in glob.glob('config/' + "*.json"):
        config = json.loads(open(config_file, 'r').read())
        name = config['name'] + '_' + datetime.datetime.now().strftime("%m%d_%H_%M_%S")
        config['log_folder'] = 'experiments/'+name
        os.makedirs(config['log_folder'])
        run_experiment(config, prep_experiment)
