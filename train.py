# -----------------------------------------------------------
# Main training script. Requires a config folder to be passed
# as a parameter. For each .json file in that folder an
# experiment is run.
# CIL 2020 - Team NaN
# -----------------------------------------------------------

import json
import argparse
import sys
from glob import glob
import datetime, os
import logging
import shutil
import numpy as np
import random
import tensorflow as tf
from tensorflow.keras.callbacks import ModelCheckpoint, CSVLogger
from tensorflow.keras.optimizers import Adam

from code.preprocessing import *
from code.postprocessing import *
from code.utils import *
from code.models import *
from code.loss import *
from code.metrics import *

import code.baseline_regression.regression as baseline_regression
import code.baseline_patch_based.patch_based as baseline_patch_based


def prepare_gpus():
    """
    Enable memory growth and detect GPU
    :return: None
    """
    print(f"Tensorflow ver. {tf.__version__}")
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            logical_gpus = tf.config.experimental.list_logical_devices('GPU')
            print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
        except RuntimeError as e:
            print(e)


def get_dataset_from_path(training_data_glob, val_data_glob, config, autotune):
    """
    Retrieves tf.Datasets from globs of images paths, applies preprocessing.
    :param training_data_glob: glob of training images
    :param val_data_glob: glob of validation images
    :param config: configuration dictionary
    :param autotune: tensorflow autotune
    :return: training and validation tf.Datasets, as well as a numpy validation copy to be used for consistent evaluation
    """
    train_dataset = tf.data.Dataset.list_files(training_data_glob, seed=config['seed'])
    train_dataset = train_dataset.map(get_parse_image(hard=config['hard_mask']))
    train_image_loader = get_load_image_train(size=config['img_resize'],
                                              normalize=config['normalize'],
                                              h_flip=config['h_flip'],
                                              v_flip=config['v_flip'],
                                              rot=config['rot'],
                                              contrast=config['contrast'],
                                              brightness=config['brightness'])
    train_dataset = train_dataset.map(train_image_loader, num_parallel_calls=autotune)
    train_dataset = train_dataset.repeat().shuffle(buffer_size=config['buffer_size'], seed=config['seed'])\
        .batch(config['batch_size']).prefetch(buffer_size=autotune)

    val_dataset = tf.data.Dataset.list_files(val_data_glob, shuffle=False, seed=config['seed'])
    val_dataset = val_dataset.map(get_parse_image(hard=config['hard_mask']))
    val_image_loader = get_load_image_val(size=config['img_resize'], normalize=config['normalize'])
    val_dataset = val_dataset.map(val_image_loader)
    val_dataset = val_dataset.batch(config['batch_size']).prefetch(buffer_size=autotune)

    # prepare numpy copy of validation set
    val_dataset_copy = val_dataset
    val_dataset_copy = list(val_dataset_copy)
    val_dataset_numpy_x = np.concatenate([a.numpy()[:, ...] for a,b in val_dataset_copy])
    val_dataset_numpy_y = np.concatenate([b.numpy()[:, ...] for a,b in val_dataset_copy])
    val_dataset_numpy = (val_dataset_numpy_x, val_dataset_numpy_y)

    return train_dataset, val_dataset, val_dataset_numpy


def get_dataset(config, autotune):
    """
    Produces datasets according to config.
    :param config: configuration dictionary
    :param autotune: tensorflow autotune
    :return: datasets, their size and the path from which they were loaded
    """
    # select dataset root according to its name
    if config['dataset'] == 'original':
        all_data_root = "data/original/all/images/"
    elif config['dataset'] == 'maps1800':
        all_data_root = "data/maps1800/all/images/"
    else:
        raise Exception('Unrecognised dataset')

    all_data_glob = glob.glob(all_data_root + "*.png")

    random.seed(config['seed'])

    if config["val_split"] == 0:
        val_data_glob = random.sample(all_data_glob, int(len(all_data_glob)*0.1))
        training_data_glob = all_data_glob
    else:
        val_data_glob = random.sample(all_data_glob, int(len(all_data_glob)*config['val_split']))
        training_data_glob = [sample for sample in all_data_glob if sample not in val_data_glob]

    train_dataset, val_dataset, val_dataset_numpy = get_dataset_from_path(training_data_glob, val_data_glob, config, autotune)

    return train_dataset, val_dataset, val_dataset_numpy, training_data_glob, val_data_glob


def get_model(config):
    """
    Builds and compiles the model according to config

    :param config: config dictionary
    :return: model ready for training
    """
    learning_rate = config['learning_rate']

    encoder_weights = None
    if config['pretrained']:
        encoder_weights = 'imagenet'

    model = RoadNet(backbone_name=config['backbone'],
                    input_shape=(config['img_resize'], config['img_resize'], config['n_channels']),
                    encoder_weights=encoder_weights, encoder_freeze=False,
                    aspp=config['aspp'], se=config['se'], residual=config['residual'], art=config['art'])

    if config['augment_loss']:
        config['loss'] = custom_loss

    model.compile(optimizer=Adam(learning_rate=learning_rate), loss=config['loss'],
                  metrics=['accuracy', kaggle_metric, f1_m, iou])

    return model


def create_and_train_model(train_dataset, val_dataset, val_dataset_numpy, steps_per_epoch, config):
    """
    Creates the model and trains it. Weights are restored from best epoch.
    :param train_dataset: training dataset
    :param val_dataset: validation dataset
    :param val_dataset_numpy: numpy version of validation dataset for simple and consistent evaluation
    :param steps_per_epoch: steps per epoch during training
    :param config: config dictionary
    :return: trained model
    """

    model = get_model(config)
    val_dataset_numpy_x, val_dataset_numpy_y = val_dataset_numpy

    # custom model saving callback to solve inconsistencies with model.evaluate() and tf.Datasets
    class CustomCallback(tf.keras.callbacks.Callback):
        def __init__(self):
            super(CustomCallback, self).__init__()
            self.lowest_loss = 100
            self.highest_metric = 0

        def on_epoch_end(self, epoch, logs=None):
            ev = model.evaluate(x=val_dataset_numpy_x, y=val_dataset_numpy_y, verbose=0)
            print('\nValidation metrics: ' + str(ev) + '\n')
            if ev[0] < self.lowest_loss and not config['stop_on_metric']:
                self.lowest_loss = ev[0]
                print('\nNew lowest loss. Saving weights.\n')
                model.save_weights(config['log_folder'] + '/best_model.h5')

            if ev[2] > self.highest_metric and config['stop_on_metric']:
                self.highest_metric = ev[2]
                print('New best metric. Saving weights.')
                model.save_weights(config['log_folder'] + '/best_model.h5')

    callbacks = [tf.keras.callbacks.TensorBoard(config['log_folder'] + '/log')]

    if config['custom_callback']:
        callbacks.append(CustomCallback())
    else:
        callbacks.append(tf.keras.callbacks.ModelCheckpoint(config['log_folder'] + '/best_model.h5',
                                                            monitor='val_kaggle_metric' and config['stop_on_metric'] or 'val_loss',
                                                            verbose=1,
                                                            save_best_only=True,
                                                            save_weights_only=True))

    csv_logger = CSVLogger(os.path.join(config['log_folder'], 'train_log.csv'), append=True, separator=';')
    callbacks.append(csv_logger)

    # train model
    model_history = model.fit(train_dataset, epochs=config['epochs'],
                              steps_per_epoch=steps_per_epoch,
                              validation_data=val_dataset,
                              callbacks=callbacks)

    if config['epochs'] > 0 and config["restore_best_model"]:
        model.load_weights(config['log_folder'] + '/best_model.h5')

    return model


def validate(model, val_dataset_numpy, config):
    """
    Computes and saves validation scores for the model

    :param model: model after training
    :param val_dataset_numpy: 3D numpy array containing validation data
    :return: nothing
    """
    val_dataset_numpy_x, val_dataset_numpy_y = val_dataset_numpy
    val_dataset_numpy_y_resized = tf.image.resize(val_dataset_numpy_y, [config['img_size'], config['img_size']])
    postprocess = get_postprocess(config['postprocess'])
    predictions = tf.image.resize(model.predict(val_dataset_numpy_x), [config['img_size'], config['img_size']])
    postprocessed_predictions = np.where(postprocess(predictions.numpy()*255) > 127, 1, 0).astype(np.float32)

    out_file = open(config['log_folder'] + "/validation_score.txt", "w")
    out_file.write("Results of model.evaluate: \n")
    out_file.write(str(model.evaluate(x=val_dataset_numpy_x, y=val_dataset_numpy_y, verbose=0)))
    out_file.write("\nKaggle metric on predictions: \n")
    out_file.write(str(kaggle_metric(predictions, val_dataset_numpy_y_resized).numpy()))
    out_file.write("\nKaggle metric on predictions after post processing: \n")
    out_file.write(str(kaggle_metric(postprocessed_predictions, val_dataset_numpy_y_resized).numpy()))
    out_file.write("\nAccuracy, F1 and IoU after post processing: \n")
    out_file.write(str(tf.keras.backend.mean(postprocessed_predictions == val_dataset_numpy_y_resized).numpy()) + ' ')
    out_file.write(str(f1_m(postprocessed_predictions, val_dataset_numpy_y_resized).numpy()) + ' ')
    out_file.write(str(iou(postprocessed_predictions, val_dataset_numpy_y_resized).numpy()) + ' ')
    out_file.write('\n')
    out_file.close()


def test(model, config):
    """
    Produces and saves predictions on the test set, both as .png images and .csv files

    :param model: fully trained model
    :return: nothing
    """

    # save predictions on test images
    in_test_path = 'data/test_images'
    # folder for predictions
    pred_test_path = os.path.join(config['log_folder'], "pred_test")
    os.mkdir(pred_test_path)
    # folder for postprocessed predictions
    postprocess_test_path = os.path.join(config['log_folder'], "postprocess_test")
    os.mkdir(postprocess_test_path)

    postprocess = get_postprocess(config['postprocess'])
    save_predictions(model=model,
                     crop=True,
                     input_path=in_test_path,
                     output_path=pred_test_path,
                     postprocessed_output_path=postprocess_test_path,
                     config=config,
                     postprocess=postprocess)

    # save predictions as csv files for simple submission
    to_csv(pred_test_path, os.path.join(config['log_folder'], 'pred_submission.csv'))
    to_csv(postprocess_test_path, os.path.join(config['log_folder'], 'postprocess_submission.csv'))


def run_experiment(config,prep_function):
    """
    Trains and evaluates a model before computing and saving test predictions, all according to the config file.
    :param config: config dictionary
    :param prep_function: data loader
    :return: nothing
    """

    logging.info("Begin single training run.")

    # tensorflow setup
    autotune = tf.data.experimental.AUTOTUNE
    prepare_gpus()

    # retrieve datasets
    train_dataset, val_dataset, val_dataset_numpy, training_data_glob, val_data_glob = prep_function(config, autotune)
    logging.info(f"Training dataset contains {len(training_data_glob)} images")
    logging.info(f"Validation dataset contains {len(val_data_glob)} images")
    steps_per_epoch = max(len(training_data_glob) // config['batch_size'], 1)

    # train
    logging.info('Begin training')
    model = create_and_train_model(train_dataset, val_dataset, val_dataset_numpy, steps_per_epoch, config)
    logging.info('Finished training')

    logging.info('Validating')
    validate(model, val_dataset_numpy, config)

    logging.info('Testing')
    test(model, config)

    if not(config['save_model']):
        if os.path.exists(os.path.join(config['log_folder'], 'best_model.h5')):
            os.remove(os.path.join(config['log_folder'], 'best_model.h5'))

    logging.info('Finished single training run')


def run_experiment_ensemble(config,prep_function):
    """
    Trains and evaluates an ensemble of model before computing and saving test predictions, all according to the config file.
    :param config: config dictionary
    :param prep_function: data loader
    :return: nothing
    """

    # train ensemble of models
    val_preds = []
    for i in range(config['n_ensemble']):

        logging.info("Begin single training run.")

        # tensorflow setup
        autotune = tf.data.experimental.AUTOTUNE
        prepare_gpus()

        # retrieve datasets
        train_dataset, val_dataset, val_dataset_numpy, training_data_glob, val_data_glob = prep_function(config, autotune)
        logging.info(f"Training dataset contains {len(training_data_glob)} images")
        logging.info(f"Validation dataset contains {len(val_data_glob)} images")
        steps_per_epoch = max(len(training_data_glob) // config['batch_size'], 1)

        # train
        logging.info('Begin training')
        model = create_and_train_model(train_dataset, val_dataset, val_dataset_numpy, steps_per_epoch, config)
        logging.info('Finished training')

        in_test_path = 'data/test_images'
        pred_test_path = os.path.join(config['log_folder'], "pred_test") + str(i)
        os.mkdir(pred_test_path)

        logging.info('Saving predictions')
        save_predictions(model=model,
                         crop=True,
                         input_path=in_test_path,
                         output_path=pred_test_path,
                         postprocessed_output_path=None,
                         config=config,
                         postprocess=None)

        val_dataset_numpy_x, val_dataset_numpy_y = val_dataset_numpy
        val_dataset_numpy_y_resized = tf.image.resize(val_dataset_numpy_y, [config['img_size'], config['img_size']])
        predictions = tf.image.resize(model.predict(val_dataset_numpy_x), [config['img_size'], config['img_size']])

        val_preds.append(predictions)

        if not (config['save_model']):
            if os.path.exists(os.path.join(config['log_folder'], 'best_model.h5')):
                os.remove(os.path.join(config['log_folder'], 'best_model.h5'))

        logging.info('Finished single ensemble training run')

        del model
        logging.info('Finished ' + config['name'] + '/' + str(i))
        config['seed'] += 1

    val_preds = tf.stack(val_preds)
    predictions = tf.reduce_mean(val_preds, axis=0)
    rounded_pred = tf.math.round(predictions)
    logging.info("Validating")
    postprocess = get_postprocess(config['postprocess'])
    postprocessed_predictions = np.where(postprocess(predictions.numpy()*255)>127, 1, 0).astype(np.float32)

    out_file = open(config['log_folder'] + "/validation_score.txt", "w")
    out_file.write("\nKaggle metric on predictions: \n")
    out_file.write(str(kaggle_metric(rounded_pred, val_dataset_numpy_y_resized).numpy()))
    out_file.write("\nAccuracy, F1 and IoU before post processing: \n")
    out_file.write(str(tf.keras.backend.mean(rounded_pred == val_dataset_numpy_y_resized).numpy()) + '\n')
    out_file.write(str(f1_m(rounded_pred, val_dataset_numpy_y_resized).numpy())+'\n')
    out_file.write(str(iou(rounded_pred, val_dataset_numpy_y_resized).numpy())+'\n')
    out_file.write("\nKaggle metric on predictions after post processing: \n")
    out_file.write(str(kaggle_metric(postprocessed_predictions, val_dataset_numpy_y_resized).numpy()))
    out_file.write("\nAccuracy, F1 and IoU after post processing: \n")
    out_file.write(str(tf.keras.backend.mean(postprocessed_predictions == val_dataset_numpy_y_resized).numpy()) + '\n')
    out_file.write(str(f1_m(postprocessed_predictions, val_dataset_numpy_y_resized).numpy())+'\n')
    out_file.write(str(iou(postprocessed_predictions, val_dataset_numpy_y_resized).numpy())+'\n')
    out_file.write('\n')
    out_file.close()

    logging.info("Testing")
    pred_test_ensemble_path = os.path.join(config['log_folder'], "pred_test_ensemble")
    os.mkdir(pred_test_ensemble_path)
    postprocess_test_ensemble_path = os.path.join(config['log_folder'], "postprocess_test_ensemble")
    os.mkdir(postprocess_test_ensemble_path)

    bag(os.path.join(config['log_folder'], "pred_test"), pred_test_ensemble_path, config)

    # apply postprocessing to validation predictions
    postprocess = get_postprocess(config['postprocess'])

    for img in glob.glob(pred_test_ensemble_path + '/*.png'):
        im = cv2.imread(img, 0)
        im = im.astype(np.uint8)
        im = postprocess(np.expand_dims(im, 0))
        im = np.squeeze(im)
        out_path = postprocess_test_ensemble_path + img[len(pred_test_ensemble_path):]
        cv2.imwrite(out_path, im)

    to_csv(pred_test_ensemble_path, os.path.join(config['log_folder'], 'pred_submission.csv'))
    to_csv(postprocess_test_ensemble_path, os.path.join(config['log_folder'], 'postprocess_submission.csv'))


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

        logging_path = os.path.join(config['log_folder'], 'train.log')
        logging.basicConfig(filename=logging_path, filemode='w', format='%(asctime)s - %(name)s - %(message)s', level=logging.INFO)
        logging.info("Begin logging for single training run")
        root = logging.getLogger()
        handler = logging.StreamHandler(sys.stdout)
        handler.setLevel(logging.INFO)
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        root.addHandler(handler)
        if config['use_baseline_code1'] and config['use_baseline_code2']:
            raise Exception('Ambiguous config file.')
        elif config['use_baseline_code1']:
            baseline_regression.run_experiment(config, get_dataset)
        elif config['use_baseline_code2']:
            baseline_patch_based.run_experiment(config, get_dataset)
        elif config['use_ensemble']:
            run_experiment_ensemble(config, get_dataset)
        else:
            run_experiment(config, get_dataset)
