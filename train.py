# -----------------------------------------------------------
# Main training script. Requires a config folder to be passed as a parameter. For each .json file in that folder
# an experiment is run.
# CIL 2020 - Team NaN
# -----------------------------------------------------------
import json
import argparse
from glob import glob
import numpy as np
import tensorflow as tf
import datetime, os
import logging
from tensorflow.keras.layers import *
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, CSVLogger
from tensorflow.keras.optimizers import Adam
import shutil

from code.preprocessing import *
from code.postprocessing import *
from code.utils import *
from code.models import *
from code.loss import *
from code.metrics import *


def prepare_gpus():
    """
    Enable memory growth and detect GPU
    :return: nothing
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
    Retrieve tf.Datasets from globs of images paths, applies preprocessing.
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
                                              brightness=config['brightness'],
                                              predict_contour=config['predict_contour'],
                                              predict_distance=config['predict_distance'])
    train_dataset = train_dataset.map(train_image_loader, num_parallel_calls=autotune)
    train_dataset = train_dataset.repeat().shuffle(buffer_size=config['buffer_size'], seed=config['seed'])\
        .batch(config['batch_size']).prefetch(buffer_size=autotune)

    val_dataset = tf.data.Dataset.list_files(val_data_glob, shuffle=False, seed=config['seed'])
    val_dataset = val_dataset.map(get_parse_image(hard=config['hard_mask']))
    val_image_loader = get_load_image_val(size=config['img_resize'], normalize=config['normalize'],
                                          predict_contour=config['predict_contour'],
                                          predict_distance=config['predict_distance'])
    val_dataset = val_dataset.map(val_image_loader)
    val_dataset = val_dataset.batch(config['batch_size']).prefetch(buffer_size=autotune)

    # prepare numpy copy of validation set
    val_dataset_copy = val_dataset
    val_dataset_copy = list(val_dataset_copy)
    if config['predict_contour'] or config['predict_distance']:
        val_dataset_numpy_x = np.concatenate([a.numpy()[:, ...] for a,b in val_dataset_copy])
        val_dataset_numpy_y_mask = np.concatenate([b[0].numpy()[:, ...] for a,b in val_dataset_copy])
        val_dataset_numpy_y_new_task = np.concatenate([b[1].numpy()[:, ...] for a,b in val_dataset_copy])
        val_dataset_numpy_y = (val_dataset_numpy_y_mask, val_dataset_numpy_y_new_task)
    else:
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
        training_data_root = "data/original/training/images/"
    elif config['dataset'] == 'maps1800_all':
        all_data_root = "data/maps1800/all/images/"
    else:
        raise Exception('Unrecognised dataset')
    val_data_root = training_data_root.replace('training', 'validation')

    training_data_glob = glob.glob(training_data_root + "*.png")
    val_data_glob = glob.glob(val_data_root + "*.png")
    trainset_size = len(training_data_glob)
    valset_size = len(val_data_glob)

    train_dataset, val_dataset, val_dataset_numpy = get_dataset_from_path(training_data_glob, val_data_glob, config, autotune)

    return train_dataset, val_dataset, val_dataset_numpy, trainset_size, valset_size, training_data_root, val_data_root


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
                    predict_distance=config['predict_distance'], predict_contour=config['predict_contour'],
                    aspp=config['aspp'], se=config['se'], residual=config['residual'], art=config['art'],
                    experimental_decoder=config['experimental_decoder'],
                    decoder_exp_setting=config['decoder_exp_setting'])

    if config['augment_loss']:
        config['loss'][0] = custom_loss

    if config['predict_distance'] and config['predict_contour']:
        model.compile(optimizer=Adam(learning_rate=learning_rate), loss=config['loss'],
                      loss_weights=config['loss_weights'], metrics=['accuracy', kaggle_metric, f1_m, iou])
    elif config['predict_distance']:
        model.compile(optimizer=Adam(learning_rate=learning_rate), loss=[config['loss'][0],config['loss'][2]],
                      loss_weights=[config['loss_weights'][0],config['loss_weights'][2]], metrics=['accuracy', kaggle_metric, f1_m, iou])
    elif config['predict_contour']:
        model.compile(optimizer=Adam(learning_rate=learning_rate), loss=config['loss'][0:2],
                      loss_weights=config['loss_weights'][0:2], metrics=['accuracy', kaggle_metric, f1_m, iou])
    else:
        model.compile(optimizer=Adam(learning_rate=learning_rate), loss=config['loss'][0],
                      metrics=['accuracy', kaggle_metric, f1_m, iou])

    return model


def create_and_train_model(train_dataset, val_dataset, val_dataset_numpy, steps_per_epoch, config):
    """
    Creates the model and trains it. Weights are restored from best epoch according to validation kaggle_metric.
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

            # select correct index of model.evaluate() result according to config
            self.metric_index = 2
            self.loss_index = 0
            if config['predict_contour'] or config['predict_distance']:
                self.metric_index = -3
                self.loss_index = 1

        def on_epoch_end(self, epoch, logs=None):
            ev = model.evaluate(x=val_dataset_numpy_x, y=val_dataset_numpy_y, verbose=0)
            print('\nValidation metrics: ' + str(ev) + '\n')
            if ev[self.loss_index] < self.lowest_loss and not config['stop_on_metric']:
                self.lowest_loss = ev[self.loss_index]
                print('\nNew lowest loss. Saving weights.\n')
                model.save_weights(config['log_folder'] + '/best_model.h5')

            if ev[self.metric_index] > self.highest_metric and config['stop_on_metric']:
                self.highest_metric = ev[self.metric_index]
                print('New best metric. Saving weights.')
                model.save_weights(config['log_folder'] + '/best_model.h5')

    callbacks = [tf.keras.callbacks.TensorBoard(config['log_folder'] + '/log')]

    if config['custom_callback']:
        callbacks.append(CustomCallback())
    else:
        # only supports single task learning
        callbacks.append(tf.keras.callbacks.ModelCheckpoint(config['log_folder'] + '/best_model.h5',
                                                            monitor='val_kaggle_metric' and config['stop_on_metric'] or 'val_loss',
                                                            verbose=1,
                                                            save_best_only=True,
                                                            save_weights_only=True))

    csv_logger = CSVLogger(os.path.join(config['log_folder'],'train_log.csv'), append=True, separator=';')
    callbacks.append(csv_logger)
    logging.info("no. callbacks: %d"%(len(callbacks)))
    logging.info(str(callbacks))
    # train model
    model_history = model.fit(train_dataset, epochs=config['epochs'],
                              steps_per_epoch=steps_per_epoch,
                              validation_data=val_dataset,
                              callbacks=callbacks)

    if config['epochs'] > 0:
        model.load_weights(config['log_folder'] + '/best_model.h5')

    return model


def prep_experiment(config,autotune):
    print('Load dataset for ' + config['name'])
    return get_dataset(config, autotune)


def run_experiment(config,prep_function):
    """
    Trains and evaluates a model before computing and saving test predictions, all according to the config file.
    :param config: config dictionary
    :param prep_function: data loader
    :return: nothing
    """

    logging.info("Begin train.run_experiment")

    # tensorflow setup
    autotune = tf.data.experimental.AUTOTUNE
    prepare_gpus()

    # retrieve datasets
    train_dataset, val_dataset, val_dataset_numpy,\
    trainset_size, valset_size, training_data_root, val_data_root = prep_function(config,autotune)
    val_dataset_numpy_x, val_dataset_numpy_y = val_dataset_numpy
    print(f"Training dataset contains {trainset_size} images.")
    print(f"Validation dataset contains {valset_size} images.")
    steps_per_epoch = max(trainset_size // config['batch_size'], 1)

    # train
    print('Begin training')
    logging.info('Begin training')
    model = create_and_train_model(train_dataset, val_dataset, val_dataset_numpy, steps_per_epoch, config)
    logging.info('Finished training')
    summary = model.summary()
    print(type(summary))
    print(summary)
    summary_file = open(config['log_folder'] + "/model_summary.txt", "w")
    summary_file.write(str(summary))
    summary_file.write("\n")
    summary_file.close()
    print("Summarizing model is successful")


    # compute and save validation scores
    postprocess = get_postprocess(config['postprocess'])
    print('Saving validation scores')
    out_file = open(config['log_folder'] + "/validation_score.txt", "w")
    out_file.write("Validation results\n")
    out_file.write("Results of model.evaluate: \n")
    out_file.write(str(model.evaluate(x=val_dataset_numpy_x, y=val_dataset_numpy_y, verbose=0)))
    out_file.write("\nKaggle metric on predictions: \n")
    if config['predict_contour'] or config['predict_distance']:
        predictions = tf.image.resize(model.predict(val_dataset_numpy_x)[0], [config['img_size'], config['img_size']])
        out_file.write(str(kaggle_metric(predictions, tf.image.resize(val_dataset_numpy_y[0], [config['img_size'], config['img_size']])).numpy()))
    else:
        predictions = tf.image.resize(model.predict(val_dataset_numpy_x), [config['img_size'], config['img_size']])
        out_file.write(str(kaggle_metric(predictions, tf.image.resize(val_dataset_numpy_y, [config['img_size'], config['img_size']])).numpy()))
    out_file.write("\nKaggle metric on predictions after post processing: \n")
    postprocessed_predictions = postprocess(predictions.numpy())
    if config['predict_contour'] or config['predict_distance']:
        out_file.write(str(kaggle_metric(postprocessed_predictions, tf.image.resize(val_dataset_numpy_y[0], [config['img_size'], config['img_size']])).numpy()))
    else:
        out_file.write(str(kaggle_metric(postprocessed_predictions, tf.image.resize(val_dataset_numpy_y, [config['img_size'], config['img_size']])).numpy()))
        out_file.write("\nAccuracy, F1 and IoU after post processing: \n")
        out_file.write(str(tf.keras.backend.mean(postprocessed_predictions == tf.image.resize(val_dataset_numpy_y, [config['img_size'], config['img_size']])).numpy()))
        out_file.write('\n')
        out_file.write(str(f1_m(postprocessed_predictions, tf.image.resize(val_dataset_numpy_y, [config['img_size'], config['img_size']])).numpy()))
        out_file.write('\n')
        out_file.write(str(iou(postprocessed_predictions, tf.image.resize(val_dataset_numpy_y, [config['img_size'], config['img_size']])).numpy()))
    out_file.write('\n')
    out_file.close()
    print('Validation is successful')

    # save predictions on test images
    in_test_path = 'data/test_images'
    # folder for predictions
    pred_test_path = os.path.join(config['log_folder'], "pred_test")
    os.mkdir(pred_test_path)
    # folder for postprocessed predictions
    postprocess_test_path = os.path.join(config['log_folder'], "postprocess_test")
    os.mkdir(postprocess_test_path)

    print('Saving predictions')
    save_predictions(model=model,
                     crop=True,
                     input_path=in_test_path,
                     output_path=pred_test_path,
                     postprocessed_output_path=postprocess_test_path,
                     config=config,
                     postprocess=postprocess)

    # save predictions as csv files for simple submission
    to_csv(pred_test_path, os.path.join(config['log_folder'], 'pred_submission.csv'))
    to_csv(postprocess_test_path, os.path.join(config['log_folder'],'postprocess_submission.csv'))

    if not(config['save_model']):
        if os.path.exists( os.path.join(config['log_folder'],'best_model.h5') ):
            os.remove( os.path.join(config['log_folder'],'best_model.h5') )

    if config['use_cv']:
        if os.path.exists( config['tmp']['tmp_cv_data_folder'] ):
            shutil.rmtree( config['tmp']['tmp_cv_data_folder'] )

    print('Finished ' + config['name'])


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-c','--config_dir',default='config')
    args = parser.parse_args()
    argsdict = vars(args)

    # load each config file and run the experiment
    for config_file in glob.glob( os.path.join(argsdict['config_dir'],"*.json") ):
        config = json.loads(open(config_file, 'r').read())
        name = config['name'] + '_' + datetime.datetime.now().strftime("%m%d_%H_%M_%S")
        config['log_folder'] = 'experiments/'+name
        os.makedirs(config['log_folder'])
        run_experiment(config,prep_experiment)
