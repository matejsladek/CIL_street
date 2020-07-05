from glob import glob
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
import json


# enable memory growth and detects gpus
def prepare_gpus():
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


# retrieve tf.Datasets from globs of images paths, applies preprocessing
def get_dataset_from_path(training_data_glob, val_data_glob, config, autotune):
    # can use from_tensor_slices to speed up
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


# produce datasets according to config
def get_dataset(config, autotune):
    if config['dataset'] == 'original':
        training_data_root = "data/original/training/images/"
    else:
        raise Exception('Unrecognised dataset')
    val_data_root = training_data_root.replace('training', 'validation')

    training_data_glob = glob.glob(training_data_root + "*.png")
    val_data_glob = glob.glob(val_data_root + "*.png")
    trainset_size = len(training_data_glob)
    valset_size = len(val_data_glob)

    train_dataset, val_dataset, val_dataset_numpy = get_dataset_from_path(training_data_glob, val_data_glob, config, autotune)

    return train_dataset, val_dataset, val_dataset_numpy, trainset_size, valset_size, training_data_root, val_data_root


# build and compile the model
def get_model(config):
    learning_rate = config['learning_rate']

    encoder_weights = None
    if config['pretrained']:
        encoder_weights = 'imagenet'

    model = PretrainedUnet(backbone_name=config['backbone'],
                           input_shape=(config['img_resize'], config['img_resize'], config['n_channels']),
                           encoder_weights=encoder_weights, encoder_freeze=False,
                           predict_distance=config['predict_distance'], predict_contour=config['predict_contour'],
                           aspp=config['aspp'])

    def custom_loss(y_pred, y_true):
        return tf.keras.losses.binary_crossentropy(y_true, y_pred) + y_pred*(1-y_pred)

    if config['augment_loss']:
        config['loss'][0] = custom_loss

    if config['predict_distance'] and config['predict_contour']:
        model.compile(optimizer=Adam(learning_rate=learning_rate), loss=config['loss'],
                      loss_weights=config['loss_weights'], metrics=['accuracy', kaggle_metric])
    elif config['predict_distance']:
        model.compile(optimizer=Adam(learning_rate=learning_rate), loss=config['loss'][0:2],
                      loss_weights=config['loss_weights'][0:2], metrics=['accuracy', kaggle_metric])
    elif config['predict_contour']:
        model.compile(optimizer=Adam(learning_rate=learning_rate), loss=config['loss'][0:2],
                      loss_weights=config['loss_weights'][0:2], metrics=['accuracy', kaggle_metric])
    else:
        model.compile(optimizer=Adam(learning_rate=learning_rate), loss=config['loss'][0],
                      metrics=['accuracy', kaggle_metric])

    return model


# create the model and train it, load weights from epoch with best val loss
def create_and_train_model(train_dataset, val_dataset_original, val_dataset_numpy, steps_per_epoch, config):
    model = get_model(config)
    val_dataset_numpy_x, val_dataset_numpy_y = val_dataset_numpy

    class CustomCallback(tf.keras.callbacks.Callback):
        def __init__(self):
            super(CustomCallback, self).__init__()
            self.lowest_loss = 100
            self.highest_metric = 0
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
        # only works with single task learning
        callbacks.append(tf.keras.callbacks.ModelCheckpoint(config['log_folder'] + '/best_model.h5',
                                                            monitor='val_kaggle_metric' and config['stop_on_metric'] or 'val_loss',
                                                            verbose=1,
                                                            save_best_only=True,
                                                            save_weights_only=True))

    model_history = model.fit(train_dataset, epochs=config['epochs'],
                              steps_per_epoch=steps_per_epoch,
                              validation_data=val_dataset_original,
                              callbacks=callbacks)

    if config['epochs'] > 0:
        model.load_weights(config['log_folder'] + '/best_model.h5')
    return model


def prep_experiment(config,autotune):
    print('Load dataset')
    return get_dataset(config, autotune)


def bag(in_path, out_path, config):
    images = {}
    name = {}
    print(in_path)
    print(out_path)
    print(glob.glob(in_path + '0/*.png'))
    for img in glob.glob(in_path + '0/*.png'):
        print(img)
        images[img[-7:]] = []
        name[img[-7:]] = img[len(in_path)+1:]

    for i in range(config['n_ensemble']):
        print(i)
        for img in glob.glob(in_path + str(i) + '/*.png'):
            print(img)
            images[img[-7:]].append(cv2.imread(img, 0))

    for img in glob.glob(in_path + '0/*.png'):
        arr = images[img[-7:]]
        arr = np.array(arr)
        m = np.mean(arr, axis=0)
        m[m < config['bagging_threshold']] = 0
        print(name[img[-7:]])
        cv2.imwrite(out_path + '/' + name[img[-7:]], m)


def run_experiment(config,prep_function):

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

    bag(os.path.join(config['log_folder'], "pred_val"), pred_val_ensemble_path, config)
    bag(os.path.join(config['log_folder'], "pred_test"), pred_test_ensemble_path, config)

    postprocess = get_postprocess(config['postprocess'])
    out_file = open(config['log_folder'] + "/validation_score.txt", "w")
    out_file.write("Validation results\n")
    out_file.write("\nKaggle metric on predictions: \n")

    val_predictions = []
    for img in glob.glob(pred_val_ensemble_path + '/*.png'):
        val_predictions.append(cv2.imread(img, 0))
    val_predictions = np.array(val_predictions)
    if config['normalize']:
        val_predictions = np.where(val_predictions > 128, 1, 0)
    val_predictions = np.expand_dims(val_predictions, axis=-1).astype(np.uint8)

    out_file.write(str(kaggle_metric(val_predictions, val_dataset_numpy_y)))
    out_file.write("\nKaggle metric on predictions after post processing: \n")
    val_postprocessed_predictions = postprocess(val_predictions)
    out_file.write(str(kaggle_metric(val_postprocessed_predictions, val_dataset_numpy_y)))
    out_file.write('\n')
    out_file.close()

    postprocess_test_ensemble_path = os.path.join(config['log_folder'], "postprocess_test_ensemble")
    os.mkdir(postprocess_test_ensemble_path)

    for img in glob.glob(pred_test_ensemble_path + '/*.png'):
        im = cv2.imread(img, 0)
        if config['normalize']:
            im = np.where(im > 128, 1, 0)
        im = np.expand_dims(im, axis=-1).astype(np.uint8)
        im = postprocess(im)
        cv2.imwrite(postprocess_test_ensemble_path + im[len(pred_test_ensemble_path)+1:], im)

    to_csv(pred_test_ensemble_path, os.path.join(config['log_folder'], 'pred_submission.csv'))
    to_csv(postprocess_test_ensemble_path, os.path.join(config['log_folder'], 'postprocess_submission.csv'))


if __name__ == '__main__':
    # load each config file and run the experiment
    for config_file in glob.glob('config/' + "*.json"):
        config = json.loads(open(config_file, 'r').read())
        name = config['name'] + '_' + datetime.datetime.now().strftime("%m%d_%H_%M_%S")
        config['log_folder'] = 'experiments/'+name
        os.makedirs(config['log_folder'])
        run_experiment(config,prep_experiment)
