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


def get_dataset(config, autotune):
    if config['dataset'] == 'original':
        training_data = "data/original/training/images/"
    else:
        raise Exception('Unrecognised dataset')
    val_data = training_data.replace('training', 'validation')
    trainset_size = len(glob.glob(training_data + "*.png"))
    valset_size = len(glob.glob(val_data + "*.png"))

    train_dataset = tf.data.Dataset.list_files(training_data + "*.png", seed=config['seed'])
    train_dataset = train_dataset.map(parse_image)
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

    val_dataset = tf.data.Dataset.list_files(val_data + "*.png", seed=config['seed'])
    val_dataset = val_dataset.map(parse_image)
    val_image_loader = get_load_image_val(size=config['img_resize'], normalize=config['normalize'],
                                          predict_contour=config['predict_contour'],
                                          predict_distance=config['predict_distance'])
    val_dataset = val_dataset.map(val_image_loader)
    val_dataset = val_dataset.repeat().batch(config['batch_size']).prefetch(buffer_size=autotune)

    return train_dataset, val_dataset, trainset_size, valset_size, training_data, val_data


def get_model(config):
    backbone = config['backbone']
    learning_rate = config['learning_rate']
    encoder_weights = None
    if config['pretrained']:
        encoder_weights = 'imagenet'

    model = PretrainedUnet(backbone_name=backbone,
                           input_shape=(config['img_resize'], config['img_resize'], config['n_channels']),
                           encoder_weights=encoder_weights, encoder_freeze=False,
                           predict_distance=config['predict_distance'], predict_contour=config['predict_contour'])

    if config['predict_distance'] and config['predict_contour']:
        model.compile(optimizer=Adam(learning_rate=learning_rate), loss=config['loss'],
                      loss_weights=config['loss_weights'], metrics=['accuracy', kaggle_metric])
    elif config['predict_distance']:
        model.compile(optimizer=Adam(learning_rate=learning_rate), loss=config['loss'][0:1],
                      loss_weights=config['loss_weights'][0:1], metrics=['accuracy', kaggle_metric])
    elif config['predict_contour']:
        model.compile(optimizer=Adam(learning_rate=learning_rate), loss=config['loss'][0:1],
                      loss_weights=config['loss_weights'][0:1], metrics=['accuracy', kaggle_metric])
    else:
        model.compile(optimizer=Adam(learning_rate=learning_rate), loss=config['loss'][0],
                      metrics=['accuracy', kaggle_metric])

    return model


def get_postprocess(config):
    if config['postprocess'] == 'morphological':
        return morphological_postprocessing
    elif config['postprocess'] == 'none':
        return no_postprocessing
    raise Exception('Unknown postprocessing')


def compute_metric(metric, in_path, gt_path):
    img_paths = glob.glob(in_path + '/*.png')
    images = []
    for img_path in img_paths:
        images.append(cv2.imread(img_path, 0))
    images = np.stack(images, axis=0)

    gt_paths = glob.glob(gt_path + '/*.png')
    gt = []
    for gt_path in gt_paths:
        gt.append(cv2.imread(gt_path, 0))
    gt = np.stack(gt, axis=0)

    return metric(gt, images)


def run_experiment(config):

    autotune = tf.data.experimental.AUTOTUNE
    prepare_gpus()
    train_dataset, val_dataset, trainset_size, valset_size, training_data, val_data = get_dataset(config, autotune)

    print(f"Training dataset contains {trainset_size} images.")
    print(f"Validation dataset contains {valset_size} images.")
    steps_per_epoch = max(trainset_size // config['batch_size'], 1)
    validation_steps = max(valset_size // config['batch_size'], 1)

    model = get_model(config)
    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(config['log_folder'] + '/best_model.h5', monitor='val_loss', verbose=1, save_best_only=True, save_weights_only=True),
        tf.keras.callbacks.TensorBoard(config['log_folder'] + '/log'),
    ]

    print('Begin training for ' + config['name'])
    model_history = model.fit(train_dataset, epochs=config['epochs'],
                              steps_per_epoch=steps_per_epoch,
                              validation_steps=validation_steps,
                              validation_data=val_dataset,
                              callbacks=callbacks)

    model.load_weights(config['log_folder'] + '/best_model.h5')
    postprocess = get_postprocess(config)

    in_val_path = val_data
    gt_val_path = val_data.replace('images', 'groundtruth')
    pred_val_path = os.path.join(config['log_folder'], "pred_val")
    os.mkdir(pred_val_path)
    postprocess_val_path = os.path.join(config['log_folder'], "postprocess_val")
    os.mkdir(postprocess_val_path)

    in_test_path = 'data/test_images'
    pred_test_path = os.path.join(config['log_folder'], "pred_test")
    os.mkdir(pred_test_path)
    postprocess_test_path = os.path.join(config['log_folder'], "postprocess_test")
    os.mkdir(postprocess_test_path)

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
    out_file.write("Validation before post processing:\n")
    val_score = compute_metric(np_kaggle_metric, pred_val_path, gt_val_path)
    out_file.write(str(val_score) + "\nValidation after post processing:\n")
    postprocess(pred_val_path, postprocess_val_path)
    val_score = compute_metric(np_kaggle_metric, postprocess_val_path, gt_val_path)
    out_file.write(str(val_score))
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


if __name__ == '__main__':
    for config_file in glob.glob('config/' + "*.json"):
        config = json.loads(open(config_file, 'r').read())
        name = config['name'] + '_' + datetime.datetime.now().strftime("%m%d_%H_%M_%S")
        config['log_folder'] = 'experiments/'+name
        os.mkdir(config['log_folder'])
        run_experiment(config)
