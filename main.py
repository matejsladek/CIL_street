from glob import glob
import numpy as np
import tensorflow as tf
import datetime, os
from tensorflow.keras.layers import *
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.optimizers import Adam

from code.preprocessing import *
from code.utils import *
from code.models import *
from code.loss import *
from code.metrics import *

# automatically chooses parameters
AUTOTUNE = tf.data.experimental.AUTOTUNE
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

SEED = 42
dataset_path = "CIL_street/"
training_data = "data/training/images/"
val_data = "data/validation/images/"
IMG_SIZE = 400
N_CHANNELS = 3
N_CLASSES = 2
BUFFER_SIZE = 1000
TRAINSET_SIZE = len(glob(training_data + "*.png"))
VALSET_SIZE = len(glob(val_data + "*.png"))
print(f"The Training Dataset contains {TRAINSET_SIZE} images.")
print(f"The Validation Dataset contains {VALSET_SIZE} images.")

IMG_RESIZE = 384 #384 <- best performing #416
BATCH_SIZE = 4  # CAREFUL! Batch size must be less than validation dataset size

# initialize datasets
train_dataset = tf.data.Dataset.list_files(training_data + "*.png", seed=SEED)
train_dataset = train_dataset.map(parse_image)
train_dataset = train_dataset.map(get_load_image_train(IMG_RESIZE), num_parallel_calls=tf.data.experimental.AUTOTUNE)
train_dataset = train_dataset.repeat().shuffle(buffer_size=BUFFER_SIZE, seed=SEED).batch(BATCH_SIZE).prefetch(buffer_size=AUTOTUNE)

val_dataset = tf.data.Dataset.list_files(val_data + "*.png", seed=SEED)
val_dataset = val_dataset.map(parse_image)
val_dataset = val_dataset.map(get_load_image_test(IMG_RESIZE))
val_dataset = val_dataset.repeat().batch(BATCH_SIZE).prefetch(buffer_size=AUTOTUNE)

EPOCHS = 150
STEPS_PER_EPOCH = max(TRAINSET_SIZE // BATCH_SIZE, 1)
VALIDATION_STEPS = max(VALSET_SIZE // BATCH_SIZE, 1)

N_RUNS=1
for run_id in range(N_RUNS):
    model = PretrainedUnet(backbone_name='seresnext101', input_shape=(IMG_RESIZE, IMG_RESIZE, 3), encoder_weights='imagenet', encoder_freeze=False)
    model.compile(optimizer=Adam(learning_rate=0.0001), loss = 'binary_crossentropy', metrics=['accuracy', kaggle_metric])

    callbacks = [
        tf.keras.callbacks.ModelCheckpoint('best_model'+str(run_id)+'.h5', monitor='val_loss', verbose=1, save_best_only=True, save_weights_only=True),
        tf.keras.callbacks.TensorBoard('log'+str(run_id)),
    ]

    # begin training
    model_history = model.fit(train_dataset, epochs=EPOCHS,
                              steps_per_epoch=STEPS_PER_EPOCH,
                              validation_steps=VALIDATION_STEPS,
                              validation_data=val_dataset,
                              callbacks=callbacks)

    model.load_weights('best_model'+str(run_id)+'.h5')

    # compute and zip predictions
    os.system('rm -r CIL_street/data/output')
    os.system('mkdir CIL_street/data/output')
    save_predictions(model, size=IMG_RESIZE, crop=True)
    os.system('zip -r output' + str(run_id) + '.zip CIL_street/data/output')
    # zip tensorboard log
    os.system('zip -r log' + str(run_id) + '.zip log' + str(run_id))