# Group project for CIL 2020 @ ETHZ - Road segmentation in aerial images - Team NaN

## Overview
This codebase implements the final methods used by our team for the project, although we experimented with different approaches.
The final model is a DCNN with an encoder-decoder structure. More information can be found on the report.

## How to get the data
Training datasets should be added to `data` folder. In particular the 100 training images provided for the project should
be added to `data/original/all/images` while their masks should be in `data/original/all/groundtruth`. Additionally,
test samples should be added to `data/test_images`. Our additional dataset can be downloaded from Polybox. It should be 
placed in `data/maps1800/all/`. Without it, the models can only be trained on the original
dataset (`dataset = 'original'`).

## Execution
The main results can be reproduced by running the training script `train.py`. This script loads
.json configurations from a folder and reproduces one experiment for each configuration. Each experiment
consists of training, validation and testing. It outputs the model's predictions on a test set
both as .png images and in .csv format. This, together with additional information such as Tensorboard
logs and validation scores is saved in a folder in ./experiments. To run this with the default config, simply use:

```python train.py --config_dir config```

Note that this config trains a single network, while for reproducing our best submission you should set `use_ensemble`
to true in the config file. We provide more configs in the folder `config_archive`.

To reproduce crossvalidation scores you can use:

```python train_cv.py --config_dir config```

Other useful scripts are contained in the scripts folder and can be used for analyzing validation scores and
generating config files.

## Additional info
Single experiments can be entirely configured via .json files. Please refer to `config/best_model.json` for an example.
The following table explains the single .json attributes.

| Parameters    | Type          | Usage         |
| ------------- | ------------- | ------------- |
| name | string | output folder name |
| dataset | string | name of the dataset to use |
| img_resize | int | size of the model's input and output, must be a multiple of 16 |
| normalize | boolean | normalizes the model's input |
| v_flip | float | probability to vertically flip a single input image |
| h_flip | float | probability to horizontally flip a single input image |
| rot | float | probability to rotate a single input image at aright angle |
| contrast | float | intensity of random contrast augmentation |
| brightness | float | intensity of random brightness augmentation |
| hard_mask | boolean | thresholds the groundtruth mask to 0/1 |
| backbone | string | name of the encoder backbone |
| pretrained | boolean | initializes pretrained weights |
| aspp | boolean | enables ASPP |
| residual | boolean | esables residual connections in the decoder |
| art | boolean | enables aggregated residual trnsformations in the decoder |
| se | boolean | enables squeeze and excitation modules in the decoder |
| batch_size | int | batch size |
| epochs | int | training epochs |
| optimizer | string | optimizer name |
| learning_rate | float | learning rate for the optimizer |
| loss | string | name of loss for segmentation |
| augment_loss | boolean | adds a quadratic loss term that pushes predictions to 0 or 1 |
| stop_on_metric | boolean | uses patch-wise accuracy instead of loss for model saving |
| custom_callback | boolean | uses a custom callback instead of tf.keras.callbacks.ModelCheckpoint |
| restore_best_model | boolean | restores the weight from the epoch with highest validation metric/loss |
| save_model | boolean | enables model saving |
| postprocess | string | name of postprocessing technique to apply |
| n_ensemble | int | number of networks to train for ensemble method |
| use_ensemble | boolean | enables ensemble learning |
| cv_k | int | folds of CV |
| cv_k_todo | int | folds of CV |
| val_split | float | fraction of data to use or validation |
| use_baseline_code1 | boolean | replaces our model with logistic regression baseline |
| use_baseline_code2 | boolean | replaces our model with the second baseline (not reported on)|
| seed | int | random seed for the scripts |
