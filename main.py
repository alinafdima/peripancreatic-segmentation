# ------------------------------------------------------
# File: main.py
# Author: Alina Dima <alina.dima@tum.de>
#
# ------------------------------------------------------

import datetime
import glob
import os
from os.path import join
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Needs to be placed before the tf import
import random
import time
import shutil
import sys
import argparse
import numpy as np
import tensorflow as tf
import tensorflow.keras.backend as K
from tensorflow.python.framework import random_seed
from tensorlayer.cost import dice_hard_coe as tf_dice_metric

# Local imports
from dataset import Dataset
from network import build_model
import test


# Fixing random seeds
random.seed(142)
np.random.seed(142)
random_seed.set_seed(142)

metric_names = ['loss', 'precision', 'recall', 'dice']


def get_experiment_folder():
    return 'experiments'


class DummyCrossValidation():
    def __init__(self, permutation_id):
        self.train_ids = []
        self.test_ids = []
        self.validation_ids = []
        self.permutation_id = permutation_id

    def get_data_paths(self, split, data_source):
        image_paths = [join(split, self.permutation_id, x) for x in ['dummy_image1.nii.gz', 'dummy_image2.nii.gz']]
        segmentation_paths = [join(split, self.permutation_id, x) for x in ['dummy_segmentation1.nii.gz', 'dummy_segmentation2.nii.gz']]
        return [image_paths, segmentation_paths]


class MainLoop(object):
    def __init__(self, options):
        self.cv_id = options.cv
        self.experiment_base = get_experiment_folder()
        self.data_source = options.source
        self.experiment_name = f'final_{options.name}_cv{self.cv_id}'
        self.cross_validation = DummyCrossValidation(self.cv_id)
        self.load_experiment = None
        self.load_model_number = None
        self.experiment_folder = join(self.experiment_base, self.experiment_name)
        self.checkpoint_folder = join(self.experiment_folder, 'checkpoints')
        if not os.path.exists(self.checkpoint_folder):
            os.makedirs(self.checkpoint_folder)

        base_lr = options.base_lr
        # base_lr = 1e-4  # Cross Entropy
        # base_lr = 1e-5  # CL dice   # This is part two of the training
        first_phase = options.ph == 1
        if first_phase is True:
            lr = base_lr
            epochs = 150
            frequency = 10
        else:
            lr = base_lr / 10.
            epochs = 50
            frequency = 5

        self.loss_function_id = options.loss
        self.learning_rate = lr
        self.epochs = epochs
        self.crop_size = (128, 256, 256)
        self.input_size = [None, None, None, 1]
        self.batch_size = 1
        self.save_model_frequency = frequency
        self.test_frequency = frequency
        self.val_frequency = frequency

        print(f'Experiment name: {self.experiment_name}')
        print(f'Data type: {self.data_source}')
        print(f'Cross validation: {self.cv_id}')
        print(f'Loss: {self.loss_function_id}')
        print(f'First phase: {first_phase}')
        print(f'Learning rate: {self.learning_rate}')

        self.log_folder = join(self.experiment_folder, 'logs')
        if not os.path.exists(self.log_folder):
            os.makedirs(self.log_folder)
        timestamp = time.strftime("%d-%m-%Y %H:%M:%S", time.localtime())
        self.log_file = join(self.log_folder, f'log_{timestamp}.txt')
        self.dataset = Dataset(
            batch_size=self.batch_size,
            crop_size=self.crop_size,
            cross_validation=self.cross_validation,
            data_source=self.data_source
        )
        self.train_dataset_size = self.dataset.dataset_size_train
        self.current_epoch = 0
        self.copy_files = ['main.py', 'dataset.py',
                           'network.py', 'test.py']
        self.model, self.callbacks = None, None

    def initialize_network(self, is_training):
        """
        Initializes the network

        - If self.load_experiment is not None, the network will be initialized from an
        experiment folder different from the current experiment. If self.load_experiment
        is not found, the program will assume it was a mistake and exit.
        - If self.load_model_number is not None, then the model will be initailized from
        the model file indicated by the model number. If not found, the program will exit.
        - If no model number is indicated, the program will search in the checkpoint folder
        and load the model with the largest number (the most recent).
        - If no model is found, the network will have a random initialization.

        Args:
            is_training (bool): Only used for copying the source files
        """
        # Check if the model is to be taken from another experiment
        if self.load_experiment is not None:
            checkpoint_folder = join(self.load_experiment, 'checkpoints')
            if not os.path.exists(checkpoint_folder):
                print(f'Checkpoint folder {self.load_experiment} not found. Make sure that the variable load_experiment is set appropriately.')
                sys.exit()
        else:
            checkpoint_folder = self.checkpoint_folder

        # Find the model file to load.
        model_file = None
        if self.load_model_number is not None:
            model_file = join(checkpoint_folder, f'model_{self.load_model_number:03d}.h5')
            self.current_epoch = self.load_model_number
            if not os.path.exists(model_file):
                print(f'Checkpoint file  number {self.load_model_number} not found in the experiment folder {self.load_experiment}.')
                sys.exit()
        else:
            model_files_found = sorted(glob.glob(join(checkpoint_folder, 'model_*.h5')))
            if len(model_files_found) > 0:
                model_file = model_files_found[-1]
                checkpoint_epoch = int(model_file.split('/')[-1].split('.')[0].split('_')[-1])
                self.current_epoch = checkpoint_epoch
            else:
                print('No checkpoint found.')

        # Initialize the network and load the model
        self.model = build_model(self.input_size)
        if model_file is not None:
            print(f'Loading model from checkpoint {self.current_epoch}: {model_file}')
            self.model.load_weights(model_file)

        # Save the training files only when training from scratch or fine-tuning
        if is_training and (self.model is None or self.load_experiment is not None):
            copy_folder = join(self.experiment_base, self.experiment_name, 'source_files')
            if not os.path.exists(copy_folder):
                os.makedirs(copy_folder)
            for file in self.copy_files:
                shutil.copy(file, join(copy_folder, file))

    def compile_model(self) -> None:
        """
        Compiles the model by setting the loss function, optimizer, as well as metrics.
        This step is important for testing as well as training, since the metrics are
        needed in both cases.
        """
        self.optimizer = tf.keras.optimizers.Adam(self.learning_rate)
        self.metrics = [tf.keras.metrics.Precision(), tf.keras.metrics.Recall(), tf_dice_metric]
        self.metrics_names = ['precision', 'recall', 'dice']

        if self.loss_function_id == 'bce':
            self.loss_function = tf.keras.losses.BinaryCrossentropy()
        if self.loss_function_id == 'weighted_bce':
            def weighted_bce(weight_positives, weight_negatives):
                def wbce(y_true, y_pred, weight1=1, weight0=1):
                    y_true = K.clip(y_true, K.epsilon(), 1 - K.epsilon())
                    y_pred = K.clip(y_pred, K.epsilon(), 1 - K.epsilon())
                    logloss = -(y_true * K.log(y_pred) * weight1 + (1 - y_true) * K.log(1 - y_pred) * weight0)
                    return K.mean(logloss)
                return lambda x, y: wbce(x, y, weight_positives, weight_negatives)
            self.loss_function = weighted_bce(0.7, 0.3)
        elif self.loss_function_id == 'focal_loss':
            from focal_loss import BinaryFocalLoss
            self.loss_function = BinaryFocalLoss(gamma=2)
            # self.loss_function = tfa.losses.SigmoidFocalCrossEntropy()
        else:
            Exception(f'Unknown loss function {self.loss_function_id}')

        self.model.compile(loss=self.loss_function,
                           optimizer=self.optimizer, metrics=self.metrics)

    def train_custom_loop(self):
        if self.model is None:
            self.initialize_network(is_training=True)
            self.compile_model()
        model = self.model

        dataset = self.dataset.get_dataset('train')
        train_epochs = self.epochs
        metrics_names = ['loss'] + self.metrics_names
        output_file = join(self.checkpoint_folder, 'data.csv')
        val_output_file = join(self.checkpoint_folder, 'data_val.csv')
        self.starting_epoch = self.current_epoch + 1
        folder_train_input = join(self.experiment_folder, 'train_image_pairs')
        if not os.path.exists(folder_train_input):
            os.makedirs(folder_train_input)
        if not(os.path.exists(output_file)):
            with open(output_file, 'a') as f:
                metrics_write = ', '.join(
                    ['epoch', 'loss'] + self.metrics_names)
                f.write(f'{metrics_write}\n')
        log_dir = join(self.experiment_folder, 'logs', 'train', datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
        summary_writer = tf.summary.create_file_writer(logdir=log_dir)
        self.step = 0
        test_object = test.Testing(self)

        print(f'Starting training at epoch {self.current_epoch}')
        for epoch in range(train_epochs):
            t0 = time.time()
            self.current_epoch = self.starting_epoch + epoch
            current_time = time.strftime("%d-%m-%Y %H:%M:%S", time.localtime())
            print(f"\n[{current_time}] Start of epoch {self.current_epoch}")

            epoch_metrics = []
            for step, (X, Y, _) in enumerate(dataset):
                self.step += 1
                with tf.GradientTape() as tape:
                    pred = model(X, training=True)
                    loss_value = self.loss_function(Y, pred)
                grads = tape.gradient(loss_value, model.trainable_weights)
                self.optimizer.apply_gradients(zip(grads, model.trainable_weights))
                iter_metrics = [loss_value.numpy()] + [metric(Y, pred).numpy()
                                                       for metric in self.metrics]
                epoch_metrics.append(iter_metrics)
                current_time = time.strftime("%H:%M:%S", time.localtime())
                metrics_print = ' '.join([f'{val:.4f},' for val in iter_metrics[1:]])
                print(
                    f'[{current_time}] Iter {step+1}: loss {loss_value:.4f}, {metrics_print}')

            metric_averages = [np.mean([row[i] for row in epoch_metrics])
                               for i in range(len(epoch_metrics[0]))]
            metrics_display = ' | '.join(
                [f'{n}: {v:.6f}' for n, v in zip(metrics_names, metric_averages)])
            t1 = time.time()
            epoch_time = t1 - t0
            current_time = time.strftime("%H:%M:%S", time.localtime())
            print(
                f'[{current_time}] Epoch {epoch+1} ({epoch_time:.2}s) >> {metrics_display}')

            # Save metrics to disk
            with open(output_file, 'a') as f:
                metrics_write = ', '.join(
                    [f'{value:.6f}' for value in metric_averages])
                f.write(f'{self.current_epoch}, {metrics_write}\n')

            # Save model
            if (self.current_epoch) % self.save_model_frequency == 0 or epoch == train_epochs:
                model_file = join(self.checkpoint_folder,
                                  f'model_{self.current_epoch:03d}.h5')
                model.save_weights(model_file)

            # Tensorboard
            with summary_writer.as_default():
                for m_name, m_avg in zip(metric_names, metric_averages):
                    tf.summary.scalar(f'epoch_{m_name}', m_avg, step=self.current_epoch)

            # Validation
            if (self.current_epoch) % self.val_frequency == 0 or epoch == train_epochs:
                metric_averages_val = test_object.test(split='val', save_images=False)
                with open(val_output_file, 'a') as f:
                    metrics_write = ', '.join(
                        [f'{value:.6f}' for value in metric_averages_val])
                    f.write(f'{self.current_epoch}, {metrics_write}\n')
                with summary_writer.as_default():
                    for m_name, m_avg in zip(metric_names, metric_averages_val):
                        tf.summary.scalar(f'val_{m_name}', m_avg, step=self.current_epoch)

            if (self.current_epoch) % self.test_frequency == 0 or epoch == train_epochs:
                test_object.test(split='test', save_images=True)

    def test_checkpoints(self):
        for checkpoint_no in [200, 190]:
            self.load_model_number = checkpoint_no
            self.initialize_network(is_training=True)
            self.compile_model()
            test_object.test(split='test')


def get_command_line_arguments():
    parser = argparse.ArgumentParser('main.py')
    parser.add_argument('--name', default='unnamed', type=str, help='The name of the experiment')
    parser.add_argument('--source', default='iodine', type=str, help='iodine, arterial, or joint')
    parser.add_argument('--cv', default=1, type=int, help='1, 2, 3 or 4')
    parser.add_argument('--ph', default=1, type=int, help='1 or 2')
    parser.add_argument('--loss', default='bce', type=str, help="bce, weighted_bce, focal_loss")
    parser.add_argument('--base_lr', default=1e-4, type=float, help='egal')

    args = parser.parse_args()
    return args


if __name__ == '__main__':
    options = get_command_line_arguments()
    gpus = tf.config.experimental.list_physical_devices('GPU')
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
    print('')
    print('Call to main.py')

    loop = MainLoop(options)
    print(f'Experiment {loop.experiment_name}')
    loop.train_custom_loop()

    test_object = test.Testing(loop)
    test_object.test(split='test', save_images=True)

    print('Finished\n')
