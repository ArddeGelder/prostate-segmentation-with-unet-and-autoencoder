import os
import math

from functools import partial
from keras.callbacks import ModelCheckpoint, CSVLogger, Callback, LearningRateScheduler, TensorBoard

from utils_py.write_utils import pickle_dump


class TrainManager(object):
    """
    Module that defines settings for model.fit_generator.
    """
    def __init__(self):
        pass

    def step_decay(self, epoch, initial_lrate, drop, epochs_drop):
        # Define learning rate schedule for model.fit_generator.
        return initial_lrate * math.pow(drop, math.floor((1 + epoch) / float(epochs_drop)))

    def get_callbacks(self, model_file, initial_learning_rate, learning_rate_drop, learning_rate_epochs,
                      logging_dir="./log", ifold=0):
        # Define callbacks for model.fit_generator.
        model_checkpoint = ModelCheckpoint(model_file, save_best_only=False)
        logger = CSVLogger(os.path.join(logging_dir, "training_fold" + str(ifold) + ".log"))
        history = SaveLossHistory()
        scheduler = LearningRateScheduler(partial(self.step_decay,
                                                  initial_lrate=initial_learning_rate,
                                                  drop=learning_rate_drop,
                                                  epochs_drop=learning_rate_epochs))
        tensor_board = TensorBoard(log_dir=logging_dir, histogram_freq=0, batch_size=32,
                                   write_graph=False, write_grads=False, write_images=False,
                                   embeddings_freq=0, embeddings_layer_names=None, embeddings_metadata=None)
        return [model_checkpoint, logger, history, scheduler, tensor_board]


class SaveLossHistory(Callback):
    # Save losses to pickle file.
    def on_train_begin(self, logs={}):
        self.losses = []

    def on_batch_end(self, batch, logs={}):
        self.losses.append(logs.get('loss'))
        pickle_dump(self.losses, "loss_history.pkl")


class TrainManager_for_encoder(object):
    """
    Module that defines settings for model.fit_generator.
    """
    def __init__(self):
        pass

    def step_decay(self, epoch, initial_lrate, drop, epochs_drop):
        # Define learning rate schedule for model.fit_generator.
        return initial_lrate * math.pow(drop, math.floor((1 + epoch) / float(epochs_drop)))

    def get_callbacks(self, model_file="encoder_new.h5", initial_learning_rate=0.0001, learning_rate_drop=0.1, learning_rate_epochs=50,
                      logging_dir="./log", ifold=0):
        # Define callbacks for model.fit_generator.
        model_checkpoint = ModelCheckpoint(model_file, save_best_only=False)
        logger = CSVLogger(os.path.join(logging_dir, "training_fold" + str(ifold) + ".log"))
        history = SaveLossHistory()
        scheduler = LearningRateScheduler(partial(self.step_decay,
                                                  initial_lrate=initial_learning_rate,
                                                  drop=learning_rate_drop,
                                                  epochs_drop=learning_rate_epochs))
        tensor_board = TensorBoard(log_dir=logging_dir, histogram_freq=0, batch_size=32,
                                   write_graph=False, write_grads=False, write_images=False,
                                   embeddings_freq=0, embeddings_layer_names=None, embeddings_metadata=None)
        return [model_checkpoint, logger, history, scheduler, tensor_board]

