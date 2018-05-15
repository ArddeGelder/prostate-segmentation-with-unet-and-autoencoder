import os
import numpy as np
from random import shuffle

import keras
from keras.models import load_model
from keras.utils import to_categorical

from utils_py.write_utils import pickle_dump, pickle_load
from utils_py.augment_utils import augment_data


class BatchGenerator(object):
    """
    Creates Python generators that are input to Keras model.fit_generator to train the model.
    The generators indefinitely output batches of (training or validation) volumes in the form of numpy arrays.
    """

    def __init__(self, hdf5_file, config_data, encoding_shape, train_model=False, n_labels=1, labels=None):
        """
        :param hdf5_file: hdf5 data file that holds the training data numpy arrays.
        :param create_model: If set to True, previous files will be overwritten. The default mode is false,
        so that the training and validation splits won't be overwritten when rerunning model training.
        :param n_labels: Number of binary labels.
        :param labels: List or tuple containing the ordered label values in the image files. The length of the list
        or tuple should be equal to the n_labels value. Example: (10, 25, 50). The data generator would then return
        binary truth arrays representing the labels 10, 25, and 30 in that order.
        """
        self.hdf5_file = hdf5_file
        self.config_data = config_data
        self.n_labels = n_labels
        self.train_model = train_model
        self.labels = labels
        self.encoder = load_model(config_data['encoder_file'])
        #self.encoder_output_shape = self.encoder.output_shape
        self.encoder._make_predict_function()
        self.encoding_shape = encoding_shape
        #self.encoder.summary()

    def get_kfold_split(self, kfold_ids_file, shuffle_list=True, kfold=5):
        """
        Splits the list of volumes into one list of training volumes and one list of validation volumes,
        with the ratio given by the parameter training_validation_split.
        :param kfold_ids_file: Pickle file where the index locations of the training data will be stored.
        :param shuffle_list: if True the list of volumes from the hdf5_file are shuffled.
        :param kfold: Number of cross validation data set splits.
        :return: kfold list of cross validation file names.
        """
        if self.train_model or not os.path.exists(kfold_ids_file):
            print("Creating validation split...")
            nb_samples = self.hdf5_file.root.img.shape[0]
            sample_list = list(range(nb_samples))
            if shuffle_list:
                shuffle(sample_list)
            nsample_kfold = int(len(sample_list) / kfold)
            kfold_lists = [None] * kfold
            for k in range(kfold-1):
                kfold_lists[k] = sample_list[k * nsample_kfold:(k+1) * nsample_kfold]
                print('KFOLD LIST #' + str(k) + ': ' + str(kfold_lists[k]) + '\n')
            kfold_lists[kfold-1] = sample_list[(kfold - 1) * nsample_kfold: nb_samples]
            print('KFOLD LIST #' + str(kfold-1) + ': ' + str(kfold_lists[kfold-1]) + '\n')
            pickle_dump(kfold_lists, kfold_ids_file)

        else:
            print("Loading previous validation split...")
            kfold_lists = pickle_load(kfold_ids_file)

        return kfold_lists

    def get_data_generator(self, index_list, batch_size=1):
        """
        Constructs a Python generator for batches of volumes in the form of numpy arrays of dimensions
        (batch_size, n_channel, nx_img, ny_img, nz_img)
        :param index_list: list of data file names that are to be grouped in batches.
        :param batch_size: size of the batches that the training generator will provide.
        :return: python data generator that will output batches by looping indefinitely over the data files
        in the index_list.
        """
        while True:
            img_list = list()
            gt_list = list()
            gt_encoded_list = list()
            shuffle(index_list)
            for index in index_list:
                self.add_data_to_batch(self.hdf5_file, index, img_list, gt_list, gt_encoded_list)
                if len(img_list) == batch_size:
                    img = np.asarray(img_list)
                    gt = np.asarray(gt_list)
                    gt_encoded_list = np.asarray(gt_encoded_list)
                    yield img, [gt, gt_encoded_list]
                    #yield self.convert_to_one_hot(img_list, gt_list)
                    img_list = list()
                    gt_list = list()
                    gt_encoded_list = list()

    def add_data_to_batch(self, hdf5_file, index, img_list, gt_list, gt_encoded_list):
        """
        Adds data from the hdf5_file to the given lists of image and ground truth data
        """
        img = hdf5_file.root.img[index]
        gt = hdf5_file.root.gt[index]
        if self.config_data['augment']:
            img, gt = augment_data(img, gt, self.config_data)
        img_list.append(img)
        gt_one_hot = keras.utils.to_categorical(gt, num_classes=self.n_labels+1)
        gt_reshaped = gt_one_hot.reshape((1,) + gt_one_hot.shape)
        gt_encoded = self.encoder.predict(x=gt_reshaped)
        gt_encoded = gt_encoded.reshape(self.encoding_shape)
        gt_list.append(gt_one_hot)
        gt_encoded_list.append(gt_encoded)

    def convert_to_one_hot(self, img_list, gt_list):
        img = np.asarray(img_list)
        gt = np.asarray(gt_list)
        #gt_shape = gt.shape[:-1]
        #gt = keras.utils.to_categorical(gt, num_classes=self.n_labels+1).reshape(gt_shape + (self.n_labels + 1,))

        return img, gt


class BatchGenerator_for_encoder(object):

    def __init__(self, hdf5_file, config_data):
        self.hd5f_file = hdf5_file
        self.config_data = config_data
        self.n_labels = config_data['n_labels']

    def get_index_list(self):
        nb_samples = self.hd5f_file.root.img.shape[0]
        sample_list = list(range(nb_samples))
        shuffle(sample_list)
        return sample_list

    def get_data_generator(self, index_list, batch_size=1):
        while True:
            img_list = list()
            gt_list = list()
            shuffle(index_list)
            for index in index_list:
                self.add_data_to_batch(self.hd5f_file, index, img_list, gt_list)
                if len(img_list) == batch_size:
                    img = np.asarray(img_list)
                    gt = np.asarray(gt_list)
                    yield img, gt
                    img_list = list()
                    gt_list = list()

    def add_data_to_batch(self, hdf5_file, index, img_list, gt_list):
        img = hdf5_file.root.img[index]
        gt = hdf5_file.root.gt[index]
        if self.config_data['augment']:
            img, gt = augment_data(img, gt, self.config_data)
        gt_cat = to_categorical(gt, num_classes=self.n_labels+1)

        img_list.append(gt_cat) # bij encoder is de gt_cat ook de input img
        gt_list.append(gt_cat)
