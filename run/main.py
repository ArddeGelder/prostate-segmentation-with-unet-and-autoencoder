import os
import sys
from time import time
sys.path.append("../")

import csv
import pprint
import matplotlib
matplotlib.use('Agg')

import numpy as np
import tables
from keras.utils import to_categorical
from keras.models import load_model

from model_py.VnetBuilder_same_padding import VnetBuilder
from model_py.TestManager import TestManager
from model_py.TrainManager import TrainManager, TrainManager_for_encoder
from data_py.DataLoader import DataLoader
from data_py.Hdf5Writer import Hdf5Writer
from data_py.BatchGenerator import BatchGenerator, BatchGenerator_for_encoder
from encoder_py.EncoderBuilder import EncoderBuilder
from utils_py.visualize_utils import show_hdf5_slices, show_train_slices, show_test_slices
from utils_py.write_utils import create_output_folders, write_predictions_to_sitk, write_slices_to_sitk
from config import config

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


def main():
    global hdf5_file_opened

    #Import run configuration data
    create_hdf5 = config['crun']['create_hdf5']
    build_new_encoder = config['crun']['build_new_encoder']
    train_encoder = config['crun']['train_encoder']
    test_encoder = config['crun']['test_encoder']
    build_new_unet = config['crun']['build_new_unet']
    train_unet = config['crun']['train_unet']
    train_encoder_with_unet = config['crun']['train_encoder_with_unet']
    cross_validation = config['crun']['cross_validation']
    test_unet = config['crun']['test_unet']
    visualize = config['crun']['visualize']

    # Create directory structure if it does not yet exist
    create_output_folders(config)

    n_labels = config['data']['n_labels']

    if create_hdf5:
        # Load the training data and write to hdf5 file
        print("Loading training data from ", config['data']['train_dir'], "...")
        print("Grid resolution: ", config['data']['grid_resolution'])
        print("Grid size: ", config['data']['output_shape'])
        data_loader_train = DataLoader(config_data=config['data'])
        data_loader_train.load_train_data()

        print("Saving training data as hdf5 file")
        hdf5_writer_train = Hdf5Writer(data_loader_train, config['data'], type='train')
        hdf5_writer_train.write_hdf5_file(with_gt=True)

        print("Loading test data from ", config['data']['test_dir'], "...")
        data_loader_test = DataLoader(config_data=config['data'])
        data_loader_test.load_test_data(with_gt=True)

        print("Saving test data as hdf5 file")
        hdf5_writer_test = Hdf5Writer(data_loader_test, config['data'], type='test')
        hdf5_writer_test.write_hdf5_file(with_gt=True)

        print()
        print("---------")
        print()

    # Build an encoder
    encoder_file = config['encoder']['file']
    encoder_decoder_file = config['encoder']['file_with_decoder']
    if build_new_encoder:
        print("Building new encoder...")
        max_filters = config['encoder']['max_filters']
        model, encoder, encoding_shape = EncoderBuilder(config['model']).build_convolution_model(max_filters=max_filters)
        model.save(encoder_decoder_file)
        encoder.save(encoder_file)
    else:
        print("Loading encoder from", encoder_file, "...")
        model = load_model(encoder_decoder_file)
        encoder = load_model(encoder_file)
        encoding_shape = (9, 9, 5, 3) # Bugged
    print()
    print("----------")
    print()

    if train_encoder:
        n_encoder_epochs = config['encoder']['n_epochs']

        print("Training encoder...")
        tic = time()
        hdf5_file_opened = tables.open_file(config['data']['hdf5_train'], "r")  # Open hdf5 file in read mode
        batch_generator = BatchGenerator_for_encoder(hdf5_file=hdf5_file_opened,
                                                    config_data=config['data'])
        #batch_size = 1
        index_list = batch_generator.get_index_list()
        nb_samples = len(index_list)
        training_batch_generator = batch_generator.get_data_generator(index_list=index_list)
        train_manager = TrainManager_for_encoder()

        model.fit_generator(generator=training_batch_generator,
                            steps_per_epoch=nb_samples,
                            epochs=n_encoder_epochs,
                            callbacks=train_manager.get_callbacks())

        toc = time()
        print("Training the model took", (toc - tic) / 60, "min")
        print("Saving the trained model as", encoder_file, "...")
        model.save(encoder_decoder_file)
        encoder.save(encoder_file)
        hdf5_file_opened.close()
        print()
        print("----------")
        print()

    if test_encoder:
        print("Testing encoder...")
        # Read the test data
        hdf5_file_opened = tables.open_file(config['data']['hdf5_test'], "r")  # Open hdf5 file in read mode
        img_test = hdf5_file_opened.root._v_children['img'][:, :, :, :, :]
        gt_test = hdf5_file_opened.root._v_children['gt'][:, :, :, :, :]
        cat_shape = gt_test.shape[:4] + (n_labels + 1,)
        gt_test_cat = to_categorical(y=gt_test, num_classes=n_labels + 1).reshape(cat_shape)
        hdf5_file_opened.close()
        n_test_images = img_test.shape[0]

        model_test_predictions_cat = model.predict(x=gt_test_cat)
        model_test_predictions = np.argmax(model_test_predictions_cat, axis=4).astype(np.float32)

        print("Saving test predictions as itk...")
        print()
        for i in range(n_test_images):
            write_slices_to_sitk(gt_test[i, :, :, :, 0], 'test_' + str(i))
            write_slices_to_sitk(model_test_predictions[i, :, :, :], 'test_' + str(i) + '_reconstruction')

        print("Saving test encodings as itk...")
        print()
        encoded_images = encoder.predict(x=gt_test_cat)
        for i in range(n_test_images):
            write_slices_to_sitk(encoded_images[i, :, :, :, :], 'test_' + str(i) + '_encoded')

        def dice(y_true_cat, y_pred_cat):
            n_labels = y_true_cat.shape[3]
            dice = np.zeros(shape=(n_labels))
            # Normalize y_pred
            y_pred_cat /= np.sum(y_pred_cat, axis=3, keepdims=True)

            for i in range(n_labels):
                slice_true = y_true_cat[:, :, :, i]
                slice_pred = y_pred_cat[:, :, :, i]
                intersection = slice_true * slice_pred
                dice[i] = 2. * np.sum(intersection) / (np.sum(slice_true) + np.sum(slice_pred))
            return dice

        print("Calculating DICE scores for test predictions...")
        dice_score = np.zeros(shape=(n_test_images, n_labels + 1))
        for i in range(n_test_images):
            dice_score[i, :] = dice(gt_test_cat[i, :, :, :, :], model_test_predictions_cat[i, :, :, :, :])
            print("DICE score for test ", i, ":", dice_score[i, :])
        print("Average DICE score label 1:", "%.3f" % np.mean(dice_score[:, 1]))
        print("Average DICE score label 2:", "%.3f" % np.mean(dice_score[:, 2]))
        print("Average DICE score:", "%.3f" % np.mean(dice_score[:, 1:3], axis=(0, 1)))
        print()

    print()
    print("Done")
    print()

    """
    Classes that are used in this code:
    DataLoader: Loads data volumes in ITK format, and pre-processes data (resampling, cropping, normalization).
    HdfWriter: Writes data volume numpy arrays to file in hdf5 format.
    VnetBuilder: Builds a VNet Keras model with build_model method.
    TrainManager: Manages training, defines callbacks for Keras model.fit_generator.
    TestManager: Manages testing.
    """
    vnet_builder = VnetBuilder(config_data=config['data'],
                               wt_fac=config['model']['loss_weight_factors'],
                               train_encoder_with_unet=train_encoder_with_unet)
    train_manager = TrainManager()
    test_manager = TestManager()

    pprint.pprint(config)

    if build_new_unet:
        # Build a new Vnet model
        print("Building new model...")
        model_with_encoder, model_without_encoder = vnet_builder.build_model(input_shape=config['model']['input_shape'],
                                                                             padding=config['model']['padding'],
                                                                             downsize_filters_factor=config['model'][
                                                                                 'downsize_nb_filters_factor'],
                                                                             pool_size=config['model']['pool_size'],
                                                                             initial_learning_rate=config['model'][
                                                                                 'initial_learning_rate'])
    else:
        # Read a pre-trained model from file
        # Doesn't work with encoder
        metrics = []
        for label in range(config['data']['n_labels']):
            vnet_builder.dice_coef = vnet_builder.create_dice_coef(label)
            vnet_builder.dice_coef.__name__ = 'dice_coef_lbl' + str(label+1)
            metrics.append(vnet_builder.dice_coef)
        if not os.path.exists(config['model']['pretrained_model']):
            sys.exit("Cannot find model " + config['model']['pretrained_model'])
        print("Loading pre-trained model ", config['model']['pretrained_model'], " ...")
        # Custom loss functions are not stored in a pre-trained model, and are redefined here.
        custom_objects = {'weighted_categorical_crossentropy': vnet_builder.weighted_categorical_crossentropy}
        for label in range(1, config['data']['n_labels'] + 1):
            custom_objects.update({'dice_coef_lbl' + str(label): metrics[label - 1]})
        model = load_model(config['model']['pretrained_model'], custom_objects=custom_objects)

    # Save initial model weights to use for reinitialisation later
    model_initial_weights = model_with_encoder.get_weights()

    """
    Train a Keras model with model.fit_generator(params) where params are:
    :param model: Keras model that will be trained. 
    :param model_file: Where to save the Keras model.
    :param training_generator: Generator that iterates through the training data.
    :param validation_generator: Generator that iterates through the validation data.
    :param steps_per_epoch: Number of batches that the training generator will provide during a given epoch.
    :param validation_steps: Number of batches that the validation generator will provide during a given epoch.
    :param initial_learning_rate: Learning rate at the beginning of training.
    :param learning_rate_drop: How much at which to the learning rate will decay.
    :param learning_rate_epochs: Number of epochs after which the learning rate will drop.
    :param n_epochs: Total number of epochs to train the model.
    :return: 
    """

    if train_unet:
        # Open an hdf5 file with the training data
        hdf5_file_opened = tables.open_file(config['data']['hdf5_train'], "r")  # Open hdf5 file in read mode

        if visualize:
            print("Showing some slices from hdf5 file to check if volumes are retrieved ok...")
            show_hdf5_slices(config['data']['n_labels'], config['data']['figures_dir'] + "/img_hdf5",
                             config['data']['padding'], hdf5_file_opened)

        print("Getting training and validation batch generators...")
        batch_generator = BatchGenerator(hdf5_file=hdf5_file_opened,
                                         config_data=config['data'],
                                         encoding_shape=encoding_shape,
                                         train_model=train_unet,
                                         n_labels=config['model']['n_labels'])

        # training_list, validation_list = batch_generator.get_validation_split(
        kfold = config['model']['cross_validation_kfold']
        kfold_lists=batch_generator.get_kfold_split(
            kfold_ids_file=config['model']['kfold_ids_file'],
            shuffle_list=True,
            kfold=kfold)

        if cross_validation:
            jfold = kfold
        else:
            jfold = 0

        for ifold in range(jfold+1):
            if jfold == 0:
                # Single validation on first fold
                training_list = [item for jfold in range(kfold) for item in kfold_lists[jfold] if jfold != ifold]
                validation_list = kfold_lists[ifold]
                print('SINGLE VALIDATION ' + str(ifold) + '\n')
                print('TRAINING FILE LIST: ' + str(training_list) + '\n')
                print('VALIDATION FILE LIST: ' + str(validation_list) + '\n')
            elif ifold < jfold:
                # k-fold cross-validation
                training_list = [item for jfold in range(kfold) for item in kfold_lists[jfold] if jfold != ifold]
                validation_list = kfold_lists[ifold]
                print('K-FOLD CROSS-VALIDATION, FOLD ' + str(ifold) + '\n')
                print('TRAINING FILE LIST: ' + str(training_list) + '\n')
                print('VALIDATION FILE LIST: ' + str(validation_list) + '\n')
            else:
                # Final training on all volumes after k-fold cross-validation
                training_list = [item for jfold in range(kfold) for item in kfold_lists[jfold]]
                validation_list = []
                print('FINAL TRAINING ON ALL VOLUMES, AFTER K-FOLD CROSS-VALIDATION \n')
                print('TRAINING FILE LIST: ' + str(training_list) + '\n')
                print('VALIDATION FILE LIST: ' + str(validation_list) + '\n')

            batch_size = config['model']['batch_size']
            # Python generator to provide batches of training data as input to model.fit_generator
            training_batch_generator = batch_generator.get_data_generator(index_list=training_list,
                                                                          batch_size=batch_size)
            # Python generator to provide batches of validation data as input to model.fit_generator
            validation_batch_generator = batch_generator.get_data_generator(index_list=validation_list,
                                                                            batch_size=batch_size)
            nb_train_samples = len(training_list) // batch_size
            nb_validation_samples = len(validation_list) // batch_size

            print("Showing some training slices to check if input is generated ok...")
            if visualize:
                show_train_slices(config['data']['n_labels'], config['data']['figures_dir'] + "/img_gt",
                                  config['data']['padding'], training_batch_generator)


            '''
            Train the model, output a trained model config['model']['newmodel_file'] and scores for Tensorboard: 
            Run command 'tensorboard --config['data']['logging_dir']' and open webpage 'localhost:6006'.
            '''
            print("Training model...")
            # For info about inputs and outputs of fit_generator, see Keras model API at https://keras.io/models/model/
            if ifold == kfold:
                validation_data = None
            else:
                validation_data = validation_batch_generator

            # Reinitialize model weights each training fold
            model_with_encoder.set_weights(model_initial_weights)
            model_with_encoder.fit_generator(generator=training_batch_generator,
                                steps_per_epoch=nb_train_samples,
                                epochs=config['model']['n_epochs'],
                                validation_data=validation_data,
                                validation_steps=nb_validation_samples,
                                pickle_safe=False,
                                callbacks=train_manager.get_callbacks(
                                                config['model']['newmodel_file'],
                                                initial_learning_rate=config['model']['initial_learning_rate'],
                                                learning_rate_drop=config['model']['learning_rate_drop'],
                                                learning_rate_epochs=config['model']['decay_learning_rate_every_x_epochs'],
                                                logging_dir=config['data']['logging_dir'],
                                                ifold=ifold))

            if train_encoder_with_unet:
                print("Saving the trained encoder as", encoder_file, "...")
                model.save(encoder_decoder_file)
                encoder.save(encoder_file)

        hdf5_file_opened.close()

    if test_unet:
        # Test the trained model with test data loaded by data_loader, and write to .csv file
        # the dice scores as defined in vnet_builder.
        print("Loading test data from ", config['data']['hdf5_test'], " ...")
        with_gt = True
        data_loader = DataLoader(config['data'])  # empty data_loader
        data_loader.load_test_data(with_gt=with_gt)

        print("Testing model...")
        test_pred, test_gt, test_dice = test_manager.test_model(data_loader, model_without_encoder, vnet_builder.dice_score,
                                                                config['data']['n_labels'], with_gt)
        print(" ")
        print(n_labels)
        print(config['data']['label_names'][:n_labels])
        print(config['data']['label_reassign'][:n_labels+1])
        print("{:<25} {:<25}".format('Patient', '[Dice scores per label]'))
        with open('test_dice_scores.csv', 'w', newline='') as csv_file:
            csv_writer = csv.writer(csv_file)
            csv_writer.writerow(["n_labels: ", n_labels])
            csv_writer.writerow(config['data']['label_names'][:n_labels])
            csv_writer.writerow(config['data']['label_reassign'][:n_labels+1])
            for pat, dice in test_dice.items():
                print(pat, dice)
                csv_writer.writerow([pat] + dice)

        if visualize:
            print("Showing some predicted test slices...")
            show_test_slices(config['data']['n_labels'], config['data']['figures_dir'] + "/img_pred",
                             with_gt, data_loader, config['data']['padding'], test_pred)

        write_sitk = True
        if write_sitk:
            print("Writing predicted test volumes to ITK...")
            # Write predictions to .mhd/.raw files that can be inspected in ITK-SNAP,
            # both in the original resolution and in the Vnet input resolution.
            write_predictions_to_sitk(config['data'], data_loader, test_pred)


if __name__ == "__main__":
    main()

