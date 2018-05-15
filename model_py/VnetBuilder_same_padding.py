import sys
import numpy as np
from keras import backend as K
from keras.engine import Input, Model
from keras import layers
from keras.layers import ZeroPadding3D, Conv3D, MaxPooling3D, UpSampling3D, Cropping3D, Activation, Dropout, Dense
from keras.optimizers import Adam
from keras import regularizers
import tensorflow as tf
#K.set_image_dim_ordering('tf')
from keras.models import load_model


try:
    from keras.engine import merge
except ImportError:
    from keras.layers.merge import concatenate


class VnetBuilder(object):
    """
    Builds a VNet Keras model with build_model method.
    """

    def __init__(self, config_data, wt_fac=None, train_encoder_with_unet=False):
        self.n_labels = config_data['n_labels']
        self.wt_fac = wt_fac[:self.n_labels+1]
        self.encoder = load_model(config_data['encoder_file'])
        self.encoder.trainable = train_encoder_with_unet
        print("Succesfully loaded encoder from", config_data['encoder_file'])

    def build_model(self, input_shape, padding=(44, 44, 22), downsize_filters_factor=1, pool_size=(2, 2, 2),
                    initial_learning_rate=0.0001):
        """
        :param input_shape: Shape of the input data (x_size, y_size, z_size, n_channels).
        :param padding: padding applied to images
        :param downsize_filters_factor: Factor to which to reduce the number of filters.
        Making this value larger will reduce the amount of memory the model will need during training.
        :param pool_size: Pool size for the max pooling operations.
        :param n_labels: Number of binary labels that the model is learning.
        :param initial_learning_rate: Initial learning rate for the model. This will be decayed during training.
        :param deconvolution: If set to True, will use transpose convolution(deconvolution) instead of upsampling.
        This increases the amount memory required during training.
        :return: Untrained 3D UNet Model
        """

        """
        It does a matrix multiply, bias add, and then uses relu to nonlinearize.
        It also sets up name scoping so that the resultant graph is easy to read,
        and adds a number of summary ops.
        """

        padding_style = 'same'

        #Fix initial padding
        padding = tuple()
        for i in range(3):
            pool_max = pool_size[i] ** 3
            pad = (pool_max - (input_shape[i] % pool_max)) // 2
            padding += (pad,)

        # Image input
        inputs = Input(input_shape)

        # Downsampling
        conv1p = ZeroPadding3D(padding=padding, name='conv1p')(inputs)
        conv1a = Conv3D(int(32 / downsize_filters_factor), (3, 3, 3), activation='relu', padding=padding_style,
                        kernel_regularizer=regularizers.l2(0), kernel_initializer='glorot_uniform',
                        name='conv1a')(conv1p)
        conv1b = Conv3D(int(64 / downsize_filters_factor), (3, 3, 3), activation='relu', padding=padding_style,
                        kernel_regularizer=regularizers.l2(0), kernel_initializer='glorot_uniform',
                        name='conv1b')(conv1a)
        pool1 = MaxPooling3D(pool_size=pool_size, name='pool1')(conv1b)
        conv2a = Conv3D(int(64 / downsize_filters_factor), (3, 3, 3), activation='relu', padding=padding_style,
                        kernel_regularizer=regularizers.l2(0), kernel_initializer='glorot_uniform',
                        name='conv2a')(pool1)
        conv2b = Conv3D(int(128 / downsize_filters_factor), (3, 3, 3), activation='relu', padding=padding_style,
                        kernel_regularizer=regularizers.l2(0), kernel_initializer='glorot_uniform',
                        name='conv2b')(conv2a)
        pool2 = MaxPooling3D(pool_size=pool_size, name='pool2')(conv2b)
        conv3a = Conv3D(int(128 / downsize_filters_factor), (3, 3, 3), activation='relu', padding=padding_style,
                        kernel_regularizer=regularizers.l2(0), kernel_initializer='glorot_uniform',
                        name='conv3a')(pool2)
        conv3b = Conv3D(int(256 / downsize_filters_factor), (3, 3, 3), activation='relu', padding=padding_style,
                        kernel_regularizer=regularizers.l2(0), kernel_initializer='glorot_uniform',
                        name='conv3b')(conv3a)
        pool3 = MaxPooling3D(pool_size=pool_size, name='pool3')(conv3b)
        conv4a = Conv3D(int(256 / downsize_filters_factor), (3, 3, 3), activation='relu', padding=padding_style,
                        kernel_regularizer=regularizers.l2(0), kernel_initializer='glorot_uniform',
                        name='conv4a')(pool3)
        conv4b = Conv3D(int(512 / downsize_filters_factor), (3, 3, 3), activation='relu', padding=padding_style,
                        kernel_regularizer=regularizers.l2(0), kernel_initializer='glorot_uniform',
                        name='conv4b')(conv4a)

        # upsampling
        up5a = UpSampling3D(size=pool_size)(conv4b)
        up5b = self.bridge(conv3b, up5a)
        conv5a = Conv3D(int(256 / downsize_filters_factor), (3, 3, 3), activation='relu', padding=padding_style,
                        kernel_regularizer=regularizers.l2(0), kernel_initializer='glorot_uniform',
                        name='conv5a')(up5b)
        conv5b = Conv3D(int(256 / downsize_filters_factor), (3, 3, 3), activation='relu', padding=padding_style,
                        kernel_regularizer=regularizers.l2(0), kernel_initializer='glorot_uniform',
                        name='conv5b')(conv5a)
        up6a = UpSampling3D(size=pool_size, name='up6a')(conv5b)
        up6b = self.bridge(conv2b, up6a)
        conv6a = Conv3D(int(128 / downsize_filters_factor), (3, 3, 3), activation='relu', padding=padding_style,
                        kernel_regularizer=regularizers.l2(0), kernel_initializer='glorot_uniform',
                        name='conv6a')(up6b)
        conv6b = Conv3D(int(128 / downsize_filters_factor), (3, 3, 3), activation='relu', padding=padding_style,
                        kernel_regularizer=regularizers.l2(0), kernel_initializer='glorot_uniform',
                        name='conv6b')(conv6a)
        up7a = UpSampling3D(size=pool_size, name='up7a')(conv6b)
        up7b = self.bridge(conv1b, up7a)
        conv7a = Conv3D(int(64 / downsize_filters_factor), (3, 3, 3), activation='relu', padding=padding_style,
                        kernel_regularizer=regularizers.l2(0), kernel_initializer='glorot_uniform',
                        name='conv7a')(up7b)
        conv7b = Conv3D(int(64 / downsize_filters_factor), (3, 3, 3), activation='relu', padding=padding_style,
                        kernel_regularizer=regularizers.l2(0), kernel_initializer='glorot_uniform',
                        name='conv7b')(conv7a)

        # Final softmax layer for class probabilities for each input voxel
        conv7d = Dense(int(64 / downsize_filters_factor))(conv7b)
        conv8 = Conv3D(self.n_labels + 1, (1, 1, 1), name='conv8')(conv7d)  # include background label
        act = Activation('softmax', name='seg')(conv8)

        model_without_encoder = Model(inputs=inputs, outputs=act)
        model_without_encoder.summary()

        aux_output = self.encoder(act)
        encoded_output = Activation('linear', name='seg_encoded')(aux_output)

        # Model definition
        model_with_encoder = Model(inputs=inputs, outputs=[act, encoded_output])
        model_without_encoder = Model(inputs=inputs, outputs=act)
        model_with_encoder.summary()

        # Metrics
        metrics = []
        for label in range(self.n_labels):
            dice_coef = self.create_dice_coef(label)
            dice_coef.__name__ = 'dice_coef_lbl' + str(label+1)
            metrics.append(dice_coef)

        # Add optimizer and loss function
            model_with_encoder.compile(optimizer=Adam(lr=initial_learning_rate),
                      loss={'seg': self.weighted_categorical_crossentropy,
                            'seg_encoded': self.weighted_categorical_crossentropy},
                      loss_weights={'seg': 1.,
                                    'seg_encoded': 1000.},
                      metrics=metrics)

        return model_with_encoder, model_without_encoder

    def bridge(self, layer1, layer2):
        # Defines bridge between layers of the same resolution from the downsampling and form the upsampling paths.
        crop = np.ceil(np.subtract(layer1.shape.as_list()[1:4], layer2.shape.as_list()[1:4]) / 2).astype(int)
        if all(layer2.shape.as_list()[1:4] + 2 * crop == layer1.shape.as_list()[1:4]):
            cropping = tuple(map(tuple, np.transpose([crop, crop])))
        else:
            sys.exit("Cropping at " + layer1.name + "-" + layer2.name + "bridge is not symmetric, "
                     "fix image and input dimensions to fit Unet structure.")
        layer2 = concatenate([layer2, Cropping3D(cropping=cropping)(layer1)], axis=4)

        return layer2

    def combined_loss(self, y_true, y_pred):
        cat_loss = self.weighted_categorical_crossentropy(y_true, y_pred)
        encoder_loss = self.encoder_loss(y_true, y_pred)
        return cat_loss * encoder_loss

    def weighted_categorical_crossentropy(self, y_true, y_pred):
        # Weighted categorical cross entropy is used as loss function, because the classes are imbalanced.
        # This function is not available for 3D data, in Keras, and is defined here based on the code of the
        # Keras function 'categorical_crossentropy'.
        _EPSILON = 10e-8
        # Scale predictions so that the class probabilities of each sample sum to 1
        y_pred /= tf.reduce_sum(y_pred,
                                axis=len(y_pred.get_shape()) - 1,
                                keep_dims=True)
        # Ensuring probabilities <0, 1>
        epsilon = tf.cast(_EPSILON, y_pred.dtype.base_dtype)
        y_pred = tf.clip_by_value(y_pred, epsilon, 1. - epsilon)
        # Weights used in loss function
        wt_fac = self.wt_fac
        wt_map = y_true * wt_fac
        loss_map = wt_map * y_true * tf.log(y_pred)
        # Manual computation of crossentropy
        loss = - tf.reduce_sum(loss_map, axis=len(y_pred.get_shape()) - 1)
        return loss

    def create_dice_coef(self, label):
        # Create dice coeffients for label=label.
        def dice_coef(y_true, y_pred, smooth=1.):
            y_true_f = K.flatten(y_true[:, :, :, :, -label])  # exclude background label
            y_pred_f = K.flatten(y_pred[:, :, :, :, -label])
            intersection = K.sum(y_true_f * y_pred_f)
            dice = (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)

            return dice

        return dice_coef

    def dice_score(self, y_true, y_pred, smooth=1.):  # , wt_fac=(2, 4)):
        # Dice score for use in non-GPU python code.
        dice = []
        for label in range(self.n_labels):
            y_true_f = y_true[:, :, :, :, label + 1].flatten()  # exclude background label
            y_pred_f = y_pred[:, :, :, :, label + 1].flatten()
            intersection = np.sum(y_true_f * y_pred_f)
            dice.append((2. * intersection + smooth) / (np.sum(y_true_f) + np.sum(y_pred_f) + smooth))
        return dice

    def encoder_loss(self, y_true, y_pred):
        print(y_true)
        y_true_encoded = self.encoder.predict_on_batch(x=y_true)
        y_pred_encoded = self.encoder.predict_on_batch(x=y_pred)
        dif = y_true_encoded - y_pred_encoded
        loss = dif ** 2
        return loss

    def generalized_dice_loss(self, y_true, y_pred):
        #UNFINISHED
        _EPSILON = 10e-8

        #Scale predictions so that probabilities of each sample sum to 1
        y_pred /= tf.reduce_sum(y_pred, axis=len(y_pred.get_shape())-1, keep_dims=True)
        epsilon = tf.cast(_EPSILON, y_pred.dtype.base_dtype)
        y_pred = tf.clip_by_value(y_pred, epsilon, 1. - epsilon)
        numerator = 0
        denominator = 0
        for label in range(self.n_labels):
            y_pred_flat = K.flatten(y_pred[:, :, :, :, label]) #to flatten and exclude background label
            y_true_flat = K.flatten(y_true[:, :, :, :, label])
            weight = K.sum(y_true_flat) ** -2
            intersection = K.sum(y_true_flat * y_pred_flat)
            numerator += 2*weight*intersection
            denominator += weight*(K.sum(y_true_flat) + K.sum(y_pred_flat))

        gdl = 1 - numerator / denominator

        return gdl



    # def weighted_dice_coef(self, y_true, y_pred, smooth=1.):
    #     # Weighted dice score for use in non-GPU python code.
    #     # weights
    #     wt_fac = self.wt_fac
    #     wt_map = y_true * wt_fac
    #     y_true_f = K.flatten(wt_map[:, :, :, :, 1:] * y_true[:, :, :, :, 1:])  # exclude background label
    #     y_pred_f = K.flatten(y_pred[:, :, :, :, 1:])  # exclude background label, only label=1 of interest
    #     intersection = K.sum(y_true_f * y_pred_f)
    #     return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)
