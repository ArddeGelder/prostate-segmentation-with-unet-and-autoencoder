from keras.backend import shape
from keras.layers import Input, Conv3D, Flatten, UpSampling3D, MaxPooling3D, ZeroPadding3D, Conv3DTranspose, Cropping3D
from keras.engine import Model
from keras.optimizers import Adam
import tensorflow as tf
import numpy as np

class EncoderBuilder(object):

    def __init__(self, config_model):
        self.n_labels = config_model['n_labels']
        self.input_shape = config_model['input_shape'][:3] + (self.n_labels+1,)
        self.input_size = np.prod(self.input_shape)
        self.wt_fac = config_model['loss_weight_factors'][:self.n_labels+1]
        self.lr = config_model['initial_learning_rate']

    def build_convolution_model(self, max_filters=16):
        inputs = Input(self.input_shape) #36 36 18
        padding = (0, 0, 1)
        x = inputs
        x = ZeroPadding3D(padding=padding)(x) #36 36 20
        x = Conv3D(filters=max_filters // 4, kernel_size=(3, 3, 3), activation='relu', padding='same')(x)
        x = Conv3D(filters=max_filters // 4, kernel_size=(3, 3, 3), activation='relu', padding='same')(x)
        x = MaxPooling3D(padding='same')(x) #18 18 10
        x = Conv3D(filters=max_filters // 2, kernel_size=(3, 3, 3), activation='relu', padding='same')(x)
        x = Conv3D(filters=max_filters // 2, kernel_size=(3, 3, 3), activation='relu', padding='same')(x)
        x = MaxPooling3D(padding='same')(x) #9 9 5
        x = Conv3D(filters=max_filters, kernel_size=(3, 3, 3), activation='relu', padding='same')(x)
        x = Conv3D(filters=self.n_labels-1, kernel_size=(3, 3, 3), activation='relu', padding='same')(x)
        encoded = x
        x = Conv3D(filters=max_filters, kernel_size=(3, 3, 3), activation='relu', padding='same')(x)
        x = UpSampling3D()(x)
        x = Conv3D(filters=max_filters // 2, kernel_size=(3, 3, 3), activation='relu', padding='same')(x)
        x = Conv3D(filters=max_filters // 2, kernel_size=(3, 3, 3), activation='relu', padding='same')(x)
        x = UpSampling3D()(x)
        x = Conv3D(filters=max_filters // 4, kernel_size=(3, 3, 3), activation='relu', padding='same')(x)
        x = Conv3D(filters=max_filters // 4, kernel_size=(3, 3, 3), activation='relu', padding='same')(x)
        x = Conv3D(filters=self.n_labels+1, kernel_size=(1, 1, 1), activation='softmax', padding='same')(x)
        x = Cropping3D(cropping=(0, 0, 1))(x)
        decoded = x

        model = Model(inputs=inputs, outputs=decoded)
        encoder = Model(inputs=inputs, outputs=encoded)

        model.compile(optimizer=Adam(lr=self.lr),
                      loss='binary_crossentropy')
        model.summary()
        print("optimizer=Adam(lr=", self.lr, ", loss=binary_crossentropy")
        #print("Weight factors:", self.wt_fac)

        encoding_x = (self.input_shape[0]+2*padding[0]) // 4
        encoding_z = (self.input_shape[2]+2*padding[2]) // 4
        encoding_shape = (encoding_x, encoding_x, encoding_z, 1)
        return model, encoder, encoding_shape

    def weighted_categorical_crossentropy(self, y_true, y_pred):
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