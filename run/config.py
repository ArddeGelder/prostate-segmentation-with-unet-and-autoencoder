import os

config = dict()
config['crun'] = dict()
config['data'] = dict()
config['model'] = dict()
config['encoder'] = dict()

# Run configurations
#
config['crun']['create_hdf5'] = True
config['crun']['build_new_encoder'] = True
config['crun']['train_encoder'] = True
config['crun']['test_encoder'] = True
config['crun']['build_new_unet'] = True
config['crun']['train_unet'] = True
config['crun']['train_encoder_with_unet'] = False
config['crun']['cross_validation'] = False
config['crun']['test_unet'] = True
config['crun']['visualize'] = False

# Data configurations
#
# Input data folder locations
#
config['data']['data_dir'] = os.path.abspath('../../priorinput_data_2labels')
config['data']['seg_dir'] = os.path.abspath('../../priorinput_data_2labels')
config['data']['train_dir'] = os.path.abspath(config['data']['data_dir'] + '/train')
config['data']['train_seg_dir'] = os.path.abspath(config['data']['seg_dir'] + '/train')
config['data']['test_dir'] = os.path.abspath(config['data']['data_dir'] + '/test')
config['data']['test_seg_dir'] = os.path.abspath(config['data']['seg_dir'] + '/test')
#
# Output data folder locations
#
config['data']['figures_dir'] = os.path.abspath('./figures')
config['data']['hdf5data_dir'] = os.path.abspath('./hdf5_data')
config['data']['newmodel_dir'] = os.path.abspath('./new_model')
config['data']['pretrained_dir'] = os.path.abspath('../../pretrained_models')
config['data']['logging_dir'] = os.path.abspath('./tensorboard_logs')
config['data']['pickle_dir'] = os.path.abspath('./pickle_files')
config['data']['mhd_dir'] = os.path.abspath('./mhd_predictions')
config['data']['hdf5_train'] = config['data']['hdf5data_dir'] + "/train.hdf5"
config['data']['hdf5_test'] = config['data']['hdf5data_dir'] + "/test.hdf5"
config['data']['encoder_dir'] = os.path.abspath('./encoder')
#
# Grid resolution and dimensions
#
# Resampling grid resolution [mm, mm, mm]
config['data']['grid_resolution'] = [1, 1, 3.6]   # [1, 1, 1.5]
# Resampling grid size
config['data']['output_shape'] = [92, 92, 34] #[36, 36, 18]
# Padding of input arrays before passing on to Unet
config['data']['padding'] = [0, 0, 0]
# Unet input size
config['data']['input_shape'] = tuple([x[0] + 2 * x[1] for x in
                                       zip(config['data']['output_shape'], config['data']['padding'])])
#
# Parameters for resampling by SimpleITK
#
config['data']['resample'] = True
config['data']['crop'] = True
config['data']['normalize'] = True
# Parameters for future use of more modalities
config['data']['modalities'] = ['T2']  # ['T1', 'T1c', 'Flair', 'T2']
config['data']['nb_channels'] = len(config['data']['modalities'])
#
# Data labels
#
# ------> NUMBER OF LABELS FROM THE LIST BELOW INCLUDED IN THE SEGMENTATION, NOT INCLUDING THE BACKGROUND
config['data']['n_labels'] = 2
# Label names
# 1=TZ: prostate transition zone
# 2=PZ: prostate peripheral zone
# 3=OT: obturator_internus
# 4=BL: bladder
# 5=FB: femur bone
# 6=RE: rectum
# 7=LA: levator ani
# 8=PB: pelvic bone
# 9=FA: fat
config['data']['label_names'] = ['TZ', 'PZ', 'OT', 'BL', 'FB', 'RE', 'LA', 'PB', 'FA']
# Reassignment of labels if swapped position or turned to background (=0)
config['data']['label_reassign'] = [0, 1, 2, 0, 0, 0, 0, 0, 0, 0]
#
# Augmentation parameters, training data will be distorted on the fly so as to avoid over-fitting.
#
# Data will be randomly rotated in the xy plane, by a maximum angle rotate_max_angle
config['data']['augment_rotate'] = True
config['data']['augment_prob_rotate'] = 0.3
config['data']['augment_rotate_max_angle'] = 10
# Data will be randomly shifted along the x, y and z axis, by a maximum distance augment_shift_max_mm (in mm)
config['data']['augment_shift'] = True
config['data']['augment_prob_shift'] = 0.3
config['data']['augment_shift_max_mm'] = 10
# Data will be flipped in the x-dimension
config['data']['augment_flip'] = True
config['data']['augment_prob_flip'] = 0.1
# Data will be randomly zoomed in or out, by a factor between 1 - dzoom_max and 1 + dzoom_max
config['data']['augment_zoom'] = True
config['data']['augment_prob_zoom'] = 0.3
config['data']['augment_dzoom_max'] = 0.1
# Data will be randomly deformed by elastic deformation, with parameters
# - alpha_m: length scale of deformation in mm
# - sigma: standard deviation for Gaussian kernel
# Note: larger sigma means smaller deformations!
config['data']['augment_elastic_deformation'] = True
config['data']['augment_prob_deform'] = 0.3
config['data']['augment_alpha_mm'] = 35
config['data']['augment_sigma'] = 6
# The fraction of images that will NOT be augmented is (1 - 0.3)^4 * 0.9 = 0.2
config['data']['augment'] = config['data']['augment_rotate'] or config['data']['augment_shift'] or \
                            config['data']['augment_flip'] or config['data']['augment_zoom'] or \
                            config['data']['augment_elastic_deformation']
							#
# Training run hyper-parameters
#
config['model']['batch_size'] = 1
config['model']['initial_learning_rate'] = 0.0001
config['model']['decay_learning_rate_every_x_epochs'] = 100
config['model']['learning_rate_drop'] = 0.1
config['model']['cross_validation_kfold'] = 5
config['model']['kfold_ids_file'] = config['data']['pickle_dir'] + "/kfold_ids.pkl"
#
# Unet model parameters
#
config['model']['n_labels'] = config['data']['n_labels']
config['model']['padding'] = [44, 44, 22]
config['model']['pool_size'] = (2, 2, 2)
# Parameters for future use of more modalities
config['model']['modalities'] = config['data']['modalities']
config['model']['nb_channels'] = config['data']['nb_channels']
# Channels last for Tensorflow
config['model']['input_shape'] = tuple(list(config['data']['input_shape']) + [config['model']['nb_channels']])
# divide the number of filters used by by a given factor. This will reduce memory consumption.
config['model']['downsize_nb_filters_factor'] = 1
# weight factors for [background, 'TZ', 'PZ', 'OT', 'BL', 'FB', 'RE', 'LA', 'PB', 'FA']
config['model']['loss_weight_factors'] = [1., 2., 6., 1., 1., 1., 1., 1., 1., 1.]
#config['model']['loss_weight_factors'] = [1., 2., 6.]
config['model']['newmodel_file'] = config['data']['newmodel_dir'] + "/3d_unet_model_with_encoder.h5"
config['model']['pretrained_model'] = config['model']['newmodel_file']
config['model']['deconvolution'] = False  # use deconvolution instead of up-sampling. Requires keras-contrib.

config['model']['n_epochs'] = 300

# Encoder hyperparameters
config['encoder']['n_epochs'] = 100
config['encoder']['name'] = 'encoder_conv_basic'
config['encoder']['file'] = config['data']['encoder_dir'] + "/" + config['encoder']['name'] + ".h5"
config['encoder']['file_with_decoder'] = config['data']['encoder_dir'] + "/" + config['encoder']['name'] + "_with_decoder.h5"
config['data']['encoder_file'] = config['encoder']['file']

# Encoder parameters
config['encoder']['max_filters'] = 256