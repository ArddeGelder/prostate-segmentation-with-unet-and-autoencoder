import numpy as np
from numpy.random import random
from scipy.ndimage.interpolation import map_coordinates, rotate, shift, zoom
from scipy.ndimage.filters import gaussian_filter


def augment_data(img, gt, config_data):
    """
    Training data will be distorted on the fly so as to avoid over-fitting.
    :param img: image data array
    :param gt: ground truth array
    :param config_data: configuration parameters for augmentation
    :return: distorted img and gt
    """
    grid_resolution = config_data['grid_resolution']
    # Data will be randomly rotated in the xy plane, by a maximum angle rotate_max_angle
    if config_data['augment_rotate']:
        if random() < config_data['augment_prob_rotate']:
            img, gt = get_random_rotation(img, gt, rotate_max_angle=config_data['augment_rotate_max_angle'])
    # Data will be randomly shifted along the x, y and z axis, by a maximum distance augment_shift_max_mm (in mm)
    if config_data['augment_shift']:
        if random() < config_data['augment_prob_shift']:
            img, gt = get_random_shift(img, gt, grid_resolution, shift_max_mm=config_data['augment_shift_max_mm'])
    # Data will be flipped in the x-dimension
    if config_data['augment_flip']:
        if random() < config_data['augment_prob_flip']:
            img = img[::-1, :, :, :]
            gt = gt[::-1, :, :, :]
    # Data will be randomly zoomed in or out, by a factor between 1 - dzoom_max and 1 + dzoom_max
    if config_data['augment_zoom']:
        if random() < config_data['augment_prob_zoom']:
            img, gt = get_random_zoom(img, gt, dzoom_max=config_data['augment_dzoom_max'])
    # Data will be randomly deformed by elastic deformation
    if config_data['augment_elastic_deformation']:
        if random() < config_data['augment_prob_deform']:
            img, gt = get_random_elastic(img, gt, grid_resolution, alpha_mm=config_data['augment_alpha_mm'],
                                         sigma=config_data['augment_sigma'], random_state=None)

    return img, gt


def get_random_rotation(img, gt, rotate_max_angle=10):
    # Rotation will be by default in the xy-plane
    angle_rnd = (random() * 2 - 1) * rotate_max_angle
    img_rotate = rotate(img, angle_rnd, reshape=False)
    gt_rotate = rotate(gt, angle_rnd, reshape=False, order=0)

    return img_rotate, gt_rotate


def get_random_shift(img, gt, grid_resolution, shift_max_mm=10):
    # The image array has 4 dimensions: x, y, z, and channel
    shift_rnd = [(random() * 2 - 1) * shift_max_mm/res for res in grid_resolution] + [0]
    img_shift = shift(img, shift_rnd)  # default order=3
    gt_shift = shift(gt, shift_rnd, order=0)

    return img_shift, gt_shift


def get_random_zoom(img, gt, dzoom_max=0.1):
    # The image array has 4 dimensions: x, y, z, and channel
    zoom_rnd = 1 + (random() * 2 - 1) * dzoom_max

    # save the current shape of X for reshaping later
    shape = img.shape

    # reshape the data
    x, y, z = np.meshgrid(np.arange(shape[0]), np.arange(shape[1]), np.arange(shape[2]), indexing='ij')
    dx = - (x - np.average(x)) * (zoom_rnd - 1)
    dy = - (y - np.average(y)) * (zoom_rnd - 1)
    dz = - (z - np.average(z)) * (zoom_rnd - 1)
    indices = (np.reshape(x + dx, (-1, 1)), np.reshape(y + dy, (-1, 1)), np.reshape(z + dz, (-1, 1)))

    img_zoom = map_coordinates(img[:, :, :, 0], indices).reshape(shape)
    gt_zoom = map_coordinates(gt[:, :, :, 0], indices, order=0).reshape(shape)

    return img_zoom, gt_zoom


def get_random_elastic(img, gt, grid_resolution, alpha_mm=35, sigma=6, random_state=None):
    """
    Elastic deformation of images as described in [Simard2003]
    Simard, Steinkraus and Platt,
    "Best Practices for Convolutional Neural Networks applied to Visual Document Analysis",
    in Proc. of the International Conference on Document Analysis and Recognition, 2003.
    :param img: image data array
    :param gt: ground truth array
    :param grid_resolution: voxel size in mm to convert lengths in mm into voxel sizes
    :param alpha_mm: length scale of deformation in mm
    :param sigma: standard deviation for Gaussian kernel. Note: larger sigma means smaller deformations!
    :return: deformed img and gt
    """
    if random_state is None:
        random_state = np.random.RandomState(None)

    # save the current shape of X for reshaping later
    shape = img.shape

    # determine reshaping coordinates
    dx = gaussian_filter((random_state.rand(*shape[0:3]) * 2 - 1), sigma, mode="constant", cval=0) * \
         alpha_mm/grid_resolution[0]
    dy = gaussian_filter((random_state.rand(*shape[0:3]) * 2 - 1), sigma, mode="constant", cval=0) * \
         alpha_mm/grid_resolution[1]
    dz = gaussian_filter((random_state.rand(*shape[0:3]) * 2 - 1), sigma, mode="constant", cval=0) * \
         alpha_mm/grid_resolution[2]

    # reshape the data
    x, y, z = np.meshgrid(np.arange(shape[0]), np.arange(shape[1]), np.arange(shape[2]), indexing='ij')
    indices = (np.reshape(x + dx, (-1, 1)), np.reshape(y + dy, (-1, 1)), np.reshape(z + dz, (-1, 1)))

    img_elastic = map_coordinates(img[:, :, :, 0], indices).reshape(shape)
    gt_elastic = map_coordinates(gt[:, :, :, 0], indices, order=0).reshape(shape)

    return img_elastic, gt_elastic




