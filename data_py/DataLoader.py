from os import listdir
from os.path import isfile, join, splitext
import numpy as np
import SimpleITK as sitk


class DataLoader(object):
    """
    Loads data volumes in ITK format, pre-processes data (resampling, cropping, normalization), converts
    each data volume to a numpy array and organises data volumes in a list of numpy arrays.
    :return: The following lists are accessed by the module Hdf5Writer:
    data_loader.img_numpy: list of training volume numpy arrays
    data_loader.gt_numpy: list of ground truth volume numpy arrays
    data_loader.img_list: list of names of training volume files
    data_loader.gt_list: list of names of ground truth volume files
    """

    def __init__(self, config_data):
        """
        :param config_data: data configuration parameters.
        """

        self.config_data = config_data
        self.input_shape = config_data['input_shape']
        self.output_shape = config_data['output_shape']
        self.padding = config_data['padding']
        self.orig_shape = [None] * 3
        self.train_dir = config_data['train_dir']
        self.train_seg_dir = config_data['train_seg_dir']
        self.test_dir = config_data['test_dir']
        self.test_seg_dir = config_data['test_seg_dir']

        self.img_list = list()
        self.gt_list = list()
        self.img_sitk = dict()
        self.img_crop = dict()
        self.gt_sitk = dict()
        self.img_numpy = None
        self.gt_numpy = None

        self.new_size = None
        self.mean_intensity_train = None

    def load_train_data(self):
        self.create_images_list(self.train_dir, self.train_seg_dir)
        self.load_images(self.train_dir)
        self.img_numpy = self.get_data_numpy(self.img_sitk, self.img_list, self.input_shape,
                                             resample=True, crop=True, normalize=True, dtype=np.float32)
        self.create_ground_truths_list()
        self.load_ground_truths(self.train_seg_dir)
        self.gt_numpy = self.get_data_numpy(self.gt_sitk, self.gt_list, self.output_shape,
                                            resample=True, crop=True, normalize=False, dtype=np.uint8)
        self.gt_numpy = self.discrete_gt_label(self.gt_numpy, self.gt_list)
        # self.gt_volume_ratio_avg, self.gt_volume_ratios = self.calc_gt_stats(self.gt_numpy, self.gt_list)

    def load_test_data(self, with_gt=False):
        self.create_images_list(self.test_dir, self.test_seg_dir)
        self.load_images(self.test_dir)
        self.img_numpy = self.get_data_numpy(self.img_sitk, self.img_list, self.input_shape,
                                             resample=True, crop=True, normalize=True, dtype=np.float32)
        if with_gt:
            self.create_ground_truths_list()
            self.load_ground_truths(self.test_seg_dir)
            self.gt_numpy = self.get_data_numpy(self.gt_sitk, self.gt_list, self.output_shape,
                                                resample=True, crop=True, normalize=False, dtype=np.uint8)
            self.gt_numpy = self.discrete_gt_label(self.gt_numpy, self.gt_list)
            # self.gt_volume_ratio_avg, self.gt_volume_ratios = self.calc_gt_stats(self.gt_numpy, self.gt_list)

    def create_images_list(self, data_dir, seg_dir):
        # Get list of annotated volumes
        seg_list = [f[:7] for f in listdir(seg_dir)]
        # Get list of volumes for which annotations are available
        self.img_list = [f for f in listdir(data_dir) if isfile(join(data_dir, f)) and 'segmentation' not in f
                         and 'raw' not in f and 'itksnap' not in f and f[:7] in seg_list]
        print('IMG FILE LIST: ' + str(self.img_list) + '\n')

    def create_ground_truths_list(self):
        for f in self.img_list:
            filename, ext = splitext(f)
            # self.gt_list.append(join(filename + '_segmentation' + ext))
            self.gt_list.append(filename[:7] + '_segmentation.nii.gz')
        print('GT FILE LIST: ' + str(self.gt_list) + '\n')

    def load_images(self, data_dir):
        """
        Loads images in ITK format from disk and performs the following pre-processing steps:
        rescale intensity: rescale image intensities to a 0-1 range.
        :param data_dir: data directory with data volumes in ITK format.
        :return:
        """
        rescale_filter = sitk.RescaleIntensityImageFilter()
        rescale_filter.SetOutputMaximum(1)
        rescale_filter.SetOutputMinimum(0)
        stats = sitk.StatisticsImageFilter()
        m = 0.
        for f in self.img_list:

            sitk_image = sitk.Cast(sitk.ReadImage(join(data_dir, f)), sitk.sitkFloat32)
            # dims = (sitk.Image.GetWidth(sitk_image), sitk.Image.GetHeight(sitk_image), sitk.Image.GetDepth(sitk_image))
            # spacing = sitk.Image.GetSpacing(sitk_image)
            # print("Image ", f, " has dimensions ", dims, " and spacings ", spacing)
            self.img_sitk[f] = rescale_filter.Execute(sitk_image)
            stats.Execute(self.img_sitk[f])
            m += stats.GetMean()
        self.mean_intensity_train = m/len(self.img_sitk)

    def load_ground_truths(self, data_dir):
        for f in self.gt_list:
            # print("Ground truth name in load_ground_truths ", f)
            # self.gt_sitk[f] = sitk.Cast(sitk.ReadImage(join(data_dir, f)) > 0.5, sitk.sitkFloat32)
            self.gt_sitk[f] = sitk.Cast(sitk.ReadImage(join(data_dir, f)), sitk.sitkUInt8)

    def get_data_numpy(self, data_sitk, data_list, shape, resample=False, crop=False, normalize=False, dtype=np.float32):
        """
        Converts sitk data into numpy arrays, and performs the following pre-processing steps:
        smooth: STILL TO BE IMPLEMENTED smoothing of data before resampling to avoid resampling artifacts
        :param data_sitk: list of data volumes in ITK format
        :param resample: resample images and ground truths to resolution config_data['grid_resolution']
        :param crop: crop images and ground truths to dimensions config_data['output_shape'], around the center of the originals
        :param normalize: normalize image numpy arrays to mean = 0 and a standard deviation = 1
        :param dtype: dtype of numpy array
        :return: list of data volumes as numpy arrays
        """
        data_numpy = dict()
        for f in data_list:
            # print("f in get_data_numpy ", f)
            data_numpy[f] = np.zeros(shape, dtype=dtype)
            data = data_sitk[f]
            size = data.GetSize()

            if resample:
                if dtype==np.uint8:
                    method = sitk.sitkNearestNeighbor
                else:
                    method = sitk.sitkLinear
                data = self.resample(data, shape, size, method=method)
            if crop:
                data = self.crop(data, shape, size)
            self.img_crop[f] = data

            data_numpy[f] = np.transpose(sitk.GetArrayFromImage(data).astype(dtype=dtype),
                                         [2, 1, 0])[:, :, :, np.newaxis]
            if normalize:
                data_numpy[f] = self.normalize(data_numpy[f])

        return data_numpy

    def discrete_gt_label(self, gt_numpy, gt_list):

        for f in gt_list:
            # print("f in discrete_gt_label ", f)
            gt = gt_numpy[f]
            gt_numpy[f][np.where(gt > self.config_data['n_labels'])] = 0
            for label in range(self.config_data['n_labels'] + 1):
                gt_numpy[f][np.where(gt == label)] = self.config_data['label_reassign'][label]
            gt_numpy[f] = gt_numpy[f].astype(np.uint8)

        return gt_numpy

    def calc_gt_stats(self, gt_numpy, gt_list):

        gt_volume_ratios = dict()
        for f in gt_list:
            gt = gt_numpy[f]
            unique, counts = np.unique(gt, return_counts=True)
            gt_volume_ratios[f] = counts / np.sum(counts)
            # print("Image ", f, " has voxel counts ", counts, " and ratios ", gt_volume_ratios[f])
        gt_volume_ratio_avg = sum(gt_volume_ratios.values()) / len(gt_volume_ratios)

        return gt_volume_ratio_avg, gt_volume_ratios

    def resample(self, data, shape, size, method=sitk.sitkLinear):
        # we rotate the image according to its transformation using the direction and
        # according to the final spacing we want
        # factor = np.asarray(data.GetSpacing()) / self.config_data['grid_resolution']
        spacing = list(data.GetSpacing())
        new_spacing = self.config_data['grid_resolution']
        factor = [None] * 3
        self.new_size = [None] * 3
        for d in range(3):
            factor[d] = spacing[d] / self.config_data['grid_resolution'][d]
            self.new_size[d] = max(int(size[d] * factor[d]), shape[d])
        T = sitk.AffineTransform(3)
        T.SetMatrix(data.GetDirection())
        resampler = sitk.ResampleImageFilter()
        resampler.SetReferenceImage(data)
        resampler.SetOutputSpacing(new_spacing)
        resampler.SetSize(self.new_size)
        resampler.SetInterpolator(method)

        data_resampled = resampler.Execute(data)

        return data_resampled

    def crop(self, data, shape, size):
        data_centroid = [None] * 3
        data_start_pxl = [None] * 3
        for d in range(3):
            data_centroid[d] = int(self.new_size[d] / 2)
            data_start_pxl[d] = int(data_centroid[d] - shape[d] / 2)
        region_extractor = sitk.RegionOfInterestImageFilter()
        region_extractor.SetSize(shape)
        region_extractor.SetIndex(data_start_pxl)

        data_cropped = region_extractor.Execute(data)

        return data_cropped

    def normalize(self, data):
        mean = np.mean(data, axis=(0, 1, 2))
        std = np.std(data, axis=(0, 1, 2))
        data_norm = (data - mean[:, np.newaxis, np.newaxis, np.newaxis])/std[:, np.newaxis, np.newaxis, np.newaxis]

        return data_norm
