import os
import numpy as np
from os.path import join, splitext
import matplotlib.pyplot as plt


def reverse_one_hot(gt_or_pred_batch):
    gt_or_pred_not_one_hot = np.expand_dims(np.argmax(gt_or_pred_batch, axis=4), axis=5)

    return gt_or_pred_not_one_hot


def show_loaded_slices(n_labels, file_png, padding, data_loader):
    i = 0
    n_img = 1
    for img_name in data_loader.img_list:
        #img_name = "Case25.mhd"
        filename, ext = splitext(img_name)
        gt_name = filename[:7] + '_segmentation.nii.gz'
        # print("Loaded image ", img_name, " and ground truth ", gt_name)
        img = data_loader.img_numpy[img_name]
        gt = data_loader.gt_numpy[gt_name]
        img_batch = np.expand_dims(img, axis=0)
        # img_batch = img_batch[:, padding[0]:-padding[0], padding[1]:-padding[1], padding[2]:-padding[2], :]
        gt_batch = np.expand_dims(gt, axis=0)
        show_slices(n_labels, file_png + str(i) + ".png", img_batch=(img_batch + 2) / 4, gt_batch=gt_batch, pred_batch=None)
        i += 1
        # Break because generator loops over the data set indefinitely.
        if i >= n_img:
            break


def show_hdf5_slices(n_labels, file_png, padding, hdf5_file):
    i = 0
    n_img = 1
    nb_samples = hdf5_file.root.img.shape[0]
    for index in range(nb_samples):
        # print("Loaded index ", index)
        img = hdf5_file.root.img[index]
        gt = hdf5_file.root.gt[index]
        img_batch = np.expand_dims(img, axis=0)
        # img_batch = img_batch[:, padding[0]:-padding[0], padding[1]:-padding[1], padding[2]:-padding[2], :]
        gt_batch = np.expand_dims(gt, axis=0)
        show_slices(n_labels, file_png + str(i) + ".png", img_batch=(img_batch + 2) / 4, gt_batch=gt_batch, pred_batch=None)
        i += 1
        # Break because generator loops over the data set indefinitely.
        if i >= n_img:
            break


def show_train_slices(n_labels, file_png, padding, training_batch_generator):
    i = 0
    n_batch = 3
    for img_batch, gt_batch in training_batch_generator:
        # img_batch = img_batch[:, padding[0]:-padding[0], padding[1]:-padding[1], padding[2]:-padding[2], :]
        show_slices(n_labels, file_png + str(i) + ".png", img_batch=(img_batch + 2) / 4, gt_batch=gt_batch, pred_batch=None)
        i += 1
        # Break to show only first n_img volumes.
        if i >= n_batch:
            break


def show_test_slices(n_labels, file_png, with_gt, data_loader, padding, test_pred):
    i = 0
    n_img = 1
    for test_name in data_loader.img_list:
        img = data_loader.img_numpy[test_name]
        img_batch = np.expand_dims(img, axis=0)
        # img_batch = img_batch[:, padding[0]:-padding[0], padding[1]:-padding[1], padding[2]:-padding[2], :]
        pred_batch = np.expand_dims(np.argmax(test_pred[test_name], axis=-1), axis=-1)
        if with_gt:
            gt_name = data_loader.gt_list[data_loader.img_list.index(test_name)]
            gt = data_loader.gt_numpy[gt_name]
            gt_batch = np.expand_dims(gt, axis=0)
            show_slices(n_labels, file_png + str(i) + ".png", img_batch=(img_batch + 2) / 4, gt_batch=gt_batch, pred_batch=pred_batch)
        else:
            show_slices(n_labels, file_png + str(i) + ".png", img_batch=(img_batch + 2) / 4, gt_batch=None, pred_batch=pred_batch)
        i += 1
        # Break to show only first n_img volumes.
        if i >= n_img:
            break


def show_slices(n_labels, file_png, img_batch=None, gt_batch=None, pred_batch=None):
    n_slice = 4
    n_row = 3
    n_img = len(img_batch)
    n_z = img_batch[0].shape[2]
    # z_list = list(int(n_z / 2 + 2 * (i_slice - n_slice/2)) for i_slice in range(n_slice))
    pos_gt = np.sum(gt_batch, (1, 2))
    #img_gt_batch = np.copy(img_batch)
    img_gt_batch = np.zeros(img_batch.shape)
    if gt_batch is not None:
        if gt_batch.shape[4] > 1:  # if one-hot representation revert to single label number in range [0:n_label]
            gt_batch = reverse_one_hot(gt_batch)
        for label in range(n_labels):
            img_gt_batch[np.where(gt_batch[:, :, :, :, 0] == label + 1)] = (label + 1) / n_labels
    img_pred_batch = np.copy(img_batch)
    if pred_batch is not None:
        if pred_batch.shape[4] > 1:  # if one-hot representation revert to single label number in range [0:n_label]
            pred_batch = reverse_one_hot(pred_batch)
        for label in range(n_labels):
            img_pred_batch[np.where(pred_batch[:, :, :, :, 0] == label + 1)] = (label + 1) / n_labels
    fig, axes = plt.subplots(n_row, n_slice)
    for i_slice in range(n_slice):
        i_img = np.random.randint(n_img)
        if np.max(pos_gt) > 0:
            z_img = np.random.choice(np.where(pos_gt[i_img] > np.max(pos_gt) / 2)[0])
            z_img = np.random.choice(np.where(pos_gt[i_img] > 0)[0])
        else:
            z_img = np.random.randint(n_z)
        # z_img = int(n_z / 2)
        # z_img = z_list[i_slice]
        if img_batch is not None:
            axes[0, i_slice].imshow(np.flipud(img_batch[i_img, :, :, z_img, 0].T),
                                    cmap="gray", origin="lower", clim=(0, 1))
        if gt_batch is not None:
            axes[1, i_slice].imshow(np.flipud(img_gt_batch[i_img, :, :, z_img, 0].T),
                                    cmap="gray", origin="lower", clim=(0, 1))
        else:
            axes[1, i_slice].axis('off')
        if pred_batch is not None:
            axes[2, i_slice].imshow(np.flipud(img_pred_batch[i_img, :, :, z_img, 0].T),
                                    cmap="gray", origin="lower", clim=(0, 1))
        else:
            axes[2, i_slice].axis('off')
    if os.path.exists(file_png):
        os.remove(file_png)
    fig.savefig(file_png)
    plt.close(fig)

