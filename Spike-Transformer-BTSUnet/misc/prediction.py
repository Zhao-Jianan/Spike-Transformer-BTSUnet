import os

import nibabel as nib
import numpy as np
import tables
from keras.layers import Input, LeakyReLU, Add, UpSampling3D, Activation, SpatialDropout3D, Conv3D
#from keras.layers import Flatten
from functools import partial
from .training import load_old_model
from .utils import pickle_load
#import tensorflow as tf
from keras.models import Model
from .patches import reconstruct_from_patches, get_patch_from_3d_data, compute_patch_indices, compute_patch_indices_for_prediction
import pdb
from tqdm import tqdm
import time
import random
from dev_tools.my_tools import print_red
from progressbar import *
activation_name='sigmoid'
from unet3d.model.unet import create_convolution_block, concatenate
create_convolution_block = partial(create_convolution_block, activation=LeakyReLU, instance_normalization=True)

import numpy as np
import nibabel as nib
from nilearn.image import new_img_like, resample_to_img
import random
import itertools
import pdb

import numpy as np
import nibabel as nib
from nilearn.image import new_img_like, resample_to_img
import random




import xlrd
import numpy as np
import pandas as pd
import os
import re
from progressbar import *
import nibabel as nib
import pdb
import scipy.ndimage
import matplotlib.pyplot as plt
import time
from tensorflow.python import pywrap_tensorflow
from tensorflow.python.tools import inspect_checkpoint as chkp


import os
import numpy as np
import nibabel as nib
import tables
from scipy import ndimage
from tqdm import tqdm
import time

import numpy as np
from scipy import ndimage
from scipy.ndimage import binary_closing, binary_opening, generate_binary_structure

import numpy as np
from scipy import ndimage
from scipy.ndimage.morphology import generate_binary_structure, binary_closing, binary_opening



def show_ckpt(filename):
    reader = pywrap_tensorflow.NewCheckpointReader(filename)
    var_to_shape_map = reader.get_variable_to_shape_map()
    #     print(var_to_shape_map)
    print_sep()
    for key in var_to_shape_map:
        print("tensor_name: ", key)

    #     print_sep()
    #     chkp.print_tensors_in_checkpoint_file(filename, tensor_name='', all_tensors=True)
    return


def print_red(something):
    print("\033[1;31m{}\033[0m".format(something))


def print2d(npy_img, trivial=False, img_name='', save=False, save_name='./test.jpg'):
    '''
    !!!dataset specific
    plot 2d mri images in Sagittal, Coronal and Axial dimension.
    img: 3d ndarray
    '''
    dim = npy_img.shape
    #     print('Dimension: ',npy_img.shape)
    f, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))

    img = npy_img[round(dim[0] / 2), :, :]
    #     img = npy_img[87,:,:]
    ax1.imshow(np.rot90(img), cmap=plt.cm.gray)
    #     ax1.imshow(np.rot90(img))
    if trivial:
        print('value of point(0,0) = ', img[0, 0])
    ax1.set_title('Sagittal ' + (img_name if trivial else ''), fontsize=15)
    #     ax1.imshow(img, cmap=plt.cm.gray)
    ax1.axis('off')
    img = npy_img[:, round(dim[1] / 2), :]
    #     img = npy_img[:,123,:]
    ax2.imshow(np.rot90(img), cmap=plt.cm.gray)
    ax2.set_title('Coronal ' + (str(npy_img.shape) if trivial else ''), fontsize=15)
    ax2.axis('off')
    img = npy_img[:, :, round(dim[2] / 2)]
    #     img = npy_img[:,:,154]
    ax3.imshow(np.rot90(img), cmap=plt.cm.gray)
    ax3.set_title('Axial', fontsize=15)
    #     ax3.imshow(img, cmap=plt.cm.gray)
    ax3.axis('off')
    # plt.subplot(131); plt.imshow(np.rot90(img), cmap=plt.cm.gray)
    # img = npy_img[:,65,:]
    # plt.subplot(132); plt.imshow(img, cmap=plt.cm.gray)
    # img = npy_img[65,:,:]
    # plt.subplot(133); plt.imshow(np.rot90(img,2), cmap=plt.cm.gray)
    if trivial:
        print(np.max(npy_img))
        print(np.min(npy_img))

    if save:
        plt.savefig(save_name)
    return


def print2d_origin(npy_img, img_name='', save=False, save_name='./test.jpg'):
    '''
    !!!dataset specific
    plot 2d mri images in Sagittal, Coronal and Axial dimension.
    img: 3d ndarray
    '''
    dim = npy_img.shape
    #     print('Dimension: ',npy_img.shape)
    f, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))

    img = npy_img[round(dim[0] / 2), :, :]
    ax1.imshow(img, cmap=plt.cm.gray)
    print(img[0, 0])
    ax1.axis('off')
    img = npy_img[:, round(dim[1] / 2), :]
    ax2.imshow(img, cmap=plt.cm.gray)
    ax2.axis('off')
    img = npy_img[:, :, round(dim[2] / 2)]
    ax3.imshow(img, cmap=plt.cm.gray)
    ax3.axis('off')

    if save:
        plt.savefig(save_name)
    return


def printimg(filename, size=10):
    f, (ax1) = plt.subplots(1, 1, figsize=(size, size))
    npy = plt.imread(filename)
    ax1.imshow(npy)
    ax1.axis('off')
    return


def print_sep(something='-'):
    print('----------------------------------------', something, '----------------------------------------')
    return


# 3D rotatation
def rot_clockwise(arr, n=1):
    return np.rot90(arr, n, (0, 2))


def rot_anticlockwise(arr, n=1):
    return np.rot90(arr, n, (2, 0))


def rot_ixi2abide(img_ixi):
    '''
    to rot IXI to the same direction as ABIDE
    '''
    temp = np.rot90(img_ixi, axes=(1, 2))
    temp = np.rot90(temp, axes=(1, 0))
    return temp


def rot_oasis2abide(img_ixi):
    '''
    to rot OASIS to the same direction as ABIDE
    '''
    temp = np.rot90(img_ixi, axes=(1, 2))
    temp = np.rot90(temp, axes=(0, 1))
    return temp


def time_now():
    return time.strftime('%Y.%m.%d.%H:%M:%S', time.localtime(time.time()))


def sec2hms(seconds):
    m, s = divmod(seconds, 60)
    h, m = divmod(m, 60)
    d, h = divmod(h, 24)
    return str(int(d)) + ' days, ' + str(int(h)) + ' hours, ' + str(int(m)) + ' mins, ' + str(round(s, 3)) + ' secs.'


#     print("%d:%02d:%02d" % (h, m, s))

def my_mkdir(path_name):
    try:
        os.mkdir(path_name)
    except FileExistsError:
        print(path_name, ' exists already!')
    return


def my_makedirs(path_name):
    try:
        os.makedirs(path_name)
    except FileExistsError:
        print(path_name, ' exists already!')
    return


def get_shuffled(imgs, labels):
    temp = np.array([imgs, labels])
    temp = temp.transpose()
    np.random.shuffle(temp)
    image_list = list(temp[:, 0])
    label_list = list(temp[:, 1])
    return image_list, label_list


def minmax_normalize(img_npy):
    '''
    img_npy: ndarray
    '''
    min_value = np.min(img_npy)
    max_value = np.max(img_npy)
    return (img_npy - min_value) / (max_value - min_value)


def z_score_norm(img_npy):
    '''
    img_npy: ndarray
    '''
    return (img_npy - np.mean(img_npy)) / np.std(img_npy)


def dist_check(img_npy):
    '''
    have a look at the distribution of the img
    '''
    f, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))
    ax1.hist(img_npy.reshape(-1))
    ax1.set_title('origin')
    ax2.hist(minmax_normalize(img_npy).reshape(-1))
    ax2.set_title('minmax')
    ax3.hist(z_score_norm(img_npy).reshape(-1))
    ax3.set_title('z score')


def pad_image(img_npy, target_image_shape):
    '''
    image: ndarray
    target_image_shape: tuple or list
    '''
    source_shape = np.asarray(img_npy.shape)
    target_image_shape = np.asarray(target_image_shape)
    edge = (target_image_shape - source_shape) / 2
    pad_width = tuple((i, j) for i, j in zip(np.floor(edge).astype(int), np.ceil(edge).astype(int)))
    padded_img = np.pad(img_npy, pad_width, 'constant', constant_values=0)
    return padded_img, pad_width


# Function to apply random scaling to the image
def patch_wise_prediction(model, data, brain_width, overlap=0, batch_size=1, permute=False, center_patch=True):
    pdb_set = False
    data_shape = data.shape[-3:]
    brain_width = brain_width.copy()
    for i in range(3):
        brain_width[:, i] = np.clip(brain_width[:, i], 0, data_shape[i] - 1)


    patch_shape = tuple([int(dim) for dim in model.input.shape[-3:]])
    #patch_shape = tuple([int(dim) for dim in model.input.shape[-3:]])
    predictions = list()

    brain_wise_image_shape = brain_width[1] - brain_width[0] + 1
    brain_wise_data = data[0, :, brain_width[0, 0]:brain_width[1, 0] + 1,
                      brain_width[0, 1]:brain_width[1, 1] + 1,
                      brain_width[0, 2]:brain_width[1, 2] + 1]

    indices = compute_patch_indices_for_prediction(brain_wise_image_shape, patch_size=patch_shape,
                                                   center_patch=center_patch)
    batch = list()
    i = 0
    if pdb_set:
        pbar = ProgressBar().start()
        print('Predicting patches of single subject...')
    while i < len(indices):
        while len(batch) < batch_size:
            patch = get_patch_from_3d_data(brain_wise_data, patch_shape=patch_shape, patch_index=indices[i])
            batch.append(patch)
            i += 1
        # Ensure that the strides parameter is set to (1, 1, 1) or another valid value
        model.layers[-1].strides = (1, 1, 1)
        prediction = predict(model, np.asarray(batch), batch_size=batch_size, permute=permute)
        if pdb_set:
            pbar.update(int((i - 1) * 100 / (len(indices) - 1)))
        batch = list()
        for predicted_patch in prediction:
            predictions.append(predicted_patch)
    output_shape = [int(model.output.shape[1])] + list(brain_wise_image_shape)
    brain_wise_output = reconstruct_from_patches(predictions, patch_indices=indices, data_shape=output_shape)
    origin_shape = [int(model.output.shape[1])] + list(data.shape[-3:])
    final_output = np.zeros(origin_shape)
    final_output[:, brain_width[0, 0]:brain_width[1, 0] + 1,
    brain_width[0, 1]:brain_width[1, 1] + 1,
    brain_width[0, 2]:brain_width[1, 2] + 1] = brain_wise_output

    return final_output


def get_prediction_labels(prediction, threshold=0.5, labels=None):
    label_data = np.argmax(prediction[0], axis=0) + 1
    label_data[np.max(prediction[0], axis=0) < threshold] = 0
    if labels:
        for value in np.unique(label_data).tolist()[1:]:
            label_data[label_data == value] = labels[value - 1]
    label_data = label_data.astype(np.uint8)
    return label_data


def get_prediction_labels_overlap(prediction, threshold=0.5, uncertainty_threshold=0.2):
    min_component_size_et = 5  # Minimum component size for ET
    neighborhood_size_et = 2  # Neighborhood size for morphological operations for ET
    
    label_data = np.zeros(prediction[0, 0].shape, dtype=np.uint8)
    uncertainty_mask = np.max(prediction[0, :3], axis=0) < uncertainty_threshold

    # Class assignments
    label_data[prediction[0, 1] >= threshold] = 2  # WT (also part of TC)
    label_data[prediction[0, 0] >= threshold] = 1  # TC (but will be overwritten by ET if overlaps)
    label_data[prediction[0, 2] >= threshold] = 4  # ET (takes precedence in overlap with TC)

    # Process ET
    et_mask = label_data == 4
    labeled_array_et, num_features_et = ndimage.label(et_mask, structure=generate_binary_structure(3, 1))
    for i in range(1, num_features_et + 1):
        component = labeled_array_et == i
        if component.sum() < min_component_size_et:
            et_mask[component] = False

    # Removing isolated voxels (additional step)
    et_mask = ndimage.binary_erosion(et_mask, structure=np.ones((3,3,3)))
    et_mask = ndimage.binary_dilation(et_mask, structure=np.ones((3,3,3)))

    struct_et = generate_binary_structure(3, neighborhood_size_et)
    et_mask = binary_closing(et_mask, structure=struct_et)
    et_mask = binary_opening(et_mask, structure=struct_et)

    # Reintegrate processed ET mask
    label_data[et_mask] = 4

    # Apply uncertainty mask selectively (avoiding TC and ET)
    tc_and_et_mask = (label_data == 1) | (label_data == 4)
    label_data[uncertainty_mask & ~tc_and_et_mask] = 0

    return label_data


def prediction_to_image(prediction, affine, brain_mask, threshold=0.5, labels=None, output_dir='', overlap_label=False, uncertainty_threshold=0.1):
    '''
    for multi categories classification please refer to Isensee's repository.
    '''
    #     pdb.set_trace()
    pdb_set = False

    if prediction.shape[1] == 3:
        #  print("RZY")
        if overlap_label:
            data = get_prediction_labels_overlap(prediction, threshold=threshold, uncertainty_threshold=uncertainty_threshold)
            print(data.shape)
        else:
            data = get_prediction_labels(prediction, threshold=threshold, labels=labels)
    else:
        raise RuntimeError("Invalid prediction array shape: {0}".format(prediction.shape))
    masked_output = data * brain_mask
    if np.sum(masked_output - data):
        if pdb_set:
            print_red('changed after mask')
            print_red(output_dir)
            print_red(np.array(np.where(masked_output != data)).shape[1])
        nib.Nifti1Image(data, affine).to_filename(os.path.join(output_dir, "prediction_before_mask.nii.gz"))
    return nib.Nifti1Image(masked_output, affine)


def run_validation_case(data_index, output_dir, model, data_file, training_modalities,
                        threshold=0.5, labels=None, overlap=16,
                        permute=False, center_patch=True, overlap_label=True,
                        final_val=False, uncertainty_threshold=0.2):
    case_name = data_file.root.subject_ids[data_index].decode()
    case_dir = os.path.join(output_dir, case_name)
    if not os.path.exists(case_dir):
        os.makedirs(case_dir)

    affine = np.load('affine.npy')

    test_data = np.array([modality_img[data_index, 0]
                          for modality_img in [data_file.root.t1,
                                               data_file.root.t1ce,
                                               data_file.root.flair,
                                               data_file.root.t2]])[np.newaxis]

    expected_shape = (240, 240, 155)
    if test_data.shape[-3:] != expected_shape:
            test_data = test_data[..., :expected_shape[0], :expected_shape[1], :expected_shape[2]]
    brain_mask = np.any(test_data != 0, axis=1)[0]

    for i, modality in enumerate(training_modalities):
        image = nib.Nifti1Image(test_data[0, i], affine)
        image.to_filename(os.path.join(case_dir, f"data_{modality}.nii.gz"))

    if not final_val and 'truth' in data_file.root:
        test_truth = nib.Nifti1Image(data_file.root.truth[data_index][0], affine)
        test_truth.to_filename(os.path.join(case_dir, "truth.nii.gz"))

    brain_width = data_file.root.brain_width[data_index]

    prediction = patch_wise_prediction(model=model, data=test_data, brain_width=brain_width,
                                       overlap=overlap, permute=permute, center_patch=center_patch)[np.newaxis]
    
    prediction_image = prediction_to_image(prediction, affine, brain_mask,
                                           threshold=threshold, labels=labels, output_dir=case_dir,
                                           overlap_label=overlap_label, uncertainty_threshold=uncertainty_threshold)
    
    if isinstance(prediction_image, list):
        for i, image in enumerate(prediction_image):
            image.to_filename(os.path.join(output_dir, f"prediction_{i+1}.nii.gz"))
    else:
        prediction_image.to_filename(os.path.join(output_dir, f"{case_name}.nii.gz"))


import time

def run_validation_cases(validation_keys_file, model_file, training_modalities, labels, hdf5_file,
                         output_dir=".", threshold=0.5, overlap=16,
                         permute=False, center_patch=True, overlap_label=True, final_val=False):
    validation_indices = pickle_load(validation_keys_file)
    model = load_old_model(model_file)
    print(model.summary())
    
    data_file = tables.open_file(hdf5_file, "r")

    for index in tqdm(validation_indices):
        if 'subject_ids' in data_file.root:
            case_directory = os.path.join(output_dir, data_file.root.subject_ids[index].decode('utf-8'))
        else:
            case_directory = os.path.join(output_dir, "validation_case_{}".format(index))
        
        # Measure inference time for each case
        start_time = time.time()
        
        run_validation_case(data_index=index, output_dir=case_directory, model=model, data_file=data_file,
                            training_modalities=training_modalities, labels=labels,
                            threshold=threshold, overlap=overlap, permute=permute, center_patch=center_patch,
                            overlap_label=overlap_label,
                            final_val=final_val)
        
        end_time = time.time()
        inference_time = end_time - start_time
        
        print(f"Inference time for case {index}: {inference_time} seconds")

    data_file.close()


#     pdb.set_trace()

def predict(model, data, batch_size=1, permute=False):
    # Ensure the data has the correct shape
    if len(data.shape) != 5:
        raise ValueError("Input data must have 5 dimensions (batch_size, channels, depth, height, width)")

    # Permute the data if needed
    if permute:
        data = np.transpose(data, (0, 2, 3, 4, 1))

    # Predict using the model
    predictions = model.predict(x=data, batch_size=batch_size)

    return predictions

