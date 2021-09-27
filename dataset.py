# ------------------------------------------------------
# File: dataset.py
# Author: Alina Dima <alina.dima@tum.de>
#
# Data loader
# Our dataset is private, but feel free to have a look at the implementation of the data loader
# ------------------------------------------------------


import math
import random
from typing import Tuple
import numpy as np
import SimpleITK as sitk
import tensorflow as tf
from data_utils.seedpoints import crop_seedpoints


class Dataset(object):
    def __init__(
        self,
        batch_size,
        crop_size,
        cross_validation,
        data_source,
        limit_images=None
    ):
        self.cv = cross_validation
        self.data_source = data_source
        self.train_filenames = self.cv.get_data_paths('train', data_source=self.data_source)
        self.test_filenames = self.cv.get_data_paths('test', data_source=self.data_source)
        self.val_filenames = self.cv.get_data_paths('validation', data_source=self.data_source)
        self.limit_images = limit_images
        if self.limit_images is not None:
            self.test_filenames_original = self.test_filenames.copy()
            self.test_filenames = [x[:self.limit_images]
                                   for x in self.test_filenames]
            self.train_filenames_original = self.train_filenames.copy()
            self.train_filenames = [x[:self.limit_images]
                                    for x in self.train_filenames]
        self.dataset_size_train = len(self.train_filenames[0])
        self.dataset_size_test = len(self.test_filenames[0])
        self.dataset_size_val = len(self.val_filenames[0])
        self.batch_size = batch_size
        self.crop_size = crop_size

        self.augmentation_settings = {
            'p_rotation': 0.5,
            'p_translation': 0.5,
            'p_scaling': 0.5,
            'p_gamma_correction': 0.5,
            'p_gaussian_noise': 0.5,
            'rotation_params': [10., 10., 10.],
            'translation_params': [20., 20., 5.],
            'scaling_params': 0.2,
            'gaussian_sigma': [0.05, 0.1],
            'gamma_correction_range': [0.9, 1.1]
        }

    def get_dataset(self, dataset_type):
        arguments = {
            'batch_size': self.batch_size,
            'crop_size': self.crop_size,
            'augmentation_settings': self.augmentation_settings,
        }
        if dataset_type == 'train':
            return SeedpointCropLoader(
                filenames=self.train_filenames, **arguments,
                data_augmentation=True, shuffle=True)
        elif dataset_type == 'train_no_augmentation':
            return SeedpointCropLoader(
                filenames=self.train_filenames, **arguments,
                data_augmentation=False, shuffle=False)
        elif dataset_type == 'test':
            return SeedpointCropLoader(
                filenames=self.test_filenames, **arguments,
                data_augmentation=False, shuffle=False)
        elif dataset_type == 'val':
            return SeedpointCropLoader(
                filenames=self.val_filenames, **arguments,
                data_augmentation=False, shuffle=False)
        else:
            raise Exception('Unknown dataset type')


def get_image_center(img):
    size_vector = list(img.GetSize())
    center_idx = [int(np.ceil(x / 2)) for x in size_vector]
    return img.TransformIndexToPhysicalPoint(tuple(center_idx))


class SeedpointCropLoader(tf.keras.utils.Sequence):
    def __init__(
            self, filenames,
            shuffle,
            crop_size,
            data_augmentation=False,
            augmentation_settings=None,
            batch_size=1
    ):
        self.data_augmentation = data_augmentation
        self.img_filenames, self.seg_filenames = filenames
        self.shuffle = shuffle
        self.batch_size = batch_size
        self.num_images = len(self.img_filenames)
        self.crop_size = crop_size
        self.augmentation_settings = augmentation_settings
        self.subject_ids = [x.split('/')[-1][:6] for x in self.img_filenames]
        if self.num_images != len(self.seg_filenames):
            Exception('Number of images and masks is different!')
        self.indices = list(range(self.num_images))
        if self.shuffle:
            random.shuffle(self.indices)

    def on_epoch_end(self):
        if self.shuffle:
            random.shuffle(self.indices)

    def __len__(self):
        return self.num_images // self.batch_size

    def read_data_pair(self, idx):
        data_filenames = [self.img_filenames[idx], self.seg_filenames[idx]]
        data_raw_sitk = [sitk.ReadImage(x) for x in data_filenames]
        subject_id = self.subject_ids[idx]
        aug_params = None

        # Data augmentation sitk
        if self.data_augmentation:
            aug_params = generate_random_augmentation_parameters(self.augmentation_settings)
            data_sitk = []
            for is_mask, x_sitk in zip([False, True], data_raw_sitk):
                x_augmented = data_augmentation_sitk(x_sitk, {**aug_params, 'is_mask': is_mask})
                data_sitk.append(x_augmented)
        else:
            data_sitk = data_raw_sitk

        # Crop image and segmentation
        data_np = [sitk.GetArrayFromImage(x) for x in data_sitk]
        crop_center = crop_seedpoints[subject_id]
        img, seg = [crop_input(x, self.crop_size, crop_center) for x in data_np]
        img = rescale_array(img, (-1., 1.))  # Rescale to the range [-1, 1]

        # Data augmentation numpy
        if self.data_augmentation:
            img = data_augmentation_np(img, aug_params)

        auxiliary_data = {
            'augmentation': aug_params,
            'subject_id': subject_id,
        }

        # Convert to tf shapes
        img, seg = [np.expand_dims(x, axis=-1) for x in [img, seg]]
        return img.astype(np.float32), seg.astype(np.float32), auxiliary_data

    def __getitem__(self, batch_idx):
        batch_indices = self.indices[batch_idx * self.batch_size: (batch_idx + 1) * self.batch_size]
        batch_data = [self.read_data_pair(x) for x in batch_indices]
        imgs = [np.expand_dims(x[0], axis=0) for x in batch_data]
        segs = [np.expand_dims(x[1], axis=0) for x in batch_data]
        img_info = [x[2] for x in batch_data]
        img = np.concatenate(imgs, axis=0)
        seg = np.concatenate(segs, axis=0)
        return img, seg, img_info


def rescale_array_to_range(
        input_array: np.ndarray,
        source_range: Tuple[float, float],
        target_range: Tuple[float, float]):
    t_min, t_max = target_range
    s_min, s_max = source_range
    assert t_max - \
        t_min > 0, f'rescale_array: Invalid target range [{t_min}, {t_max}]'
    # In order to avoid overflow issues
    input_array = input_array.astype('float64')
    x = t_min + ((input_array - s_min) * (t_max - t_min) / (s_max - s_min))
    return x


def generate_random_augmentation_parameters(augmentation_settings):
    aug_set = augmentation_settings
    scale = aug_set['scaling_params']
    gamma1, gamma2 = aug_set['gamma_correction_range']
    s_min, s_max = aug_set['gaussian_sigma']
    augmentation_parameters = {
        'rotate_image': random.uniform(0, 1) <= aug_set['p_rotation'],
        'rotation_angles': [random.uniform(-i, i) for i in aug_set['rotation_params']],
        'scale_image': random.uniform(0, 1) <= aug_set['p_scaling'],
        'scale': random.uniform(1 - scale, 1 + scale),
        'translate_image': random.uniform(0, 1) <= aug_set['p_translation'],
        'translation_offset': [random.uniform(-i, i) for i in aug_set['translation_params']],
        'gamma_correction': random.uniform(0, 1) <= aug_set['p_gamma_correction'],
        'gamma': random.uniform(gamma1, gamma2),
        'gaussian_noise': random.uniform(0, 1) <= aug_set['p_gaussian_noise'],
        'sigma': random.uniform(s_min, s_max),
    }
    return augmentation_parameters


def data_augmentation_sitk(img, augmentation_parameters):
    transformations = []
    image_center = get_image_center(img)
    if augmentation_parameters['rotate_image']:
        angles = [x * math.pi / 180. for x in augmentation_parameters['rotation_angles']]
        transform_r = sitk.AffineTransform(3)
        transform_r.Rotate(1, 2, angle=angles[0])  # Rotation about the x axis
        transform_r.Rotate(0, 2, angle=angles[1])  # Rotation about the y axis
        transform_r.Rotate(0, 1, angle=angles[2])  # Rotation about the z axis
        transform_r.SetCenter(image_center)
        transformations.append(transform_r)

    if augmentation_parameters['scale_image']:
        transform_s = sitk.AffineTransform(3)
        transform_s.Scale(augmentation_parameters['scale'])
        transform_s.SetCenter(image_center)
        transformations.append(transform_s)

    if augmentation_parameters['translate_image']:
        transform_t = sitk.AffineTransform(3)
        t_pixels = augmentation_parameters['translation_offset']
        t_coord_space = tuple([pix * dim for pix, dim in zip(t_pixels, img.GetSpacing())])
        transform_t.Translate(t_coord_space)
        transformations.append(transform_t)

    if len(transformations) == 0:
        return img

    composite_transformation = sitk.CompositeTransform(transformations)
    if augmentation_parameters['is_mask']:
        interpolator = sitk.sitkNearestNeighbor
    else:
        interpolator = sitk.sitkLinear
    x_sitk_augmented = sitk.Resample(img, img, composite_transformation, interpolator, 0)
    return x_sitk_augmented


def data_augmentation_np(img, augmentation_parameters):
    if augmentation_parameters['gamma_correction']:
        img = gamma_correction(img, augmentation_parameters['gamma'])
    if augmentation_parameters['gaussian_noise']:
        img = gaussian(img, augmentation_parameters['sigma'])
    return img


def rescale_array(input_array: np.ndarray, target_range: Tuple[float, float]) -> np.ndarray:
    input_array = input_array.astype('float64')
    t_min, t_max = target_range
    a_min, a_max = input_array.min(), input_array.max()
    assert a_max - a_min > 0, f'rescale_array: Source array consists of identical values: {a_min}'
    assert t_max - t_min > 0, f'rescale_array: Invalid target range [{t_min}, {t_max}]'
    x = t_min + ((input_array - a_min) * (t_max - t_min) / (a_max - a_min))
    return x


def gamma_correction(x: np.ndarray, gamma: float) -> np.ndarray:
    original_range = (np.min(x), np.max(x))
    x = rescale_array(x, (0., 1.))
    x = np.power(x, gamma)
    x = rescale_array(x, original_range)
    return x


def gaussian(image, stddev):
    return image + np.random.normal(scale=stddev, size=image.shape)


def crop_3d_array(x: np.ndarray, crop_parameters, crop_size):
    final_crop_parameters = []
    image_size = x.shape
    for i in range(3):
        crop1, crop2 = crop_parameters[i]
        im_size = image_size[i]
        if im_size < crop_size[i]:
            print(im_size, crop_size, crop_parameters[i])
            import ipdb
            ipdb.set_trace()
        assert im_size >= crop_size[i], f'Image size in direction {i} is smaller than the crop size'
        if crop2 > im_size:
            final_crop_parameters.append((im_size - crop_size[i], im_size))
        elif crop1 < 0:
            final_crop_parameters.append((0, crop_size[i]))
        else:
            final_crop_parameters.append((crop1, crop2))
    cp = final_crop_parameters
    x = x[cp[0][0]:cp[0][1], cp[1][0]:cp[1][1], cp[2][0]:cp[2][1]]
    return x


def crop_input(x: np.ndarray, crop_size, crop_center) -> np.ndarray:
    crop_parameters = [
        (crop_center[i] - hs // 2, crop_center[i] + hs // 2)
        for i, hs in enumerate(crop_size)]
    x = crop_3d_array(x, crop_parameters, crop_size)
    return x
