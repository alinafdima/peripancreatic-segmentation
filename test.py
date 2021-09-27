# ------------------------------------------------------
# File: test.py
# Author: Alina Dima <alina.dima@tum.de>
#
# ------------------------------------------------------


import os
from os.path import join
import numpy as np
import SimpleITK as sitk
import tensorflow as tf
from main import metric_names
from typing import List, Tuple
from collections.abc import Callable

from main import MainLoop


color_map_gmr = {
    'tp': 2,  # Green in ITKSnap
    'fp': 6,  # Magenta in ITKSnap
    'fn': 1,  # Red in ITKSnap
}


def combine_suffixes(suffix_list):
    return '_'.join([x for x in suffix_list if x is not None])


def get_metrics(gt: np.ndarray, pred: np.ndarray, metrics: List[Callable]) -> List[float]:
    """
    Evaluates multiple TP/FP/FN metrics without having to compute the TP/FP/FN values multiple times

    Args:
        gt (np.ndarray): Ground truth
        pred (np.ndarray): Prediction
        metrics (list[function]): list of metric functions

    Returns:
        list[int]: List of metric values
    """
    tp = np.sum(gt * pred)
    fn = np.sum(gt * (1 - pred))
    fp = np.sum((1 - gt) * pred)
    ret = []
    for metric in metrics:
        if metric.__name__ == 'dice_score':
            ret.append(metric(tp, fp, fn))
        elif metric.__name__ == 'iou':
            ret.append(metric(tp, fp, fn))
        elif metric.__name__ == 'precision':
            ret.append(metric(tp, fp))
        elif metric.__name__ == 'recall':
            ret.append(metric(tp, fn))
        else:
            Exception(f'Unkown metric {metric.__name__}')
    return ret


def iou(tp: np.ndarray, fp: np.ndarray, fn: np.ndarray) -> float:
    if tp == 0:
        return 0
    return tp / (tp + fp + fn)


def dice_score(tp: np.ndarray, fp: np.ndarray, fn: np.ndarray) -> float:
    if tp == 0:
        return 0
    return 2 * tp / (2 * tp + fp + fn)


def precision(tp: np.ndarray, fp: np.ndarray) -> float:
    if tp == 0:
        return 0
    return tp / (tp + fp)


def recall(tp: np.ndarray, fn: np.ndarray) -> float:
    if tp == 0:
        return 0
    return tp / (tp + fn)


def get_predidiction_with_errors(gt: np.ndarray, pred: np.ndarray, color_map=None):
    """
    Returns a prediction image with color-coded TP, FP, and FN.
    TODO: Check if calling the color map with an argument different from None
    sets it for all subsequent calls with an arugment of None

    Args:
        gt (np.ndarray): Binary ground truth predictions
        pred (np.ndarray): Binary predictions
        color_map (dict, optional): Defines what color to set for each of TP, FP or FN.

    Returns:
        [type]: color-coded prediction image
    """
    if np.max(gt) != 1 or np.min(gt) != 0:
        print('Min and max values of the ground truth segmentation are different from 0 and 1.')
        print('Enter debug mode..')
        import ipdb
        ipdb.set_trace()
    if color_map is None:
        color_map = color_map_gmr
    tp = gt * pred
    fn = gt * (1 - pred)
    fp = (1 - gt) * pred
    output = tp * color_map['tp'] + fn * color_map['fn'] + fp * color_map['fp']
    return output


def save_quantitative_results(quantitative_results: List[Tuple[int, List[float]]], metric_names: List[str], output_file: str):
    """
    Saves the quantitative results into an output file.
    The quantitative results are passed as a list of entries, where each entry consists
        of the entry id, as well as a list of metric values.
    Example quantitative results: [(0, [0.81, 0.78, 0.88), (1, [0.62, 0.7, 0.5), ...]
    A list of metric names in the same oreder they appear in the quantitative results is also
        to be provided.

    Args:
        quantitative_results (list(tuple(int, list[int]))): the quantitative results
        metric_names (list[str]): the names of the metrics
        output_file (str): the output location
    """
    metric_means = []
    with open(output_file, 'w') as f:
        f.write('Overall results: AVERAGE +- STD DEVIATION\n')
        for idx, metric_name in enumerate(metric_names):
            raw_numbers = np.array([x[1][idx] for x in quantitative_results])
            metric_avg = raw_numbers.mean()
            metric_std = raw_numbers.std()
            metric_means.append(metric_avg)
            f.write(
                f'{metric_name: <15} {metric_avg:.4f} +- {metric_std:.4f}\n')
        f.write('\n\nIndividual examples:\n')
        f.write(f'Metrics order {" - ".join(metric_names)}\n')
        for (idx, values) in quantitative_results:
            rounded_values = [str(round(i, 4)) for i in values]
            f.write(f'{idx:<3}: {", ".join(rounded_values)}\n')
    return metric_means


class Testing:
    def __init__(self, loop: MainLoop):
        self.loop = loop
        if loop.model is None:
            loop.initialize_network(is_training=False)
            loop.compile_model()
        self.network = loop.model
        self.quantitative_folder = join(loop.experiment_folder, 'quantitative_results')
        if not os.path.exists(self.quantitative_folder):
            os.makedirs(self.quantitative_folder)
        self.color_map = color_map_gmr

    def test(self, split='test', save_suffix=None, save_images=False):
        print('Call to test_model/Testing.test')
        dataset = self.loop.dataset.get_dataset(split)
        current_epoch = self.loop.current_epoch
        test_id = combine_suffixes(['images', f'{current_epoch:03d}', save_suffix, split])
        quantitative_output_file = join(self.quantitative_folder, f'{test_id}.txt')
        save_folder = join(self.loop.experiment_folder, test_id)
        if save_images:
            if not os.path.exists(save_folder):
                os.makedirs(save_folder)
        quantitative_results = []

        for batch_img, batch_seg, img_info_raw in dataset:
            img_info = img_info_raw[0]
            subject_id = img_info['subject_id']

            # Test
            prediction = self.network.predict(batch_img)
            prediction_binary = (prediction > 0.5).astype(np.int8)

            prediction_tpfpfn = get_predidiction_with_errors(
                batch_seg, prediction_binary, self.color_map)

            # Save images
            if save_images:
                output_images = [batch_img, prediction_tpfpfn, prediction_binary]
                save_suffixes = ['01_img', '02_pred_cm', '03_pred_binary']
                for im, suffix in zip(output_images, save_suffixes):
                    save_batch_image(im, img_info, save_folder, suffix)
                print(f'Finished saving {subject_id}')

            # Evaluate
            ev = self.network.evaluate(batch_img, batch_seg)
            loss = ev[0]
            metrics = [e * 100 for e in ev[1:]]  # Human readable
            evaluation = [loss] + metrics
            quantitative_results.append((subject_id, evaluation))

        metric_means = save_quantitative_results(
            quantitative_results, metric_names, quantitative_output_file)
        print(f'Finished evaluating the {split} set.')

        return metric_means


def save_batch_image(image, img_info, save_folder, save_suffix):
    subject_id = img_info['subject_id']
    output_file = join(save_folder, f'test_{subject_id}_{save_suffix}.nii.gz')
    img_sitk = sitk.GetImageFromArray(image[0, :, :, :, 0])
    sitk.WriteImage(img_sitk, output_file)
