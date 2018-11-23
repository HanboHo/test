# Copyright 2017 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

r"""Convert raw COCO panoptic dataset to segmentation dataset."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import time
import argparse
import io
import json
import os
import numpy as np
import PIL.Image
import PIL.ImageFont
import PIL.ImageDraw
import PIL.JpegImagePlugin

from panoptic.utils import dataset_util

BASE_DIR = '/mnt/internal/99_Data/07_COCO'
PANOPTIC_DIR = '/panoptic_annotations_trainval2017/annotations'
BASE_PANOPTIC_DIR = BASE_DIR + PANOPTIC_DIR
REDUCED_LABELID_MAPPING_FILE = '/home/chen/work/06_tf_models/tf_models/models' \
                               '/research/panoptic/dataset_tools' \
                               '/original_to_reduced_labels.json'

parser = argparse.ArgumentParser()
parser.add_argument('--reduced_labelid_mapping_file', nargs='?',
                    default=REDUCED_LABELID_MAPPING_FILE,
                    help="Training panoptic segmentation directory.")
parser.add_argument('--train_panoptic_dir', nargs='?',
                    default=BASE_PANOPTIC_DIR + '/panoptic_train2017',
                    help="Training panoptic segmentation directory.")
parser.add_argument('--val_panoptic_dir', nargs='?',
                    default=BASE_PANOPTIC_DIR + '/panoptic_val2017',
                    help="Validation panoptic segmentation directory.")
parser.add_argument('--train_annotations_file', nargs='?',
                    default=BASE_PANOPTIC_DIR + '/panoptic_train2017.json',
                    help="Training annotations JSON file.")
parser.add_argument('--val_annotations_file', nargs='?',
                    default=BASE_PANOPTIC_DIR + '/panoptic_val2017.json',
                    help="Validation annotations JSON file.")
FLAGS = parser.parse_args()


def create_train_ids(filename,
                     annotations_list,
                     panoptic_dir,
                     reduced_labelid_mapping):
    """Converts label ids to train ids.

    Args:
        filename: name of file without extension.
        annotations_list: list of dicts with keys:
            [u'segments_info', u'file_name', u'image_id']
        annotations_list['segments_info']: list of dicts with keys:
            [u'id', u'category_id', u'iscrowd', u'bbox', u'area']
        Notice that bounding box coordinates in the official COCO dataset are
        given as [x, y, width, height] tuples using absolute coordinates where
        x, y represent the top-left (0-indexed) corner.  This function converts
        to the format expected by the Tensorflow Object Detection API (which is
        which is [ymin, xmin, ymax, xmax] with coordinates normalized relative
        to image size).
        panoptic_dir: directory containing the panoptic label files.
        reduced_labelid_mapping: Mapping to remove unused label classes.
    Returns:
        segmentation_ids: 2D array of train ids.

    Raises:
        ValueError: if the image pointed to by data['filename'] is not a valid
        JPEG
    """
    full_path_panoptic = os.path.join(panoptic_dir, filename + ".png")

    with open(full_path_panoptic, 'rb') as fid:
        panoptic_data = fid.read()

    encoded_png_io = io.BytesIO(panoptic_data)

    panoptic_image = PIL.Image.open(encoded_png_io)
    panoptic_ids = dataset_util.convert_rgb_to_ids(np.asarray(panoptic_image))

    label_shape = panoptic_ids.shape
    panoptic_ids = np.reshape(panoptic_ids, panoptic_ids.size)
    segmentation_ids = np.zeros(panoptic_ids.shape, dtype=np.uint8)

    # is_crowd = []
    label_train_map = dict()
    for annotations in annotations_list:
        objects_list = annotations['segments_info']
        for objects in objects_list:
            label_train_map[objects['id']] = objects['category_id']

    map_keys = label_train_map.keys()
    for i in map_keys:
        segmentation_ids[panoptic_ids == i] = \
            reduced_labelid_mapping[str(label_train_map[i])]
    segmentation_ids = np.reshape(segmentation_ids, label_shape)
    # PIL.Image.fromarray(np.reshape(segmentation_ids, label_shape)).show()
    return segmentation_ids


def _create_train_ids_from_label_ids(annotations_file,
                                     panoptic_dir,
                                     reduced_labelid_mapping_file):
    """Creates COCO train ids from label ids for pixel-wise semantic
    segmentation.

    Args:
        annotations_file: JSON file containing bounding box annotations.
        panoptic_dir: Directory containing the panoptic segmentation files.
        reduced_labelid_mapping_file: JSON file containing mapping to reduced
            labelids.
    """
    output_dir = panoptic_dir.replace(panoptic_dir.split('/')[-1],
                                      panoptic_dir.split('/')[-1].replace(
                                          'panoptic', 'segmentation'))
    if not os.path.isdir(output_dir):
        os.mkdir(output_dir)
    else:
        assert not os.path.isfile(output_dir), "File exists in " + output_dir

    with open(annotations_file, 'r') as fid:
        groundtruth_data = json.load(fid)

    with open(reduced_labelid_mapping_file, 'r') as fid:
        mapping_data = json.load(fid)

    images = groundtruth_data['images']

    annotations_index = {}
    if 'annotations' in groundtruth_data:
        print('Found groundtruth annotations. Building annotations index.')
        for annotation in groundtruth_data['annotations']:
            image_id = annotation['image_id']
            if image_id not in annotations_index:
                annotations_index[image_id] = []
            annotations_index[image_id].append(annotation)

    missing_annotation_count = 0
    for image in images:
        image_id = image['id']
        if image_id not in annotations_index:
            missing_annotation_count += 1
            annotations_index[image_id] = []
    print('{} images are missing annotations.'.format(missing_annotation_count))

    for idx, image in enumerate(images):
        filename = image['file_name'].split('.')[0]
        if idx % 1000 == 0:
            print('Processing image {} of {}'.format(idx, len(images)))
        annotations_list = annotations_index[image['id']]
        train_ids = create_train_ids(filename, annotations_list,
                                     panoptic_dir, mapping_data)
        output_path = os.path.join(output_dir, filename + ".png")
        PIL.Image.fromarray(train_ids).save(output_path)
        print('writing image: {}'.format(output_path))
    print('Finished writing.')


def main():
    assert FLAGS.reduced_labelid_mapping_file, \
        FLAGS.reduced_labelid_mapping_file + 'is missing.'
    assert FLAGS.train_panoptic_dir, '`train_panoptic_dir` missing.'
    assert FLAGS.val_panoptic_dir, '`val_panoptic_dir` missing.'
    assert FLAGS.train_annotations_file, '`train_annotations_file` missing.'
    assert FLAGS.val_annotations_file, '`val_annotations_file` missing.'

    _create_train_ids_from_label_ids(
        FLAGS.train_annotations_file,
        FLAGS.train_panoptic_dir,
        FLAGS.reduced_labelid_mapping_file)
    _create_train_ids_from_label_ids(
        FLAGS.val_annotations_file,
        FLAGS.val_panoptic_dir,
        FLAGS.reduced_labelid_mapping_file)


if __name__ == '__main__':
    main()
