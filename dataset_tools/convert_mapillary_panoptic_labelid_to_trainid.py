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

BASE_DIR = '/mnt/internal/99_Data/08_Mapillary'
PANOPTIC_DIR = '/mapillary-vistas-dataset_public_v1.1'
BASE_PANOPTIC_DIR = BASE_DIR + PANOPTIC_DIR
MAPPING_FILE = '/home/chen/work/06_tf_models/tf_models/models' \
               '/research/panoptic/dataset_tools' \
               '/original_to_rearranged_labels.json'

parser = argparse.ArgumentParser()
parser.add_argument('--mapping_file', nargs='?',
                    default=MAPPING_FILE,
                    help="Training panoptic segmentation directory.")
parser.add_argument('--train_panoptic_dir', nargs='?',
                    default=BASE_PANOPTIC_DIR + '/training/instances',
                    help="Training panoptic segmentation directory.")
parser.add_argument('--val_panoptic_dir', nargs='?',
                    default=BASE_PANOPTIC_DIR + '/validation/instances',
                    help="Validation panoptic segmentation directory.")
parser.add_argument('--train_annotations_file', nargs='?',
                    default=BASE_PANOPTIC_DIR + '/training/panoptic/panoptic_2018.json',
                    help="Training annotations JSON file.")
parser.add_argument('--val_annotations_file', nargs='?',
                    default=BASE_PANOPTIC_DIR + '/validation/panoptic/panoptic_2018.json',
                    help="Validation annotations JSON file.")
FLAGS = parser.parse_args()


def create_train_ids(filename,
                     annotations_list,
                     panoptic_dir,
                     labelid_mapping):
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
        labelid_mapping: Mapping to remove unused label classes.
    Returns:
        segmentation_ids: 2D array of train ids.

    Raises:
        ValueError: if the image pointed to by data['filename'] is not a valid
        JPEG
    """
    instance_data = dataset_util.read_data(
        panoptic_dir, filename, dataset_util.FLAGS.panoptic_format)

    instance_image = PIL.Image.open(io.BytesIO(instance_data))
    instance_array = np.array(instance_image, dtype=np.uint16)
    # instance_ids_array = np.array(instance_array % 256, dtype=np.uint8)
    instance_label_array = np.array(instance_array / 256, dtype=np.uint8)
    instance_label_array += 1
    instance_label_array[instance_label_array == 66] = 0
    instance_label_array_mem = np.array(instance_label_array)
    for k, v in labelid_mapping.items():
        instance_label_array[instance_label_array_mem == int(k)] = int(v)
    # segmentation_image = PIL.Image.fromarray(
    #     np.array(instance_label_array, dtype=np.uint8))
    # output_io = io.BytesIO()
    # segmentation_image.save(output_io, format='PNG')
    # segmentation_data = output_io.getvalue()
    # PIL.Image.open(io.BytesIO(segmentation_data)).show()
    # PIL.Image.fromarray(instance_label_array_mem).show()
    return instance_label_array


def _create_train_ids_from_label_ids(annotations_file,
                                     panoptic_dir,
                                     mapping_file):
    """Creates train ids from label ids for pixel-wise semantic
    segmentation.

    Args:
        annotations_file: JSON file containing bounding box annotations.
        panoptic_dir: Directory containing the panoptic segmentation files.
        mapping_file: JSON file containing mapping to labelids.
    """
    output_dir = panoptic_dir.replace(panoptic_dir.split('/')[-1],
                                      'segmentation')

    if not os.path.isdir(output_dir):
        os.mkdir(output_dir)
    else:
        assert not os.path.isfile(output_dir), "File exists in " + output_dir

    with open(annotations_file, 'r') as fid:
        groundtruth_data = json.load(fid)

    with open(mapping_file, 'r') as fid:
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
        print('writing image {:d}: {:s}'.format(idx, output_path))
    print('Finished writing.')


def main():
    assert FLAGS.mapping_file, FLAGS.mapping_file + 'is missing.'
    assert FLAGS.train_panoptic_dir, '`train_panoptic_dir` missing.'
    assert FLAGS.val_panoptic_dir, '`val_panoptic_dir` missing.'
    assert FLAGS.train_annotations_file, '`train_annotations_file` missing.'
    assert FLAGS.val_annotations_file, '`val_annotations_file` missing.'

    _create_train_ids_from_label_ids(
        FLAGS.train_annotations_file,
        FLAGS.train_panoptic_dir,
        FLAGS.mapping_file)
    _create_train_ids_from_label_ids(
        FLAGS.val_annotations_file,
        FLAGS.val_panoptic_dir,
        FLAGS.mapping_file)


if __name__ == '__main__':
    main()
