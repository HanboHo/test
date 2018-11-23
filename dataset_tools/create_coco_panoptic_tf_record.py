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

r"""Convert raw COCO dataset to TFRecord for panoptic segmentation."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import time
import sys
import math
import hashlib
import io
import json
import os
import numpy as np
import PIL.Image
import PIL.ImageFont
import PIL.ImageDraw
import PIL.JpegImagePlugin

from pycocotools import mask
import tensorflow as tf

from panoptic.utils import dataset_util
from panoptic.utils import label_map_util

SOURCE_PATH = '/mnt/storage/07_COCO'
PANOPTIC_FOLDER = '/panoptic_annotations_trainval2017/annotations'

flags = tf.app.flags
tf.flags.DEFINE_integer('num_shards', 100,
                        'Number of shards for tf.record default: 10.')
tf.flags.DEFINE_boolean('include_masks', True,
                        'Whether to include instance segmentations masks '
                        '(PNG encoded) in the result. default: False.')

tf.flags.DEFINE_string('train_image_dir',
                       SOURCE_PATH + '/train2017',
                       'Training image directory.')
tf.flags.DEFINE_string('val_image_dir',
                       SOURCE_PATH + '/val2017',
                       'Validation image directory.')

tf.flags.DEFINE_string('train_segmentation_dir',
                       SOURCE_PATH + PANOPTIC_FOLDER + '/segmentation_train2017',
                       'Training panoptic segmentation directory.')
tf.flags.DEFINE_string('val_segmentation_dir',
                       SOURCE_PATH + PANOPTIC_FOLDER + '/segmentation_val2017',
                       'Validation panoptic segmentation directory.')

tf.flags.DEFINE_string('train_panoptic_dir',
                       SOURCE_PATH + PANOPTIC_FOLDER + '/panoptic_train2017',
                       'Training panoptic segmentation directory.')
tf.flags.DEFINE_string('val_panoptic_dir',
                       SOURCE_PATH + PANOPTIC_FOLDER + '/panoptic_val2017',
                       'Validation panoptic segmentation directory.')

tf.flags.DEFINE_string('train_annotations_file',
                       SOURCE_PATH + PANOPTIC_FOLDER + '/panoptic_train2017.json',
                       'Training annotations JSON file.')
tf.flags.DEFINE_string('val_annotations_file',
                       SOURCE_PATH + PANOPTIC_FOLDER + '/panoptic_val2017.json',
                       'Validation annotations JSON file.')

tf.flags.DEFINE_string('test_image_dir',
                       SOURCE_PATH + '/test2017',
                       'Test image directory.')
tf.flags.DEFINE_string('test_segmentation_dir',
                       '',
                       'Test panoptic segmentation directory.')
tf.flags.DEFINE_string('test_panoptic_dir',
                       '',
                       'Test panoptic segmentation directory.')
tf.flags.DEFINE_string('test_annotations_file',
                       '',
                       'Test annotations JSON file.')

tf.flags.DEFINE_string('output_dir',
                       '/mnt/internal/99_Data/coco_tf_records',
                       'Output data directory.')
tf.flags.DEFINE_string('reduced_labelid_mapping_file',
                       '/home/chen/work/06_tf_models/tf_models/models/research'
                       '/panoptic/dataset_tools'
                       '/original_to_reduced_labels.json',
                       'Training panoptic segmentation directory.')
FLAGS = flags.FLAGS


def create_panoptic_test_tf_example(filename_raw, image_dir):
    """Converts image and annotations to a tf.Example proto.

    Args:
        filename_raw: path to file
        image_dir: directory containing the image files.
    Returns:
        example: The converted tf.Example
        num_annotations_skipped: Number of (invalid) annotations that were
        ignored.

    Raises:
        ValueError: if the image pointed to by data['filename'] is not a valid
        JPEG
    """
    filename = filename_raw.split('/')[-1]
    filename = filename.split('.')[0]

    # Read data from files.
    image_data = dataset_util.read_data(
        image_dir, filename, dataset_util.FLAGS.image_format)

    # 1. Processing image
    image = PIL.Image.open(io.BytesIO(image_data))
    image_height = image.height
    image_width = image.width

    example = dataset_util.image_panoptic_test_to_tf_example(
        image_data, filename, image_height, image_width, 3)
    return example


def create_panoptic_tf_example(image,
                               annotations_list,
                               image_dir,
                               segmentation_dir,
                               panoptic_dir,
                               category_index,
                               reduced_labelid_mapping):
    """Converts image and annotations to a tf.Example proto.

    Args:
        image: dict with keys:
            [u'license', u'file_name', u'coco_url', u'height', u'width',
            u'date_captured', u'flickr_url', u'id']
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
        image_dir: directory containing the image files.
        segmentation_dir: directory containing the segmentation files.
        panoptic_dir: directory containing the panoptic segmentation files.
        category_index: a dict containing COCO category information keyed
        by the 'id' field of each category.  See the
        label_map_util.create_category_index function.
        reduced_labelid_mapping: mapping to reduced/rearranged label id.
    Returns:
        example: The converted tf.Example
        num_annotations_skipped: Number of (invalid) annotations that were
        ignored.

    Raises:
        ValueError: if the image pointed to by data['filename'] is not a valid
        JPEG
    """
    image_height = image['height']
    image_width = image['width']
    filename_raw = image['file_name']
    image_id = image['id']

    filename = filename_raw.split('.')[0]

    image_data = dataset_util.read_data(
        image_dir, filename, dataset_util.FLAGS.image_format)
    segmentation_data = dataset_util.read_data(
        segmentation_dir, filename, dataset_util.FLAGS.segmentation_format)
    panoptic_data = dataset_util.read_data(
        panoptic_dir, filename, dataset_util.FLAGS.panoptic_format)

    # image = PIL.Image.open(io.BytesIO(image_data))
    # segmentation_image = PIL.Image.open(io.BytesIO(segmentation_data))

    panoptic_image = PIL.Image.open(io.BytesIO(panoptic_data))
    panoptic_ids = dataset_util.convert_rgb_to_ids(np.asarray(panoptic_image))

    key = hashlib.sha256(image_data).hexdigest()

    bbox = dataset_util.BBox()
    is_crowd = []
    category_names = []
    category_ids = []
    object_id_mask = []
    num_annotations_skipped = 0
    for annotations in annotations_list:
        objects_list = annotations['segments_info']
        for objects in objects_list:

            # PANOPTIC DATASET INCLUDES BB FOR EVERYTHING, EVEN THE EXTRA
            # CLASSES USED IN STUFF ANNOTATION
            if objects['category_id'] > 90:
                continue

            # 1. Bounding boxes of objects. In panoptic EACH CLASS LABEL is
            # considered an object. eg: Floor, wall, ceilings have its own BB.
            (x, y, width, height) = tuple(objects['bbox'])
            if width <= 0 or height <= 0:
                num_annotations_skipped += 1
                continue
            if x + width > image_width or y + height > image_height:
                num_annotations_skipped += 1
                continue
            bbox.xmin.append(float(x) / image_width)
            bbox.xmax.append(float(x + width) / image_width)
            bbox.ymin.append(float(y) / image_height)
            bbox.ymax.append(float(y + height) / image_height)
            bbox.area.append(objects['area'])

            # 2. Whether the segment is a group of objects. See (1.)
            is_crowd.append(objects['iscrowd'])

            # 3. Class label ids and names. See (1.)
            category_id = objects['category_id']
            mapped_category_id = int(reduced_labelid_mapping[str(category_id)])
            category_ids.append(mapped_category_id)
            category_names.append(
                category_index[category_id]['name'].encode('utf8'))

            # 4. Mask of individual objects. See (1.)
            object_mask = np.asarray(
                panoptic_ids[:, :] == objects['id']).astype(np.uint8)
            object_mask = PIL.Image.fromarray(object_mask)
            output_io = io.BytesIO()
            object_mask.save(output_io, format='PNG')
            object_id_mask.append(output_io.getvalue())

            # draw = PIL.ImageDraw.Draw(image)
            # draw.rectangle(((x, y), (x + width, y + height)), outline="red")
            # draw.text((x, y), category_index[objects['category_id']]['name'])
            # tmp_image = dataset_util.create_object_mask(np.asarray(image),
            #                                             object_id_mask)
            # PIL.Image.fromarray(object_id_mask.astype(np.uint8)*126).show()
    # image.show()

    example = dataset_util.image_panoptic_to_tf_example(
        image_data, filename, image_height, image_width, 3, segmentation_data,
        bbox, category_ids, category_names,
        is_crowd, mask_data=object_id_mask, image_id=str(image_id),
        key_sha256=key)
    return key, example, num_annotations_skipped


def _create_panoptic_tf_record_from_coco_annotations(
        annotations_file, image_dir, segmentation_dir, panoptic_dir,
        reduced_labelid_mapping_file, output_name):
    """Loads COCO annotation json files and converts to tf.Record format.

    Args:
        annotations_file: JSON file containing bounding box annotations.
        image_dir: Directory containing the image files.
        segmentation_dir: Directory containing the segmentation files.
        panoptic_dir: Directory containing the panoptic segmentation files.
        reduced_labelid_mapping_file: Mapping file to remove the unused labels.
        output_name: Dataset split: train,val,test.
    """
    if output_name == 'test':
        files = [file for file in os.listdir(image_dir) if os.path.isfile(
            os.path.join(image_dir, file))]
        num_images = len(files)
        num_per_shard = int(math.ceil(num_images / float(FLAGS.num_shards)))
        tf.logging.info('writing to output path: %s', FLAGS.output_dir)
        total_num_annotations_skipped = 0
        total_duration = 0.0
        for shard_id in range(FLAGS.num_shards):
            shard_filename = '{:s}-{:05d}-of-{:05d}.tfrecord'.format(
                output_name, shard_id, FLAGS.num_shards)
            output_path = os.path.join(FLAGS.output_dir, shard_filename)
            with tf.python_io.TFRecordWriter(output_path) as tfrecord_writer:
                start_idx = shard_id * num_per_shard
                end_idx = min((shard_id + 1) * num_per_shard, num_images)
                for i in range(start_idx, end_idx):
                    start = time.time()
                    tf_example = create_panoptic_test_tf_example(files[i],
                                                                 image_dir)
                    tfrecord_writer.write(tf_example.SerializeToString())
                    duration = time.time() - start
                    total_duration += duration
                    if i % 100 == 0:
                        tf.logging.info(
                            ' Converted: %s | Progress: %d/%d | Shard: %d | '
                            'Timing: %.2f | TotalTime: %.2f',
                            files[i], i + 1, num_images,
                            shard_id, duration, total_duration)
        tf.logging.info('Finished writing, skipped %d annotations.',
                        total_num_annotations_skipped)
    else:
        with tf.gfile.GFile(annotations_file, 'r') as fid:
            groundtruth_data = json.load(fid)

        with tf.gfile.GFile(reduced_labelid_mapping_file, 'r') as fid:
            reduced_labelid_mapping = json.load(fid)

        images = groundtruth_data['images']
        category_index = label_map_util.create_category_index(
            groundtruth_data['categories'])

        category_index_file = '/home/chen/work/06_tf_models/tf_models/models' \
                              '/research/panoptic/dataset_tools' \
                              '/category_index_coco.json'
        if not tf.gfile.Exists(category_index_file):
            print("The file {:s} does not exist.".format(category_index_file))
            print("Creating {:s} ".format(category_index_file))
            with tf.gfile.GFile(category_index_file, 'w') as fid:
                json.dump(category_index, fid)

        annotations_index = {}
        if 'annotations' in groundtruth_data:
            tf.logging.info('Found groundtruth annotations.')
            tf.logging.info('Building annotations index.')
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
        tf.logging.info('%d images are missing annotations.',
                        missing_annotation_count)

        num_images = len(images)
        num_per_shard = int(math.ceil(num_images / float(FLAGS.num_shards)))

        tf.logging.info('writing to output path: %s', FLAGS.output_dir)
        total_num_annotations_skipped = 0
        for shard_id in range(FLAGS.num_shards):
            shard_filename = '{:s}-{:05d}-of-{:05d}.tfrecord'.format(
                output_name, shard_id, FLAGS.num_shards)
            output_path = os.path.join(FLAGS.output_dir, shard_filename)
            with tf.python_io.TFRecordWriter(output_path) as tfrecord_writer:
                start_idx = shard_id * num_per_shard
                end_idx = min((shard_id + 1) * num_per_shard, num_images)
                for i in range(start_idx, end_idx):
                    sys.stdout.write('\r>> Converting image {}/{} shard {}'.format(
                        i + 1, num_images, shard_id))
                    sys.stdout.flush()
                    annotations_list = annotations_index[images[i]['id']]
                    _, tf_example, num_annotations_skipped = \
                        create_panoptic_tf_example(
                            images[i], annotations_list, image_dir,
                            segmentation_dir, panoptic_dir, category_index,
                            reduced_labelid_mapping)
                    total_num_annotations_skipped += num_annotations_skipped
                    tfrecord_writer.write(tf_example.SerializeToString())
            sys.stdout.write('\n')
            sys.stdout.flush()
        tf.logging.info('Finished writing, skipped %d annotations.',
                        total_num_annotations_skipped)


def main(_):
    assert FLAGS.train_annotations_file, "`train_annotations_file` missing."
    assert FLAGS.val_annotations_file, "`val_annotations_file` missing."
    assert FLAGS.train_image_dir, "`train_image_dir` missing."
    assert FLAGS.val_image_dir, "`val_image_dir` missing."
    assert FLAGS.train_segmentation_dir, "`train_segmentation_dir` missing."
    assert FLAGS.val_segmentation_dir, "`val_segmentation_dir` missing."
    assert FLAGS.train_panoptic_dir, "`train_panoptic_dir` missing."
    assert FLAGS.val_panoptic_dir, "`val_panoptic_dir` missing."
    assert FLAGS.reduced_labelid_mapping_file, "`reduced_labelid_mapping_file` missing."

    assert os.listdir(FLAGS.train_image_dir), "`train_image_dir` is empty."
    assert os.listdir(FLAGS.val_image_dir), "`val_image_dir` is empty."
    assert os.listdir(FLAGS.train_segmentation_dir), "`train_segmentation_dir` is empty."
    assert os.listdir(FLAGS.val_segmentation_dir), "`val_segmentation_dir` is empty."
    assert os.listdir(FLAGS.train_panoptic_dir), "`train_panoptic_dir` is empty."
    assert os.listdir(FLAGS.val_panoptic_dir), "`val_panoptic_dir` is empty."

    if not tf.gfile.IsDirectory(FLAGS.output_dir):
        tf.gfile.MakeDirs(FLAGS.output_dir)
    # else:
    #     print(FLAGS.output_dir + " is not empty.", file=sys.stderr)
    #     exit(1)

    _create_panoptic_tf_record_from_coco_annotations(
        FLAGS.train_annotations_file,
        FLAGS.train_image_dir,
        FLAGS.train_segmentation_dir,
        FLAGS.train_panoptic_dir,
        FLAGS.reduced_labelid_mapping_file,
        'train')
    _create_panoptic_tf_record_from_coco_annotations(
        FLAGS.val_annotations_file,
        FLAGS.val_image_dir,
        FLAGS.val_segmentation_dir,
        FLAGS.val_panoptic_dir,
        FLAGS.reduced_labelid_mapping_file,
        'val')
    # _create_panoptic_tf_record_from_coco_annotations(
    #     FLAGS.test_annotations_file,
    #     FLAGS.test_image_dir,
    #     FLAGS.test_segmentation_dir,
    #     FLAGS.test_panoptic_dir,
    #     FLAGS.reduced_labelid_mapping_file,
    #     'test',
    #     FLAGS.include_masks)


if __name__ == '__main__':
    tf.logging.set_verbosity(tf.logging.INFO)
    tf.app.run()
