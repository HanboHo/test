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

r"""Convert raw MAPILLARY dataset to TFRecord for panoptic segmentation."""

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

from numba import vectorize
from panoptic.utils import dataset_util
from panoptic.utils import label_map_util

flags = tf.app.flags
tf.flags.DEFINE_integer('num_shards', 100,
                        'Number of shards for tf.record default: 10.')
tf.flags.DEFINE_string('output_dir',
                       '/mnt/internal/99_Data/08_Mapillary'
                       '/tf_records_test_images',
                       'Output data directory.')
tf.flags.DEFINE_string('train_image_dir',
                       '/mnt/internal/99_Data/08_Mapillary/mapillary-vistas'
                       '-dataset_public_v1.1/training/images',
                       'Training image directory.')
tf.flags.DEFINE_string('train_annotations_file',
                       '/mnt/internal/99_Data/08_Mapillary/mapillary-vistas'
                       '-dataset_public_v1.1/training'
                       '/panoptic/panoptic_2018.json',
                       'Training annotations JSON file.')
tf.flags.DEFINE_string('train_segmentation_dir',
                       '/mnt/internal/99_Data/08_Mapillary/mapillary-vistas'
                       '-dataset_public_v1.1/training/segmentation',
                       'Training panoptic segmentation directory.')
tf.flags.DEFINE_string('train_panoptic_dir',
                       '/mnt/internal/99_Data/08_Mapillary/mapillary-vistas'
                       '-dataset_public_v1.1/training/panoptic',
                       'Training panoptic segmentation directory.')
tf.flags.DEFINE_string('validation_image_dir',
                       '/mnt/internal/99_Data/08_Mapillary/mapillary-vistas'
                       '-dataset_public_v1.1/validation/images',
                       'validationing image directory.')
tf.flags.DEFINE_string('validation_annotations_file',
                       '/mnt/internal/99_Data/08_Mapillary/mapillary-vistas'
                       '-dataset_public_v1.1/validation'
                       '/panoptic/panoptic_2018.json',
                       'validationing annotations JSON file.')
tf.flags.DEFINE_string('validation_segmentation_dir',
                       '/mnt/internal/99_Data/08_Mapillary/mapillary-vistas'
                       '-dataset_public_v1.1/validation/segmentation',
                       'validationing panoptic segmentation directory.')
tf.flags.DEFINE_string('validation_panoptic_dir',
                       '/mnt/internal/99_Data/08_Mapillary/mapillary-vistas'
                       '-dataset_public_v1.1/validation/panoptic',
                       'validationing panoptic segmentation directory.')
tf.flags.DEFINE_string('test_image_dir',
                       '/mnt/internal/99_Data/08_Mapillary/mapillary-vistas'
                       '-dataset_public_v1.1/testing/images',
                       'testing image directory.')
tf.flags.DEFINE_string('test_annotations_file',
                       '',
                       'testing annotations JSON file.')
tf.flags.DEFINE_string('test_segmentation_dir',
                       '',
                       'testing panoptic segmentation directory.')
tf.flags.DEFINE_string('test_panoptic_dir',
                       '',
                       'testing panoptic segmentation directory.')
tf.flags.DEFINE_string('labelid_mapping_file',
                       '/home/chen/work/06_tf_models/tf_models/models'
                       '/research/panoptic/dataset_tools'
                       '/original_to_rearranged_labels.json',
                       'Training panoptic segmentation directory.')
FLAGS = flags.FLAGS

# Taken from dataset_util.py
# A map from image format to expected data format.
_IMAGE_FORMAT_MAP = {
    'jpg': 'jpeg',
    'jpeg': 'jpeg',
    'png': 'png',
}


def _argument_check(split):
    """Checking if the paths are valid and if the folders are empty.

    Args:
        split : train,val,test
    """
    if split == 'train':
        assert FLAGS.train_annotations_file,\
            "`train_annotations_file` missing."
        assert FLAGS.train_image_dir,\
            "`train_image_dir` missing."
        assert FLAGS.train_segmentation_dir,\
            "`train_segmentation_dir` missing."
        assert FLAGS.train_panoptic_dir,\
            "`train_panoptic_dir` missing."
        assert os.listdir(FLAGS.train_image_dir),\
            "`train_image_dir` is empty."
        assert os.listdir(FLAGS.train_segmentation_dir),\
            "`train_segmentation_dir` is empty."
        assert os.listdir(FLAGS.train_panoptic_dir),\
            "`train_panoptic_dir` is empty."
    if split == 'val':
        assert FLAGS.validation_annotations_file,\
            "`validation_annotations_file` missing."
        assert FLAGS.validation_image_dir,\
            "`validation_image_dir` missing."
        assert FLAGS.validation_segmentation_dir,\
            "`validation_segmentation_dir` missing."
        assert FLAGS.validation_panoptic_dir,\
            "`validation_panoptic_dir` missing."
        assert os.listdir(FLAGS.validation_image_dir),\
            "`validation_image_dir` is empty."
        assert os.listdir(FLAGS.validation_segmentation_dir),\
            "`validation_segmentation_dir` is empty."
        assert os.listdir(FLAGS.validation_panoptic_dir),\
            "`validation_panoptic_dir` is empty."
    if split == 'test':
        assert FLAGS.test_image_dir,\
            "`test_image_dir` missing."
        assert os.listdir(FLAGS.test_image_dir),\
            "`test_image_dir` is empty."


def vec_translate(_arr, _dict):
    # for k, v in _arr.items():
    #     _arr[_arr == int(k)] = int(v)
    return np.vectorize(_dict.__getitem__)(_arr)


def create_panoptic_test_tf_example(filename_raw,
                                    image_dir):
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
    image_height = 1024
    image_width = 1024
    image = PIL.Image.open(io.BytesIO(image_data))
    image = image.resize((image_width, image_height), PIL.Image.LANCZOS)
    output_io = io.BytesIO()
    image.save(output_io,
               format=_IMAGE_FORMAT_MAP[dataset_util.FLAGS.image_format])
    image_data = output_io.getvalue()

    example = dataset_util.image_panoptic_test_to_tf_example(
        image_data, filename, image_height, image_width, 3)
    return example


def create_panoptic_tf_example(image,
                               annotations_list,
                               image_dir,
                               segmentation_dir,
                               panoptic_dir,
                               category_index,
                               labelid_mapping):
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
        labelid_mapping: Mapping to remove unused labels.
    Returns:
        example: The converted tf.Example
        num_annotations_skipped: Number of (invalid) annotations that were
        ignored.

    Raises:
        ValueError: if the image pointed to by data['filename'] is not a valid
        JPEG
    """
    filename_raw = image['file_name']
    filename = filename_raw.split('.')[0]
    image_id = image['id']

    # Read data from files.
    image_data = dataset_util.read_data(
        image_dir, filename, dataset_util.FLAGS.image_format)
    segmentation_data = dataset_util.read_data(
        segmentation_dir, filename, dataset_util.FLAGS.segmentation_format)
    panoptic_data = dataset_util.read_data(
        panoptic_dir, filename, dataset_util.FLAGS.panoptic_format)

    # image = PIL.Image.open(io.BytesIO(image_data))
    #
    # segmentation_image = PIL.Image.open(io.BytesIO(segmentation_data))
    # instance_array = np.array(segmentation_image, dtype=np.uint16)
    # # instance_ids_array = np.array(instance_array % 256, dtype=np.uint8)
    # instance_label_array = np.array(instance_array / 256, dtype=np.uint8)
    # instance_label_array += 1
    # instance_label_array[instance_label_array == 66] = 0
    # instance_label_array_mem = np.array(instance_label_array)
    # # tmp = vec_translate(
    # #     instance_label_array_mem, labelid_mapping)
    # for k, v in labelid_mapping.items():
    #     instance_label_array[instance_label_array_mem == int(k)] = int(v)
    # segmentation_image = PIL.Image.fromarray(
    #     np.array(instance_label_array, dtype=np.uint8))
    # output_io = io.BytesIO()
    # segmentation_image.save(output_io, format='PNG')
    # segmentation_data = output_io.getvalue()
    # # PIL.Image.open(io.BytesIO(segmentation_data)).show()
    # # PIL.Image.fromarray(instance_label_array_mem).show()
    #
    # panoptic_image = PIL.Image.open(io.BytesIO(panoptic_data))
    # panoptic_ids = dataset_util.convert_rgb_to_ids(np.asarray(
    #     panoptic_image, dtype=np.uint32))
    #
    # image_height = image.height
    # image_width = image.width

    scale = 3

    # 1. Processing image
    image = PIL.Image.open(io.BytesIO(image_data))
    rescaled_image_height = image.height // scale
    rescaled_image_width = image.width // scale
    image = image.resize(
        (rescaled_image_width, rescaled_image_height), PIL.Image.LANCZOS)
    output_io = io.BytesIO()
    image.save(output_io,
               format=_IMAGE_FORMAT_MAP[dataset_util.FLAGS.image_format])
    image_data = output_io.getvalue()
    # 2. Processing segmentation
    segmentation_image = PIL.Image.open(io.BytesIO(segmentation_data))
    segmentation_image = segmentation_image.resize(
        (rescaled_image_width, rescaled_image_height), PIL.Image.NEAREST)
    output_io = io.BytesIO()
    segmentation_image.save(output_io,
                            format=dataset_util.FLAGS.segmentation_format)
    segmentation_data = output_io.getvalue()
    # 3. Processing panoptic
    panoptic_image = PIL.Image.open(io.BytesIO(panoptic_data))
    panoptic_image = panoptic_image.resize(
        (rescaled_image_width, rescaled_image_height), PIL.Image.NEAREST)
    panoptic_ids = dataset_util.convert_rgb_to_ids(np.asarray(
        panoptic_image, dtype=np.uint32))

    key = hashlib.sha256(image_data).hexdigest()

    image_height = rescaled_image_height
    image_width = rescaled_image_width

    # Saving the BB.
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
            if category_index[objects['category_id']]['isthing'] == 0:
                continue

            # 1. Bounding boxes of objects. In panoptic EACH CLASS LABEL is
            # considered an object. eg: Floor, wall, ceilings have its own BB.
            (x, y, width, height) = tuple(objects['bbox'])
            (x, y, width, height) = (x / scale, y / scale,
                                     width / scale, height / scale)
            if width <= 0 or height <= 0:
                num_annotations_skipped += 1
                continue
            if x + width > float(image_width) or \
                    y + height > float(image_height):
                if np.floor(x + width) > float(image_width) or \
                        np.floor(y + height) > float(image_height):
                    num_annotations_skipped += 1
                    continue
                else:
                    # Due to image_width, image_height being of type int,
                    # they are 'floored' and therefore will cause annotations
                    # near to borders to be skipped.
                    width = float(image_width) - x
                    height = float(image_height) - y
            bbox.xmin.append(float(x) / image_width)
            bbox.xmax.append(float(x + width) / image_width)
            bbox.ymin.append(float(y) / image_height)
            bbox.ymax.append(float(y + height) / image_height)
            bbox.area.append(objects['area'])

            # 2. Whether the segment is a group of objects. See (1.)
            is_crowd.append(objects['iscrowd'])

            # 3. Class label ids and names. See (1.)
            category_id = objects['category_id']
            mapped_category_id = labelid_mapping[category_id]
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

            # # TODO: Vizualisation for viewing.
            # draw = PIL.ImageDraw.Draw(image)
            # draw.rectangle(((x, y), (x + width, y + height)), outline="red")
            # draw.rectangle(((bbox.xmin[-1] * image_width,
            #                  bbox.ymin[-1] * image_height),
            #                 (bbox.xmax[-1] * image_width,
            #                  bbox.ymax[-1] * image_height)),
            #                outline="red")
            # draw.text((x, y), category_index[objects['category_id']]['name'])
            # tmp_image = dataset_util.create_object_mask(np.asarray(image),
            #                                             object_id_mask)
            # PIL.Image.fromarray(object_mask.astype(np.uint8)*126).show()
    # image.show()

    example = dataset_util.image_panoptic_to_tf_example(
        image_data, filename, image_height, image_width, 3,
        segmentation_data, bbox, category_ids, category_names, is_crowd,
        mask_data=object_id_mask, image_id=str(image_id), key_sha256=key)
    return key, example, num_annotations_skipped


def _create_panoptic_tf_record_from_annotations(annotations_file,
                                                image_dir,
                                                segmentation_dir,
                                                panoptic_dir,
                                                output_name,
                                                labelid_mapping_file):
    """Loads Mapillary annotation json files and converts to tf.Record format.

    Args:
        annotations_file: JSON file containing bounding box annotations.
        image_dir: Directory containing the image files.
        segmentation_dir: Directory containing the segmentation files.
        panoptic_dir: Directory containing the panoptic segmentation files.
        output_name: Dataset split: train,val,test.
        labelid_mapping_file: Remapping original labels to a new arrangement.
    """
    _argument_check(output_name)

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

        with tf.gfile.GFile(labelid_mapping_file, 'r') as fid:
            remapping_ori = json.load(fid)

        remapping = {0: 0}
        for k, v in remapping_ori.items():
            remapping[int(k)] = v

        images = groundtruth_data['images']
        category_index = label_map_util.create_category_index(
            groundtruth_data['categories'])

        category_index_file = '/home/chen/work/06_tf_models/tf_models/models' \
                              '/research/panoptic/dataset_tools' \
                              '/category_index_mapillary.json'
        if not tf.gfile.Exists(category_index_file):
            print("The file {:s} does not exist.".format(category_index_file))
            print("Creating {:s} ".format(category_index_file))
            with tf.gfile.GFile(category_index_file, 'w') as fid:
                json.dump(category_index, fid)

        category_index_file = '/home/chen/work/06_tf_models/tf_models/models' \
                              '/research/panoptic/dataset_tools' \
                              '/panoptic_mapillary_category.json'
        if not tf.gfile.Exists(category_index_file):
            print("The file {:s} does not exist.".format(category_index_file))
            print("Creating {:s} ".format(category_index_file))
            category_index_list = []
            for k, v in category_index.items():
                category_index_list.append(v)
            with tf.gfile.GFile(category_index_file, 'w') as fid:
                json.dump(category_index_list, fid)

        category_index_file = '/home/chen/work/06_tf_models/tf_models/models' \
                              '/research/panoptic/dataset_tools' \
                              '/category_index_mapillary.json'
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
                    annotations_list = annotations_index[images[i]['id']]
                    _, tf_example, num_annotations_skipped = \
                        create_panoptic_tf_example(
                            images[i], annotations_list, image_dir,
                            segmentation_dir, panoptic_dir, category_index,
                            remapping)
                    total_num_annotations_skipped += num_annotations_skipped
                    tfrecord_writer.write(tf_example.SerializeToString())
                    duration = time.time() - start
                    total_duration += duration
                    if i % 100 == 0:
                        tf.logging.info(
                            ' Converted: %s | Progress: %d/%d | Shard: %d | '
                            'SkippedAnnotation: %d | Timing: %.2f | '
                            'TotalTime: %.2f',
                            images[i]['file_name'], i + 1, num_images,
                            shard_id, total_num_annotations_skipped, duration,
                            total_duration)
        tf.logging.info('Finished writing, skipped %d annotations.',
                        total_num_annotations_skipped)


def main(_):

    assert FLAGS.labelid_mapping_file, "`labelid_mapping_file` missing."

    if not tf.gfile.IsDirectory(FLAGS.output_dir):
        tf.gfile.MakeDirs(FLAGS.output_dir)
    else:
        print(FLAGS.output_dir + " is not empty.", file=sys.stderr)
        exit(1)

    # tf.logging.info('Starting tfrecord creation for train split.')
    # _create_panoptic_tf_record_from_annotations(
    #     FLAGS.train_annotations_file,
    #     FLAGS.train_image_dir,
    #     FLAGS.train_segmentation_dir,
    #     FLAGS.train_panoptic_dir,
    #     'train',
    #     FLAGS.labelid_mapping_file)
    #
    # tf.logging.info('Starting tfrecord creation for val split.')
    # _create_panoptic_tf_record_from_annotations(
    #     FLAGS.validation_annotations_file,
    #     FLAGS.validation_image_dir,
    #     FLAGS.validation_segmentation_dir,
    #     FLAGS.validation_panoptic_dir,
    #     'val',
    #     FLAGS.labelid_mapping_file)

    tf.logging.info('Starting tfrecord creation for test split.')
    _create_panoptic_tf_record_from_annotations(
        FLAGS.test_annotations_file,
        FLAGS.test_image_dir,
        FLAGS.test_segmentation_dir,
        FLAGS.test_panoptic_dir,
        'test',
        FLAGS.labelid_mapping_file)


if __name__ == '__main__':
    tf.logging.set_verbosity(tf.logging.INFO)
    tf.app.run()
