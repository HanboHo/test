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

from panoptic.utils import dataset_util
from panoptic.utils import label_map_util
from object_detection.dataset_tools.labels import name2label

flags = tf.app.flags
tf.flags.DEFINE_integer('num_shards', 100,
                        'Number of shards for tf.record default: 10.')
tf.flags.DEFINE_string('output_dir',
                       '/mnt/internal/99_Data/01_Cityscapes/tf_records_full',
                       'Output data directory.')

tf.flags.DEFINE_string('train_image_dir',
                       '/mnt/internal/99_Data/01_Cityscapes'
                       '/leftImg8bit_trainvaltest/leftImg8bit/train',
                       'Training image directory.')
tf.flags.DEFINE_string('train_annotations_file',
                       '/mnt/internal/99_Data/01_Cityscapes/cityscapesScripts'
                       '/cityscapesscripts/helpers/labels.py',
                       'cityscapesScripts labels.py')
tf.flags.DEFINE_string('train_segmentation_dir',
                       '/mnt/internal/99_Data/01_Cityscapes'
                       '/gtFine_trainvaltest/gtFine/train',
                       'Training segmentation directory.')
tf.flags.DEFINE_string('train_panoptic_dir',
                       '/mnt/internal/99_Data/01_Cityscapes'
                       '/gtFine_trainvaltest/gtFine/train',
                       'Training panoptic segmentation directory.')

tf.flags.DEFINE_string('validation_image_dir',
                       '/mnt/internal/99_Data/01_Cityscapes'
                       '/leftImg8bit_trainvaltest/leftImg8bit/val',
                       'validationing image directory.')
tf.flags.DEFINE_string('validation_annotations_file',
                       '/mnt/internal/99_Data/01_Cityscapes/cityscapesScripts'
                       '/cityscapesscripts/helpers/labels.py',
                       'validationing annotations JSON file.')
tf.flags.DEFINE_string('validation_segmentation_dir',
                       '/mnt/internal/99_Data/01_Cityscapes'
                       '/gtFine_trainvaltest/gtFine/val',
                       'validationing panoptic segmentation directory.')
tf.flags.DEFINE_string('validation_panoptic_dir',
                       '/mnt/internal/99_Data/01_Cityscapes'
                       '/gtFine_trainvaltest/gtFine/val',
                       'validationing panoptic segmentation directory.')

# tf.flags.DEFINE_string('test_image_dir',
#                        '/mnt/internal/99_Data/01_Cityscapes/mapillary-vistas'
#                        '-dataset_public_v1.1/testing/images',
#                        'testing image directory.')
# tf.flags.DEFINE_string('test_annotations_file',
#                        '',
#                        'testing annotations JSON file.')
# tf.flags.DEFINE_string('test_segmentation_dir',
#                        '',
#                        'testing panoptic segmentation directory.')
# tf.flags.DEFINE_string('test_panoptic_dir',
#                        '',
#                        'testing panoptic segmentation directory.')
# tf.flags.DEFINE_string('labelid_mapping_file',
#                        '/home/chen/work/06_tf_models/tf_models/models'
#                        '/research/panoptic/dataset_tools'
#                        '/original_to_rearranged_labels.json',
#                        'Training panoptic segmentation directory.')
FLAGS = flags.FLAGS

# Taken from dataset_util.py
# A map from image format to expected data format.
_IMAGE_FORMAT_MAP = {
    'jpg': 'jpeg',
    'jpeg': 'jpeg',
    'png': 'png',
}

MAPPING = [i for i in reversed(range(20))]


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
                               semantics,
                               instances,
                               category_index):
    """Converts image and annotations to a tf.Example proto.

    Args:
        image: dict with keys:
            [u'license', u'file_name', u'coco_url', u'height', u'width',
            u'date_captured', u'flickr_url', u'id']
        Notice that bounding box coordinates in the official COCO dataset are
        given as [x, y, width, height] tuples using absolute coordinates where
        x, y represent the top-left (0-indexed) corner.  This function converts
        to the format expected by the Tensorflow Object Detection API (which is
        which is [ymin, xmin, ymax, xmax] with coordinates normalized relative
        to image size).
        segmentation_dir: directory containing the segmentation files.
        panoptic_dir: directory containing the panoptic segmentation files.
        category_index: a dict containing COCO category information keyed
        by the 'id' field of each category.  See the
        label_map_util.create_category_index function.
    Returns:
        example: The converted tf.Example
        num_annotations_skipped: Number of (invalid) annotations that were
        ignored.

    Raises:
        ValueError: if the image pointed to by data['filename'] is not a valid
        JPEG
    """
    # Read data from files.
    filename_raw = semantics.split('/')[-1]
    filename = filename_raw.split('.')[0]
    semantic_id = filename.replace('_gtFine_labelTrainIds', '')
    semantic_data = dataset_util.read_data('', semantics)
    semantic_format = filename_raw.split('.')[-1]

    filename_raw = instances.split('/')[-1]
    filename = filename_raw.split('.')[0]
    instances_id = filename.replace('_gtFine_instanceTrainIds', '')
    instances_data = dataset_util.read_data('', instances)
    instances_format = filename_raw.split('.')[-1]

    filename_raw = image.split('/')[-1]
    filename = filename_raw.split('.')[0]
    image_id = filename.replace('_leftImg8bit', '')
    image_data = dataset_util.read_data('', image)
    image_format = filename_raw.split('.')[-1]

    assert image_id == semantic_id, "image_id != semantic_id"
    assert image_id == instances_id, "image_id != instances_id"

    scale = 1

    # 1. Processing image
    image = PIL.Image.open(io.BytesIO(image_data))
    rescaled_image_height = image.height // scale
    rescaled_image_width = image.width // scale
    image = image.resize(
        (rescaled_image_width, rescaled_image_height), PIL.Image.LANCZOS)
    output_io = io.BytesIO()
    image.save(output_io, format=_IMAGE_FORMAT_MAP[image_format])
    image_data = output_io.getvalue()

    # 2. Processing segmentation
    segmentation_image = PIL.Image.open(io.BytesIO(semantic_data))
    segmentation_image = segmentation_image.resize(
        (rescaled_image_width, rescaled_image_height), PIL.Image.NEAREST)
    segmentation_image_tmp = segmentation_image.copy()
    segmentation_image = np.asarray(segmentation_image)
    segmentation_image_tmp = np.asarray(segmentation_image_tmp)
    assert segmentation_image_tmp.min() >= 0,\
        'Min label is not 0 but {}'.format(segmentation_image_tmp.min())
    assert segmentation_image_tmp.max() <= 19,\
        'Max label is not 19 but {}'.format(segmentation_image_tmp.max())
    segmentation_image.setflags(write=True)
    for idx, val in enumerate(MAPPING):
        segmentation_image[segmentation_image_tmp == idx] = val
    segmentation_image = PIL.Image.fromarray(segmentation_image)
    output_io = io.BytesIO()
    segmentation_image.save(output_io, format=semantic_format)
    segmentation_data = output_io.getvalue()

    # 3. Processing panoptic
    panoptic_image = PIL.Image.open(io.BytesIO(instances_data))
    panoptic_image = panoptic_image.resize(
        (rescaled_image_width, rescaled_image_height), PIL.Image.NEAREST)
    panoptic_ids = np.asarray(panoptic_image, dtype=np.uint32)

    key = hashlib.sha256(image_data).hexdigest()

    image_height = rescaled_image_height
    image_width = rescaled_image_width

    instance_ids = np.unique(panoptic_ids)
    instance_ids = instance_ids[instance_ids > 1000]

    # Saving the BB.
    bbox = dataset_util.BBox()
    is_crowd = []
    category_names = []
    category_ids = []
    object_id_mask = []
    num_annotations_skipped = 0
    for instance_id in instance_ids:
        # 3. Class label ids and names. See (1.)
        category_id = MAPPING[instance_id // 1000]
        if category_id == 0:
            continue
        category_ids.append(category_id)
        category_names.append(
            category_index[str(category_id)]['name'].encode('utf8'))

        # 1. Bounding boxes of objects.
        instance_id_img = panoptic_ids == instance_id
        rows = np.any(instance_id_img, axis=1)
        cols = np.any(instance_id_img, axis=0)
        rows = np.where(rows)[0]
        cols = np.where(cols)[0]
        rmin, rmax = rows[0], rows[-1]
        cmin, cmax = cols[0], cols[-1]
        (x, y, width, height) = (cmin, rmin, cmax-cmin, rmax-rmin)
        if width == 0:
            x = x - 1 if x == image_width - 1 else x
            width += 1
        if height == 0:
            y = y - 1 if y == image_height - 1 else y
            height += 1
        if width < 0 or height < 0:
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
        bbox.area.append(float(width*height))

        # 4. Mask of individual objects. See (1.)
        object_mask = np.asarray(
            panoptic_ids[:, :] == instance_id).astype(np.uint8)
        object_mask = PIL.Image.fromarray(object_mask)
        output_io = io.BytesIO()
        object_mask.save(output_io, format='PNG')
        object_id_mask.append(output_io.getvalue())

    #     # TODO: Vizualisation for viewing.
    #     draw = PIL.ImageDraw.Draw(image)
    #     draw.rectangle(((x, y), (x + width, y + height)), outline="red")
    #     draw.rectangle(((bbox.xmin[-1] * image_width,
    #                      bbox.ymin[-1] * image_height),
    #                     (bbox.xmax[-1] * image_width,
    #                      bbox.ymax[-1] * image_height)),
    #                    outline="red")
    #     draw.text((x, y), category_index[str(category_id)]['name'])
    #     # tmp_image = dataset_util.create_object_mask(np.asarray(image),
    #     #                                             object_id_mask)
    #     # PIL.Image.fromarray(object_mask.astype(np.uint8)*126).show()
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
                                                output_name):
    """Loads Mapillary annotation json files and converts to tf.Record format.

    Args:
        annotations_file: JSON file containing bounding box annotations.
        image_dir: Directory containing the image files.
        segmentation_dir: Directory containing the segmentation files.
        panoptic_dir: Directory containing the panoptic segmentation files.
        output_name: Dataset split: train,val,test.
    """
    # _argument_check(output_name)

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

        category_index = dict()
        for key, val in name2label.items():
            if int(MAPPING[val.trainId]) == 0 or int(val.trainId) == -1:
                category_index[str(0)] = {
                    "supercategory": "void",
                    "isthing": 0,
                    "id": 0,
                    "name": "void"
                }
            else:
                category_index[str(MAPPING[val.trainId])] = {
                    "supercategory": val.category,
                    "isthing": int(val.hasInstances),
                    "id": int(MAPPING[val.trainId]),
                    "name": val.name
                }

        category_index_file = '/home/chen/work/06_tf_models/tf_models/models' \
                              '/research/object_detection/dataset_tools' \
                              '/category_index_cityscapes.json'
        if not tf.gfile.Exists(category_index_file):
            print("The file {:s} does not exist.".format(category_index_file))
            print("Creating {:s} ".format(category_index_file))
            with tf.gfile.GFile(category_index_file, 'w') as fid:
                json.dump(category_index, fid)

        category_index_file = '/home/chen/work/06_tf_models/tf_models/models' \
                              '/research/object_detection/dataset_tools' \
                              '/panoptic_cityscapes_categories.json'
        if not tf.gfile.Exists(category_index_file):
            print("The file {:s} does not exist.".format(category_index_file))
            print("Creating {:s} ".format(category_index_file))
            category_index_list = []
            for k, v in category_index.items():
                category_index_list.append(v)
            with tf.gfile.GFile(category_index_file, 'w') as fid:
                json.dump(category_index_list, fid)

        instances = []
        instances_folder = [
            os.path.join(panoptic_dir, f) for f in os.listdir(panoptic_dir)
            if os.path.isdir(os.path.join(panoptic_dir, f))]
        for fol in instances_folder:
            instances += [
                os.path.join(fol, f) for f in os.listdir(fol)
                if os.path.isfile(os.path.join(fol, f)) and
                   'instanceTrainIds' in os.path.join(fol, f)]
        instances = sorted(instances)
        semantic = []
        semantic_folder = [
            os.path.join(segmentation_dir, f) for f in os.listdir(segmentation_dir)
            if os.path.isdir(os.path.join(segmentation_dir, f))]
        for fol in semantic_folder:
            semantic += [
                os.path.join(fol, f) for f in os.listdir(fol)
                if os.path.isfile(os.path.join(fol, f)) and
                   'labelTrainIds' in os.path.join(fol, f)]
        semantic = sorted(semantic)
        images = []
        images_folder = [
            os.path.join(image_dir, f) for f in os.listdir(image_dir)
            if os.path.isdir(os.path.join(image_dir, f))]
        for fol in images_folder:
            images += [
                os.path.join(fol, f) for f in os.listdir(fol)
                if os.path.isfile(os.path.join(fol, f))]
        images = sorted(images)
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
                    # annotations_list = annotations_index[images[i]['id']]
                    _, tf_example, num_annotations_skipped = \
                        create_panoptic_tf_example(
                            images[i], semantic[i], instances[i],
                            category_index)
                    total_num_annotations_skipped += num_annotations_skipped
                    tfrecord_writer.write(tf_example.SerializeToString())
                    duration = time.time() - start
                    total_duration += duration
                    if i % 100 == 0:
                        tf.logging.info(
                            ' Converted: %s | Progress: %d/%d | Shard: %d | '
                            'SkippedAnnotation: %d | Timing: %.2f | '
                            'TotalTime: %.2f',
                            images[i], i + 1, num_images,
                            shard_id, total_num_annotations_skipped, duration,
                            total_duration)
        tf.logging.info('Finished writing, skipped %d annotations.',
                        total_num_annotations_skipped)


def main(_):

    # assert FLAGS.labelid_mapping_file, "`labelid_mapping_file` missing."

    if not tf.gfile.IsDirectory(FLAGS.output_dir):
        tf.gfile.MakeDirs(FLAGS.output_dir)
    # else:
    #     print(FLAGS.output_dir + " is not empty.", file=sys.stderr)
    #     exit(1)

    tf.logging.info('Starting tfrecord creation for train split.')
    _create_panoptic_tf_record_from_annotations(
        FLAGS.train_annotations_file,
        FLAGS.train_image_dir,
        FLAGS.train_segmentation_dir,
        FLAGS.train_panoptic_dir,
        'train')

    tf.logging.info('Starting tfrecord creation for val split.')
    _create_panoptic_tf_record_from_annotations(
        FLAGS.validation_annotations_file,
        FLAGS.validation_image_dir,
        FLAGS.validation_segmentation_dir,
        FLAGS.validation_panoptic_dir,
        'val')

    # tf.logging.info('Starting tfrecord creation for test split.')
    # _create_panoptic_tf_record_from_annotations(
    #     FLAGS.test_annotations_file,
    #     FLAGS.test_image_dir,
    #     FLAGS.test_segmentation_dir,
    #     FLAGS.test_panoptic_dir,
    #     'test',
    #     FLAGS.labelid_mapping_file)


if __name__ == '__main__':
    tf.logging.set_verbosity(tf.logging.INFO)
    tf.app.run()
