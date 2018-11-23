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

"""Library for Faster R-CNN meta-architecture definition. """
import tensorflow as tf

from object_detection import fp_dtype
from object_detection import Options
from object_detection.utils import shape_utils
from object_detection.utils import visualization_utils as vis
from object_detection.core import losses
from object_detection.core import box_list
from object_detection.core import box_list_ops

slim = tf.contrib.slim

DATASET_DICT = {'mapillary': Options.category_index_mapillary,
                'coco': Options.category_index_coco_reduced,
                'cityscapes': Options.category_index_cityscapes}

KL_SCORE_THRESHOLD = 0.75


def draw_results(images,
                 boxes,
                 category_index,
                 classes=None,
                 scores=None,
                 instance_masks=None,
                 min_score_thresh=0.1,
                 max_boxes_to_draw=2000,
                 use_normalized_coordinates=True,
                 name=''):
    label_id_offset = 1  # Applying label id offset (b/63711816)
    if scores is None:
        scores = tf.ones(tf.shape(boxes)[0:2])
    if classes is None:
        classes = tf.cast(tf.zeros(tf.shape(scores)), dtype=tf.uint8)
    else:
        classes += label_id_offset
    image_gts = tf.cast(images, dtype=tf.uint8)
    image_gts = vis.draw_bounding_boxes_on_image_tensors(
        image_gts,
        boxes,
        classes,
        scores,
        category_index,
        instance_masks=instance_masks,
        min_score_thresh=min_score_thresh,
        max_boxes_to_draw=max_boxes_to_draw,
        use_normalized_coordinates=use_normalized_coordinates)
    tf.summary.image(name, image_gts)


def compute_clip_window(image_shapes):
    """Computes clip window for non max suppression based on image shapes.

    This function assumes that the clip window's left top corner is at (0, 0).

    Args:
      image_shapes: A 2-D int32 tensor of shape [batch_size, 3] containing
      shapes of images in the batch. Each row represents [height, width,
      channels] of an image.

    Returns:
      A 2-D float32 tensor of shape [batch_size, 4] containing the clip window
      for each image in the form [ymin, xmin, ymax, xmax].
    """
    clip_heights = image_shapes[:, 0]
    clip_widths = image_shapes[:, 1]
    clip_window = tf.cast(tf.stack([tf.zeros_like(clip_heights),
                                    tf.zeros_like(clip_heights),
                                    clip_heights, clip_widths], axis=1),
                          dtype=tf.float32)
    return clip_window


def padded_batched_proposals_indicator(num_proposals,
                                       max_num_proposals):
    """Creates indicator matrix of non-pad elements of padded batch
    proposals.

    Args:
      num_proposals: Tensor of type tf.int32 with shape [batch_size].
      max_num_proposals: Maximum number of proposals per image (integer).

    Returns:
      A Tensor of type tf.bool with shape [batch_size, max_num_proposals].
    """
    batch_size = tf.size(num_proposals)
    tiled_num_proposals = tf.tile(
        tf.expand_dims(num_proposals, 1), [1, max_num_proposals])
    tiled_proposal_index = tf.tile(
        tf.expand_dims(tf.range(max_num_proposals), 0), [batch_size, 1])
    return tf.greater(tiled_num_proposals, tiled_proposal_index)


def image_batch_shape_2d(image_batch_shape_1d):
    """Takes a 1-D image batch shape tensor and converts it to a 2-D tensor.

    Example:
    If 1-D image batch shape tensor is [2, 300, 300, 3]. The corresponding 2-D
    image batch tensor would be [[300, 300, 3], [300, 300, 3]]

    Args:
      image_batch_shape_1d: 1-D tensor of the form [batch_size, height,
        width, channels].

    Returns:
      image_batch_shape_2d: 2-D tensor of shape [batch_size, 3] were each row is
        of the form [height, width, channels].
    """
    return tf.tile(tf.expand_dims(image_batch_shape_1d[1:], 0),
                   [image_batch_shape_1d[0], 1])


def gather_instance_masks(instance_masks, classes):
    """Gathers the masks that correspond to classes.

    Args:
      instance_masks: A 4-D float32 tensor with shape
        [K, num_classes, mask_height, mask_width].
      classes: A 2-D int32 tensor with shape [batch_size, max_detection].

    Returns:
      masks: a 3-D float32 tensor with shape [K, mask_height, mask_width].
    """
    _, num_classes, height, width = instance_masks.get_shape().as_list()
    k = tf.shape(instance_masks)[0]
    instance_masks = tf.reshape(instance_masks, [-1, height, width])
    classes = tf.to_int32(tf.reshape(classes, [-1]))
    gather_idx = tf.range(k) * num_classes + classes
    return tf.gather(instance_masks, gather_idx)


def flatten_first_two_dimensions(inputs):
    """Flattens `K-d` tensor along batch dimension to be a `(K-1)-d` tensor.

    Converts `inputs` with shape [A, B, ..., depth] into a tensor of shape
    [A * B, ..., depth].

    Args:
      inputs: A float tensor with shape [A, B, ..., depth].  Note that the first
        two and last dimensions must be statically defined.
    Returns:
      A float tensor with shape [A * B, ..., depth] (where the first and last
        dimension are statically defined.
    """
    combined_shape = shape_utils.combined_static_and_dynamic_shape(inputs)
    flattened_shape = tf.stack([combined_shape[0] * combined_shape[1]] +
                               combined_shape[2:])
    return tf.reshape(inputs, flattened_shape)


def apply_boolean_mask(elements, boolean_mask):
    _assert_greater = tf.assert_greater_equal(
        tf.size(elements), tf.size(boolean_mask),
        message="Tensor and boolean mask are of different size...")
    _assert_equal = tf.assert_equal(
        tf.shape(elements)[0], 1,
        message="Batch size for 'apply_boolean_mask' must be one...")
    with tf.control_dependencies([_assert_greater, _assert_equal]):
        return tf.expand_dims(tf.boolean_mask(elements, boolean_mask), axis=0)


def select_box_and_mask(args):
    """ Based on the max_indexes, we select the relevant boxes and masks. The
    max_indexes is the class label (argmax of scores). We multiply the binary
    mask predictions with the class prediction. """
    boxes = args[0]
    scores = args[1]
    masks = args[2]
    max_indexes = args[3]
    selected_boxes = tf.gather(boxes, max_indexes)
    selected_masks = tf.gather(masks, max_indexes)
    selected_masks *= tf.gather(scores, max_indexes)
    return selected_boxes, selected_masks


def resize_semantic_to_mask(args):
    """ We crop out the mask from the semantic prediction and resize it to
    match with the mask dimension. """
    cropped_semantic_logits = args[0]
    selected_masks = args[1]
    offset_height = args[2]
    offset_width = args[3]
    target_height = args[4]
    target_width = args[5]
    cropped_semantic_logits = tf.image.crop_to_bounding_box(
        tf.expand_dims(cropped_semantic_logits, axis=0),
        offset_height, offset_width, target_height, target_width)
    cropped_semantic_logits = tf.image.resize_bilinear(
        cropped_semantic_logits, tf.shape(selected_masks)[0:2])
    return cropped_semantic_logits


def kl_divergence(args):
    """ Calculating the kl divergence loss. All input tensors are per single
    image."""

    semantic_logits = args[0]
    boxes = args[1]
    scores = args[2]
    masks = args[3]
    num_proposals = args[4]
    clip_window = args[5]
    semantic_mask = args[6]  # True/False

    def _empty_kl_divergence():
        return 0.0

    def _kl_divergence_ext(_kl_score_threshold_indicator,
                           _boxes,
                           _scores,
                           _masks,
                           _semantic_logits,
                           _clip_window):
        # 4.3. Take out proposals with scores higher than
        # 'KL_SCORE_THRESHOLD'.
        _boxes = tf.boolean_mask(_boxes, _kl_score_threshold_indicator, axis=0)
        _masks = tf.boolean_mask(_masks, _kl_score_threshold_indicator, axis=0)
        _scores = tf.boolean_mask(_scores, _kl_score_threshold_indicator,
                                  axis=0)
        # 4.4. We obtain the class label for each proposals.
        max_score_indexes = tf.argmax(_scores, axis=1)
        # 4.5. We obtain the semantic class label for each proposals.
        # +1 because label 0 is object background that was removed.
        # We reshuffle the channels to become [prediction, height, width]
        semantic_predictions = tf.nn.softmax(_semantic_logits)
        semantic_predictions = tf.transpose(
            semantic_predictions, [2, 0, 1])
        selected_semantic_predictions = tf.gather(
            semantic_predictions, max_score_indexes + 1)
        # 4.6. We obtain the relevant boxes and masks based on the
        # selected classes.
        elems = (_boxes, _scores, _masks, max_score_indexes)
        output_dtype = (_boxes.dtype, _masks.dtype)
        selected_boxes, selected_masks = tf.map_fn(
            select_box_and_mask, elems=elems,
            dtype=output_dtype)
        # Make the shape compatible for "BoxList()" class. Needed????
        # if tf.shape(selected_boxes)[0]:
        #     selected_boxes = tf.expand_dims(selected_boxes, axis=1)
        # Make the shape compatible for "tf.image" module.
        selected_masks = tf.expand_dims(selected_masks, axis=-1)
        boxlist_clipped = box_list_ops.clip_to_window(
            box_list.BoxList(selected_boxes), _clip_window)
        selected_boxes = tf.cast(boxlist_clipped.get(), dtype=tf.int32)
        offset_height = tf.reshape(
            tf.slice(selected_boxes, [0, 0], [-1, 1]), [-1])
        offset_width = tf.reshape(
            tf.slice(selected_boxes, [0, 1], [-1, 1]), [-1])
        target_height = tf.reshape(
            tf.slice(selected_boxes, [0, 2], [-1, 1]), [-1])
        target_width = tf.reshape(
            tf.slice(selected_boxes, [0, 3], [-1, 1]), [-1])
        target_height = target_height - offset_height
        target_width = target_width - offset_width
        # target_height_width_condition = tf.logical_and(
        #     tf.greater(target_height, 0),
        #     tf.greater(target_width, 0))
        # 4.7. We crop the images out based on the boxes.
        cropped_semantic_predictions = tf.expand_dims(
            selected_semantic_predictions, axis=-1)
        cropped_semantic_predictions = tf.map_fn(
            resize_semantic_to_mask,
            elems=(cropped_semantic_predictions, selected_masks, offset_height,
                   offset_width, target_height, target_width),
            dtype=cropped_semantic_predictions.dtype)
        cropped_semantic_predictions = tf.reshape(
            cropped_semantic_predictions,
            [tf.shape(cropped_semantic_predictions)[0], -1])
        resized_mask = tf.reshape(
            selected_masks, [tf.shape(selected_masks)[0], -1])
        # 4.8. KL Loss calculation.

        def _kl_divergence_with_prob(_args):
            p_prob = _args[0]
            q_prob = _args[1]
            return losses.kl_divergence_with_prob(p_prob, q_prob, None)

        if semantic_mask:
            elems = (cropped_semantic_predictions, resized_mask)
        else:
            elems = (resized_mask, cropped_semantic_predictions)
        _kl_loss = tf.reduce_sum(
            tf.map_fn(_kl_divergence_with_prob, elems=elems, dtype=tf.float32))
        return _kl_loss

    # 4.1. Take out proposals that are valid.
    scores = tf.slice(scores, [0, 0], [num_proposals, -1])
    boxes = tf.slice(boxes, [0, 0, 0], [num_proposals, -1, -1])
    masks = tf.slice(masks, [0, 0, 0, 0], [num_proposals, -1, -1, -1])
    # 4.2. We only consider kl loss for detection with scores higher
    # than 'KL_SCORE_THRESHOLD'.
    per_proposal_max = tf.reduce_max(scores, axis=1)
    kl_score_threshold_indicator = tf.greater(
        per_proposal_max, per_proposal_max * 0.0 + KL_SCORE_THRESHOLD)
    return tf.cond(
        tf.greater(tf.reduce_sum(
            tf.cast(kl_score_threshold_indicator, dtype=tf.int32)), 0),
        lambda: _kl_divergence_ext(kl_score_threshold_indicator, boxes,
                                   scores, masks, semantic_logits, clip_window),
        lambda: _empty_kl_divergence())


def restore_from_classification_checkpoint_fn(scopes_to_ignore=None):
    """Returns a map of variables to load from a foreign checkpoint.

    Args:
      scopes_to_ignore: A list of [scope_name_to_ignore]

    Returns:
      A dict mapping variable names (to load from a checkpoint) to variables in
      the model graph.
    """
    # This is due to distributed training.
    replica_scopes_to_ignore = [
        'replica',
        'replica_1',
        'replica_2',
        'replica_3',
        'replica_4',
        'replica_5',
        'replica_6',
        'replica_7',
    ]
    replica_num_scopes_to_ignore = [
        '_1',
        '_2',
        '_3',
        '_4',
        '_5',
        '_6',
        '_7',
    ]
    variables_to_restore = {}
    for variable in tf.global_variables():
        include_flag = False
        var_name = variable.op.name
        if scopes_to_ignore is not None:
            for scope_name in scopes_to_ignore:
                if scope_name in var_name:
                    var_name = var_name.replace(scope_name + '/', '')
                    include_flag = True
        for scope_name in replica_scopes_to_ignore:
            if scope_name in var_name:
                var_name = var_name.replace('/' + scope_name, '')
                include_flag = True
        if include_flag:
            if var_name[-2:] in replica_num_scopes_to_ignore:
                var_name = var_name[:-2]
            variables_to_restore[var_name] = variable
    return variables_to_restore


def restore_from_checkpoint_fn(scopes_to_ignore=[]):
    """Returns a map of variables to load from a foreign checkpoint.

    Args:
      scopes_to_ignore: A list of [scope_name_to_ignore]

    Returns:
      A dict mapping variable names (to load from a checkpoint) to variables in
      the model graph.
    """
    if not isinstance(scopes_to_ignore, list):
        scopes_to_ignore = [scopes_to_ignore]
    variables_to_restore = {}
    for variable in tf.global_variables():
        include_flag = True
        var_name = variable.op.name
        for scope_name in scopes_to_ignore:
            if scope_name in var_name:
                include_flag = False
        if include_flag:
            variables_to_restore[var_name] = variable
    return variables_to_restore


def add_semantic_image_to_summary(logit_prediction, groundtruth_image,
                                  num_things_class):
    """ Checking the validity of the training labels. Here we draw
    semantic segmentation only.

    Args:
        logit_prediction: A [1, H, W, num_class] tensor.
        groundtruth_image: A [1, H, W, 1] tensor.
        num_things_class: Number of object classes.
    Return:
        An image Tensor.
    """
    # 1. Groundtruth
    summary_label = vis.visualize_segmentation_labels(
        groundtruth_image,
        lower_half=num_things_class)
    tf.summary.image('Things/Original', summary_label)
    summary_label = vis.visualize_segmentation_labels(
        groundtruth_image,
        upper_half=num_things_class - 1)
    tf.summary.image('Stuff/Original', summary_label)
    # 2. Prediction
    prediction = tf.argmax(tf.nn.softmax(logit_prediction), axis=3)
    prediction = tf.image.resize_nearest_neighbor(
        tf.expand_dims(prediction, axis=-1),
        tf.shape(summary_label)[1:3],
        align_corners=True)
    summary_label = vis.visualize_segmentation_labels(
        prediction,
        lower_half=num_things_class)
    tf.summary.image('Things/Prediction', summary_label)
    summary_label = vis.visualize_segmentation_labels(
        prediction,
        upper_half=num_things_class - 1)
    tf.summary.image('Stuff/Prediction', summary_label)
    # 3. Prediction heat map
    prediction_heat_map = tf.reduce_max(tf.nn.softmax(logit_prediction), axis=3)
    prediction_heat_map = tf.image.resize_nearest_neighbor(
        tf.expand_dims(prediction_heat_map, axis=-1),
        tf.shape(summary_label)[1:3],
        align_corners=True)
    prediction_heat_map = tf.cast(
        tf.round(prediction_heat_map * 255.0), dtype=tf.uint8)
    tf.summary.image('zHM/PredictionHeatMap', prediction_heat_map)
    return prediction


def add_cluster_instance_image_to_summary(cluster_prediction,
                                          groundtruth_image):
    """ Checking the validity of the training labels. Here we draw
    instance segmentation only.

    Args:
        cluster_prediction: A [1, H, W, 1] tensor.
        groundtruth_image: A [1, H, W, 1] tensor.
    Return:
        An image Tensor.
    """
    # 1. Groundtruth
    groundtruth_image = tf.cast(groundtruth_image, dtype=tf.uint8)
    tf.summary.image('Instances/Original', tf.cast(groundtruth_image,
                                                   dtype=tf.float32))
    # 2. Prediction
    # cluster_prediction += 1
    summary_label = vis.visualize_segmentation_labels(
        cluster_prediction, upper_half=0)
    tf.summary.image('Instances/Cluster', summary_label)


def add_logit_instance_image_to_summary(logit_prediction, groundtruth_image):
    """ Checking the validity of the training labels. Here we draw
    instance segmentation only.

    Args:
        logit_prediction: A [1, H, W, 1] tensor.
        groundtruth_image: A [1, H, W, 1] tensor.
    Return:
        An image Tensor.
    """
    # 1. Groundtruth
    groundtruth_image = tf.cast(groundtruth_image, dtype=tf.uint8)
    tf.summary.image('Instances/Original',
                     tf.cast(groundtruth_image, dtype=tf.float32))
    # 2. Prediction
    prediction = tf.argmax(tf.nn.softmax(logit_prediction), axis=3)
    prediction = tf.image.resize_nearest_neighbor(
        tf.expand_dims(prediction, axis=-1),
        tf.shape(groundtruth_image)[1:3],
        align_corners=True)
    prediction = tf.cast(prediction, dtype=tf.uint8)
    tf.summary.image('Instances/Prediction', prediction)
    # 3. Heatmap
    prediction_heat_map = tf.reduce_max(tf.nn.softmax(logit_prediction), axis=3)
    prediction_heat_map = tf.image.resize_nearest_neighbor(
        tf.expand_dims(prediction_heat_map, axis=-1),
        tf.shape(groundtruth_image)[1:3],
        align_corners=True)
    prediction_heat_map = tf.cast(
        tf.round(prediction_heat_map * 255.0), dtype=tf.uint8)
    tf.summary.image('zHM/PredictionHeatMap', prediction_heat_map)
    return prediction


def predict_label_logits(features, num_labels, output_dtype=tf.float32,
                         scope_name='logits'):
    """Uses the last featuremap level to predict the labels as logits.

    Note resulting tensors are not postprocessed.

    Args:
        features: A 4-D float32 tensor with shape
            [batch, height, width, depth] to be used for predicting
            semantic labels / scores.
        num_labels: number of labels to predict.
        output_dtype: dtype of output.
        scope_name: scope name

    Returns:
        logits: 3-D float tensor of shape
            [batch, height, width, num_semantic_classes] containing
            semantic class predictions (logits) for each pixels.
    """
    logits = slim.conv2d(features, num_labels, kernel_size=1,
                         activation_fn=None, normalizer_fn=None,
                         scope=scope_name)
    # MixPrecision logits.
    return tf.cast(logits, dtype=output_dtype)
