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
"""Multi-Task meta-architecture based on Faster R-CNN meta-architecture  """

from abc import abstractmethod
import tensorflow as tf

from object_detection import fp_dtype
from object_detection.core import box_list
from object_detection.core import box_list_ops
from object_detection.core import clustering
from object_detection.core import losses
from object_detection.core import model
from object_detection.core import standard_fields as fields
from object_detection.core import target_assigner
from object_detection.utils import shape_utils
from object_detection.utils import ops
from object_detection.meta_architectures import meta_arch_lib as lib
from object_detection.core import layers

# For debugging purposes
# from nets import resnet_utils
# from nets import resnet_v1

slim = tf.contrib.slim

BANDWIDTH = 2.0
EPSILON = 1e-8
MAX_NUM_INSTANCES = [8]
IOU_THRESHOLD_INSTANCES = 0.5
PARAM_VAR = 1.0
PARAM_DIST = 1.0
PARAM_REG = 0.001
DELTA_V = 0.5
DELTA_D = 1.5
CLUSTERING_PYTHON = False


class MTFeatureExtractor(object):
  """Faster R-CNN Feature Extractor definition."""

  def __init__(self,
               is_training,
               first_stage_features_stride,
               batch_norm_trainable=False,
               batch_norm_param=None,
               weight_decay=0.0):
    """Constructor.

    Args:
      is_training: A boolean indicating whether the training version of the
        computation graph should be constructed.
      first_stage_features_stride: Output stride of the feature extractor.
      batch_norm_trainable: Whether to update batch norm parameters during
        training or not. When training with a relative large batch size
        (e.g. 8), it could be desirable to enable batch norm update.
      batch_norm_param: a dict of batch norm parameters.
      weight_decay: float weight decay for feature extractor (default: 0.0).
    """
    self._is_training = is_training
    self._train_batch_norm = batch_norm_trainable and is_training
    self._batch_norm_param = batch_norm_param
    self._weight_decay = weight_decay
    self._first_stage_features_stride = first_stage_features_stride

  @abstractmethod
  def preprocess(self, resized_inputs):
    """Feature-extractor specific preprocessing (minus image resizing)."""
    pass

  @abstractmethod
  def _extract_decoder_features(self, preprocessed_inputs, scope,
                                concat_levels, residual_depth):
    """Extracts first stage decoder features, to be overridden."""
    pass

  def extract_decoder_features(self, preprocessed_inputs, scope,
                               concat_levels, residual_depth):
    """Extracts first stage decoder features based on FPN.

    Args:
      preprocessed_inputs: A [batch, height, width, channels] float tensor
        representing a batch of images.
      scope: A scope name.
      concat_levels: either concat or add the features from different levels.
      residual_depth: depth of the residual conv connection.

    Returns:
      decoder_feature_map: A tensor with shape [batch, height, width, depth]
      activations: A dictionary mapping activation tensor names to tensors.
    """
    # with tf.variable_scope(scope,
    # values=[preprocessed_inputs, concat_levels]):
    return self._extract_decoder_features(preprocessed_inputs, scope,
                                          concat_levels, residual_depth)

  @abstractmethod
  def _extract_image_features(self, preprocessed_inputs, scope):
    """Extracts first stage image features, to be overridden."""
    pass

  def extract_image_features(self, preprocessed_inputs, scope):
    """Extracts first stage image features based on FPN.

    Args:
      preprocessed_inputs: A [batch, height, width, channels] float tensor
        representing a batch of images.
      scope: A scope name.

    Returns:
      decoder_feature_map: A tensor with shape [batch, height, width, depth]
      activations: A dictionary mapping activation tensor names to tensors.
    """
    # with tf.variable_scope(scope, values=[preprocessed_inputs]):
    return self._extract_image_features(preprocessed_inputs, scope)

  @staticmethod
  def restore_from_classification_checkpoint_fn(scopes_to_ignore=None):
    """Returns a map of variables to load from a foreign checkpoint.

    Returns:
      A dict mapping variable names (to load from a checkpoint) to variables in
      the model graph.
    """
    return lib.restore_from_classification_checkpoint_fn(scopes_to_ignore)


class MTMetaArch(model.DetectionModel):
  """Faster R-CNN Meta-architecture definition."""

  def __init__(self,
               is_training,
               is_evaluating,
               num_classes,
               image_resizer_fn,
               feature_extractor,
               instance_segmentation=False,
               parallel_iterations=16,
               add_summaries=True,
               use_static_shapes=False,
               resize_masks=True,
               num_semantic_classes=0,
               aspp_arg_scope_fn=None,
               refinement_arg_scope_fn=None,
               semantic_loss_weight=1.0,
               instance_loss_weight=1.0,
               dataset_name=None):
    """MTMetaArch Constructor.

    Args:
      is_training: A boolean indicating whether the training version of the
        computation graph should be constructed.
      is_evaluating: A boolean indicating whether the validation version of the
        computation graph should be constructed.
      num_classes: Number of classes.  Note that num_classes *does not*
        include the background category, so if groundtruth labels take values
        in {0, 1, .., K-1}, num_classes=K (and not K+1, even though the
        assigned classification targets can range from {0,... K}).
      image_resizer_fn: A callable for image resizing.  This callable
        takes a rank-3 image tensor of shape [height, width, channels]
        (corresponding to a single image), an optional rank-3 instance mask
        tensor of shape [num_masks, height, width] and returns a resized rank-3
        image tensor, a resized mask tensor if one was provided in the input. In
        addition this callable must also return a 1-D tensor of the form
        [height, width, channels] containing the size of the true image, as the
        image resizer can perform zero padding. See protos/image_resizer.proto.
      feature_extractor: A MTFeatureExtractor object.
      parallel_iterations: (Optional) The number of iterations allowed to run
        in parallel for calls to tf.map_fn.
      add_summaries: boolean (default: True) controlling whether summary ops
        should be added to tensorflow graph.
      use_static_shapes: If True, uses implementation of ops with static shape
        guarantees.
      resize_masks: Indicates whether the masks presend in the groundtruth
        should be resized in the model with `image_resizer_fn`

    Raises:
      ValueError: If `second_stage_batch_size` > `first_stage_max_proposals` at
        training time.
      ValueError: If first_stage_anchor_generator is not of type
        grid_anchor_generator.GridAnchorGenerator.
    """
    # TODO(rathodv): add_summaries is currently unused. Respect that directive
    # in the future.
    super(MTMetaArch, self).__init__(
        num_classes=num_classes,
        num_semantic_classes=num_semantic_classes)

    if (dataset_name is None) or (dataset_name not in
                                  lib.DATASET_DICT.keys()):
        raise ValueError(
            'dataset_name is empty or does not exist, we need to use a '
            'known dataset category index dict.')
    self._category_index = lib.DATASET_DICT[dataset_name]

    self._instance_segmentation = instance_segmentation

    self._is_training = is_training
    self._is_evaluating = is_evaluating
    self._is_predicting = is_training is False and is_evaluating is False

    self._image_resizer_fn = image_resizer_fn
    self._resize_masks = resize_masks
    self._feature_extractor = feature_extractor

    self._aspp_arg_scope_fn = aspp_arg_scope_fn
    self._refinement_arg_scope_fn = refinement_arg_scope_fn
    self._semantic_loss_weight = semantic_loss_weight
    self._instance_loss_weight = instance_loss_weight

    self._use_static_shapes = use_static_shapes

    self._second_stage_mask_loss = (
        losses.WeightedSigmoidClassificationLoss())

    self._parallel_iterations = parallel_iterations

    self._add_summaries = add_summaries

    # TODO: Need to be in config proto file.
    # _panoptic_score_threshold = threshold to consider a prediction valid.
    # _panoptic_mask_threshold = threshold to create binary mask.
    self._panoptic_score_threshold = 0.7
    self._panoptic_mask_threshold = 0.5

  @property
  def image_feature_extractor_scope(self):
    return 'ImageFeatureExtractor'

  @property
  def decoder_feature_extractor_scope(self):
    return 'DecoderFeatureExtractor'

  def preprocess(self, inputs):
    """Feature-extractor specific preprocessing.

    See base class.

    For Faster R-CNN, we perform image resizing in the base class --- each
    class subclassing MTMetaArch is responsible for any additional
    preprocessing (e.g., scaling pixel values to be in [-1, 1]).

    Args:
      inputs: a [batch, height_in, width_in, channels] float tensor representing
        a batch of images with values between 0 and 255.0.

    Returns:
      preprocessed_inputs: a [batch, height_out, width_out, channels] float
        tensor representing a batch of images.
      true_image_shapes: int32 tensor of shape [batch, 3] where each row is
        of the form [height, width, channels] indicating the shapes
        of true images in the resized images, as resized images can be padded
        with zeros.
    Raises:
      ValueError: if inputs tensor does not have type tf.float32
    """
    if inputs.dtype is not tf.float16 and inputs.dtype is not tf.float32:
      tf.logging.info("`preprocess(...)` expects a float16/32 tensor for "
                      "inputs and not `{}` ".format(inputs.dtype))
      tf.logging.info("inputs of type `{}` will be cast to "
                      "float32...".format(inputs.dtype))
      inputs = tf.cast(inputs, dtype=tf.float32)
    with tf.variable_scope('Preprocessor'):
      outputs = shape_utils.static_or_dynamic_map_fn(
          self._image_resizer_fn,
          elems=inputs,
          dtype=[inputs.dtype, tf.int32],
          parallel_iterations=self._parallel_iterations)
      resized_inputs = outputs[0]
      true_image_shapes = outputs[1]
      return (self._feature_extractor.preprocess(resized_inputs),
              true_image_shapes)

  def predict(self, preprocessed_inputs, true_image_shapes, **params):
    """Predicts unpostprocessed tensors from input tensor.

    Args:
      preprocessed_inputs: a [batch, height, width, channels] float tensor
        representing a batch of images.
      true_image_shapes: int32 tensor of shape [batch, 3] where each row is
        of the form [height, width, channels] indicating the shapes
        of true images in the resized images, as resized images can be padded
        with zeros.

    Returns:
      prediction_dict: a dictionary holding "raw" prediction tensors:
        3) image_shape: a 1-D tensor of shape [4] representing the input
          image shape.
        11) mask_predictions: (optional) a 4-D tensor with shape
          [total_num_padded_proposals, num_classes, mask_height, mask_width]
          containing instance mask predictions.

    Raises:
      ValueError: If `predict` is called before `preprocess`.
    """
    # ======================================================================== #
    # START HERE                                                               #
    # ======================================================================== #
    view_image = params['view_image'] if 'view_image' in params else False

    prediction_dict = dict()
    prediction_dict['image_shape'] = tf.shape(preprocessed_inputs)

    # FPN
    # feature_levels = ['C2', 'C3', 'C4', 'C5', 'M2', 'M3', 'M4', 'M5']
    # concat_levels = False
    # Deeplab
    feature_levels = ['C2', 'C5', 'M3']
    concat_levels = True
    # residual_depth = [48, 256]  # R2, R5

    # ======================================================================== #
    # (1.) Feature Extractor                                                   #
    # ======================================================================== #
    with tf.variable_scope(self.image_feature_extractor_scope):
        image_features_btm_up = self._feature_extractor.extract_image_features(
            preprocessed_inputs,
            scope='')
        features_btm_up = {i: image_features_btm_up[i] for i in
                           image_features_btm_up if i in feature_levels}
        features_btm_up['C3'] = features_btm_up['C2']
        features_btm_up.pop('C2')

        # Concat low level features to higher ones before further prediction.
        with slim.arg_scope(self._aspp_arg_scope_fn() if
                            self._aspp_arg_scope_fn() is not None else []):
            tmp_features_c3 = slim.conv2d(
                features_btm_up['C3'], 64, 1, activation_fn=None,
                scope='LowToHighLevelProjection')
            tmp_features_c3 = slim.max_pool2d(tmp_features_c3, [4, 4], stride=4)
            features_btm_up['C5'] = tf.concat(
                [tmp_features_c3, features_btm_up['C5']],
                axis=-1, name='LowToHighLevelConcat')

        atrous_rates = [0, 6, 12, 18]
        features_btm_up['C5'] = layers.aspp(
            features_btm_up['C5'], 256, atrous_rates=atrous_rates,
            image_pooling=True, depthwise_separable=False,
            arg_scope_fn=self._aspp_arg_scope_fn)

        # with tf.variable_scope('LowLevel'):
        #     atrous_rates = [1, 2, 3, 4]
        #     features_btm_up['C3'] = layers.aspp(
        #         features_btm_up['C3'], 64, atrous_rates=atrous_rates,
        #         image_pooling=False, depthwise_separable=False,
        #         arg_scope_fn=self._aspp_arg_scope_fn)

    # ======================================================================== #
    # (1.) Decoder 1                                                           #
    # ======================================================================== #
    # Deeplab uses concat, fpn uses addition.
    with tf.variable_scope(self.decoder_feature_extractor_scope):
        _features_btm_up = [('C3', features_btm_up['C3']),
                            ('C5', features_btm_up['C5'])]
        top_down_feature = self._feature_extractor.extract_decoder_features(
            _features_btm_up,
            scope='',
            concat_levels=concat_levels,
            residual_depth=[64, 256])

    # ======================================================================== #
    # (2.) Semantic label                                                      #
    # ======================================================================== #
    with tf.variable_scope('SemanticPredictor'):
        atrous_rates = [1, 2, 4, 8]
        semantic_feature = layers.aspp(
            list(top_down_feature.values())[-1], 64, atrous_rates=atrous_rates,
            image_pooling=False, depthwise_separable=False,
            arg_scope_fn=self._aspp_arg_scope_fn)
        refined_semantic_decoder_features = layers.refinement_layer(
            semantic_feature, 256,
            depthwise_separable=False,
            arg_scope_fn=self._refinement_arg_scope_fn)
        prediction_dict['semantic_predictions'] = lib.predict_label_logits(
                refined_semantic_decoder_features, self.num_semantic_classes)

    if self._instance_segmentation:
        # ==================================================================== #
        # (3.) Decoder 2                                                       #
        # ==================================================================== #
        with tf.variable_scope(self.decoder_feature_extractor_scope + '_2'):
            _features_btm_up = [('C3', features_btm_up['C3']),
                                ('C5', features_btm_up['C5'])]
            top_down_feature_ins = \
                self._feature_extractor.extract_decoder_features(
                    _features_btm_up,
                    scope='',
                    concat_levels=concat_levels,
                    residual_depth=[256, 256])

        # ==================================================================== #
        # (4.) Instance label                                                  #
        # ==================================================================== #
        with tf.variable_scope('InstancePredictor'):
            # # Adding the ASPP layer between before the instance branch.
            # # Here the ASPP layer is based on deeplabV3+
            # with tf.variable_scope('Aspp'):
            #     instance_aspp = layers.aspp(
            #         list(top_down_feature_ins.values())[-1], 128,
            #         atrous_rates=[6, 12, 18], image_pooling=False,
            #         arg_scope_fn=self._aspp_arg_scope_fn)
            refined_instance_features = layers.refinement_layer(
                list(top_down_feature_ins.values())[-1], 256,
                depthwise_separable=False,
                arg_scope_fn=self._refinement_arg_scope_fn)
            instance_clusters = []
            for idx, ins in enumerate(MAX_NUM_INSTANCES):
                instance_clusters.append(
                    lib.predict_label_logits(refined_instance_features, ins,
                                             scope_name='logits_' + str(idx)))
            prediction_dict['instance_predictions'] = instance_clusters

    # TODO: Hack to get original image. VGG style mean.
    summary_image_ori = preprocessed_inputs + [[123.68, 116.779, 103.939]]
    prediction_dict['summary_images'] = summary_image_ori

    if fields.InputDataFields.original_image in params:
        prediction_dict[fields.InputDataFields.original_image] = \
            params[fields.InputDataFields.original_image]

    # Viewing the results.
    if view_image:
        with tf.variable_scope('RGB'):
            self._add_image_to_summary(prediction_dict['summary_images'], 'rgb')
        with tf.variable_scope('Semantic'):
            prediction_dict['semantic_image'] = self._add_image_to_summary(
                prediction_dict['semantic_predictions'], 'semantic')
        if self._instance_segmentation:
            with tf.variable_scope('Instance'):
                prediction_dict['instance_image'] = self._add_image_to_summary(
                    prediction_dict['instance_predictions'],
                    'instance_clusters',
                    semantic_image=prediction_dict['semantic_image'])

    return prediction_dict

  def postprocess(self, prediction_dict, true_image_shapes, **params):
    """Convert prediction tensors to final detections.

    This function converts raw predictions tensors to final detection results.
    See base class for output format conventions.  Note also that by default,
    scores are to be interpreted as logits, but if a score_converter is used,
    then scores are remapped (and may thus have a different interpretation).

    If number_of_stages=1, the returned results represent proposals from the
    first stage RPN and are padded to have self.max_num_proposals for each
    image; otherwise, the results can be interpreted as multiclass detections
    from the full two-stage model and are padded to self._max_detections.

    Args:
      prediction_dict: a dictionary holding prediction tensors (see the
        documentation for the predict method.  If number_of_stages=1, we
        expect prediction_dict to contain `rpn_box_encodings`,
        `rpn_objectness_predictions_with_background`, `rpn_features_to_crop`,
        and `anchors` fields.  Otherwise we expect prediction_dict to
        additionally contain `refined_box_encodings`,
        `class_predictions_with_background`, `num_proposals`,
        `proposal_boxes` and, optionally, `mask_predictions` fields.
      true_image_shapes: int32 tensor of shape [batch, 3] where each row is
        of the form [height, width, channels] indicating the shapes
        of true images in the resized images, as resized images can be padded
        with zeros.

    Returns:
      detections: a dictionary containing the following fields
        detection_boxes: [batch, max_detection, 4]
        detection_scores: [batch, max_detections]
        detection_classes: [batch, max_detections]
          (this entry is only created if rpn_mode=False)
        num_detections: [batch]

    Raises:
      ValueError: If `predict` is called before `preprocess`.
    """
    field = fields.DetectionResultFields
    with tf.variable_scope('PostprocessInference'):
      detections_dict = {}
      # 1. Semantic prediction
      semantic_prediction, semantic_prediction_probability = \
          self._postprocess_logits(
              prediction_dict['semantic_predictions'], true_image_shapes)
      detections_dict[field.detection_semantic] = semantic_prediction
      detections_dict[field.detection_semantic_heatmap] \
          = semantic_prediction_probability
      if self._instance_segmentation:
          # 2. Instance prediction
          # instance_prediction, instance_prediction_probability = \
          #     self._postprocess_logits(
          #         prediction_dict['instance_predictions'], true_image_shapes)
          # detections_dict[field.detection_masks] = instance_prediction
          # detections_dict[field.detection_masks_heatmap] \
          #     = instance_prediction_probability
          instance_prediction = \
              self._postprocess_cluster(
                  prediction_dict['instance_predictions'],
                  semantic_prediction, true_image_shapes)
          detections_dict[field.detection_masks] = instance_prediction
          # 3. Panoptic prediction
          with tf.variable_scope('Panoptic'):
            sem_image = tf.cast(semantic_prediction, dtype=tf.uint8)
            ins_image = tf.cast(instance_prediction, dtype=tf.uint8)
            sem_mask = tf.ones_like(sem_image, dtype=sem_image.dtype)
            ins_mask = tf.where(
                tf.greater(sem_image, sem_mask*self.num_classes),
                tf.zeros_like(ins_image), ins_image)
            zero_image = tf.zeros_like(ins_image, dtype=tf.uint8)
            panoptic_image = tf.concat(
                [sem_image, ins_mask, zero_image], axis=-1)
            tf.summary.image('panoptic', panoptic_image)
            detections_dict[field.detection_masks_image] = ins_image
            detections_dict[field.detection_panoptic_image] = panoptic_image
    return detections_dict

  @staticmethod
  def _postprocess_logits(logits, true_image_shapes):
    """Postprocess the raw logits prediction.

    Args:
      logits: A float tensor with shape [batch, H, W, cls] representing
        the raw logits prediction from model.
      true_image_shapes: int32 tensor of shape [batch, 3] where each row is
        of the form [height, width, channels] indicating the shapes
        of true images in the resized images, as resized images can be padded
        with zeros.

    Returns:
      semantic_prediction: int32 tensor with shape
        [batch, H, W, 1] representing the class label for each element.
      semantic_prediction_probability: int32 tensor with shape
        [batch, H, W, 1] representing the softmax probability of
        semantic_prediction.
    """
    def resize_logits(args):
      logits_image = tf.expand_dims(args[0], axis=0)
      image_shape = args[1][0:2]
      logits_image = tf.image.resize_bilinear(
          logits_image, image_shape, align_corners=True)
      return tf.squeeze(logits_image, axis=0)

    # Looping through the batch since some batch may have different size.
    prediction_logits = tf.map_fn(
        resize_logits, elems=[logits, true_image_shapes], dtype=logits.dtype)
    prediction_softmax = tf.nn.softmax(prediction_logits, axis=-1)
    prediction_probability = tf.reduce_max(prediction_softmax, axis=3,
                                           keepdims=True)
    prediction = tf.argmax(prediction_softmax, axis=3, output_type=tf.int32)
    prediction = tf.expand_dims(prediction, axis=-1)
    return prediction, prediction_probability

  def _postprocess_cluster(self, features, semantic_prediction,
                           true_image_shapes):
      """Postprocess the raw logits prediction.

      Args:
        features: A list of float tensor with shape [batch, H, W, D]
          representing the raw embedding prediction from model.
        true_image_shapes: int32 tensor of shape [batch, 3] where each row is
          of the form [height, width, channels] indicating the shapes
          of true images in the resized images, as resized images can be padded
          with zeros.

      Returns:
        instance_prediction: int32 tensor with shape
          [batch, H, W, 1] representing the class label for each element.
      """

      def resize(args):
          image = tf.expand_dims(args[0], axis=0)
          image_shape = args[1][0:2]
          image = tf.image.resize_bilinear(
              image, image_shape, align_corners=True)
          return tf.squeeze(image, axis=0)

      # Concatenating the different output branches, each with a specified
      # dimension.
      features = tf.concat(features, axis=-1)
      # Looping through the batch since some batch may have different size.
      features = tf.map_fn(
          resize, elems=[features, true_image_shapes], dtype=features.dtype)
      instances = self._clustering_batch(features, semantic_prediction)
      return instances

  def _clustering_batch(self, features, semantic_image):
      """Cluster the instance embeddings.

      Args:
          features: instance embeddings.
      """
      def single_image_clustering(args):
          return clustering.meanshift_clustering_tf(
              args[0], args[1], DELTA_D)

      # Mask to mask out the background.
      bg_filter_batch = tf.ones_like(semantic_image) * self.num_classes
      bg_filter_batch = tf.less(semantic_image, bg_filter_batch)
      # Python code from Andre Brehme
      if CLUSTERING_PYTHON:
          bg_filter_batch = tf.cast(bg_filter_batch, dtype=tf.int32)
          bg_filter_batch = bg_filter_batch - tf.constant([1], dtype=tf.int32)
          dst_threshold = tf.constant(DELTA_D, dtype=tf.float32)
          instances = tf.py_func(
              clustering.meanshift_clustering_ext,
              inp=[bg_filter_batch, features, dst_threshold],
              Tout=tf.int64)
          instances = tf.cast(instances, dtype=tf.int64)
      else:
          # Ported code to tensorflow.
          elems = [features, bg_filter_batch]
          instances = tf.map_fn(single_image_clustering, elems=elems,
                                dtype=tf.int64)
      return instances

  def _format_groundtruth_data(self, true_image_shapes):
    """Helper function for preparing groundtruth data for target assignment.

    In order to be consistent with the model.DetectionModel interface,
    groundtruth boxes are specified in normalized coordinates and classes are
    specified as label indices with no assumed background category.  To prepare
    for target assignment, we:
    1) convert boxes to absolute coordinates,
    2) add a background class at class index 0
    3) groundtruth instance masks, if available, are resized to match
       image_shape.

    Args:
      true_image_shapes: int32 tensor of shape [batch, 3] where each row is
        of the form [height, width, channels] indicating the shapes
        of true images in the resized images, as resized images can be padded
        with zeros.

    Returns:
      groundtruth_boxlists: A list of BoxLists containing (absolute) coordinates
        of the groundtruth boxes.
      groundtruth_classes_with_background_list: A list of 2-D one-hot
        (or k-hot) tensors of shape [num_boxes, num_classes+1] containing the
        class targets with the 0th index assumed to map to the background class.
      groundtruth_masks_list: If present, a list of 3-D tf.float32 tensors of
        shape [num_boxes, image_height, image_width] containing instance masks.
        This is set to None if no masks exist in the provided groundtruth.
    """
    groundtruth_boxlists = [
        box_list_ops.to_absolute_coordinates(
            box_list.BoxList(boxes), true_image_shapes[i, 0],
            true_image_shapes[i, 1])
        for i, boxes in enumerate(
            self.groundtruth_lists(fields.BoxListFields.boxes))
    ]
    groundtruth_classes_with_background_list = [
        tf.cast(
            tf.pad(one_hot_encoding, [[0, 0], [1, 0]], mode='CONSTANT'),
            dtype=tf.float32)
        for one_hot_encoding in self.groundtruth_lists(
            fields.BoxListFields.classes)]

    groundtruth_masks_list = self._groundtruth_lists.get(
        fields.BoxListFields.masks)
    # TODO(rathodv): Remove mask resizing once the legacy pipeline is deleted.
    if groundtruth_masks_list is not None and self._resize_masks:
      resized_masks_list = []
      for mask in groundtruth_masks_list:
        _, resized_mask, _ = self._image_resizer_fn(
            # Reuse the given `image_resizer_fn` to resize groundtruth masks.
            # `mask` tensor for an image is of the shape [num_masks,
            # image_height, image_width]. Below we create a dummy image of the
            # the shape [image_height, image_width, 1] to use with
            # `image_resizer_fn`.
            image=tf.zeros(tf.stack([tf.shape(mask)[1], tf.shape(mask)[2], 1])),
            masks=mask)
        resized_masks_list.append(resized_mask)
      groundtruth_masks_list = resized_masks_list

    if self.groundtruth_has_field(fields.BoxListFields.weights):
      groundtruth_weights_list = self.groundtruth_lists(
          fields.BoxListFields.weights)
    else:
      # Set weights for all batch elements equally to 1.0
      groundtruth_weights_list = []
      for groundtruth_classes in groundtruth_classes_with_background_list:
        num_gt = tf.shape(groundtruth_classes)[0]
        groundtruth_weights = tf.ones(num_gt)
        groundtruth_weights_list.append(groundtruth_weights)

    return (groundtruth_boxlists, groundtruth_classes_with_background_list,
            groundtruth_masks_list, groundtruth_weights_list)

  def loss(self, prediction_dict, true_image_shapes, scope=None):
    """Compute scalar loss tensors given prediction tensors.

    If number_of_stages=1, only RPN related losses are computed (i.e.,
    `rpn_localization_loss` and `rpn_objectness_loss`).  Otherwise all
    losses are computed.

    Args:
      prediction_dict: a dictionary holding prediction tensors (see the
        documentation for the predict method.  If number_of_stages=1, we
        expect prediction_dict to contain `rpn_box_encodings`,
        `rpn_objectness_predictions_with_background`, `rpn_features_to_crop`,
        `image_shape`, and `anchors` fields.  Otherwise we expect
        prediction_dict to additionally contain `refined_box_encodings`,
        `class_predictions_with_background`, `num_proposals`, and
        `proposal_boxes` fields.
      true_image_shapes: int32 tensor of shape [batch, 3] where each row is
        of the form [height, width, channels] indicating the shapes
        of true images in the resized images, as resized images can be padded
        with zeros.
      scope: Optional scope name.

    Returns:
      a dictionary mapping loss keys (`first_stage_localization_loss`,
        `first_stage_objectness_loss`, 'second_stage_localization_loss',
        'second_stage_classification_loss') to scalar tensors representing
        corresponding loss values.
    """
    with tf.variable_scope(scope, 'Loss', prediction_dict.values()):
      loss_dict = dict()

      (_, _, groundtruth_masks_list, _) = self._format_groundtruth_data(
          true_image_shapes)

      (_loss_dict, semantic_logits, not_ignore_mask) = self._loss_semantic(
          prediction_dict['semantic_predictions'],
          self.groundtruth_lists('semantic'),
          ignore_label=0,
          upsample_logits=True)
      loss_dict.update(_loss_dict)

      if self._instance_segmentation:
          # _loss_dict = self._loss_instance(
          #     prediction_dict['instance_predictions'],
          #     groundtruth_masks_list,
          #     max_num_instances=MAX_NUM_INSTANCES,
          #     iou_threshold=IOU_THRESHOLD_INSTANCES,
          #     upsample_logits=True)
          _loss_dict = self._loss_instance_cluster(
              prediction_dict['instance_predictions'],
              groundtruth_masks_list,
              true_image_shape=true_image_shapes[0],
              upsample_predictions=True)
          loss_dict.update(_loss_dict)

      return loss_dict

  def _loss_semantic(self, predictions, groundtruth_list, ignore_label,
                     upsample_logits):
    """Computes scalar semantic loss tensors.

    We use the implementation used in deeplab.

    Args:
        predictions: A 4-D float tensor of shape
            [batch_size x W x H x num_semantic_classes] containing
            predicted semantic labels.
        groundtruth_list: a list of 3-D tensor of shape
            [batch_size x W x H x 1] element list of pixel-wise semantic
            label with the same shape as the input image. We use a list
            to make it compatible with the rest of the ground truths.
        ignore_label: Integer, label to ignore.
        upsample_logits: to determine if the label or the logits are
            upsampled / downsampled.

    Returns:
        a dictionary mapping loss keys (`first_stage_localization_loss`,
        `first_stage_objectness_loss`) to scalar tensors representing
        corresponding loss values.
    """
    # TODO : Semantic labels should be saved using just 1 channel.
    labels = []
    for gt in groundtruth_list:
        labels.append(
            tf.expand_dims(tf.slice(gt, [0, 0, 0], [-1, -1, 1]), axis=0))
    labels = tf.concat(labels, axis=0)
    logits = predictions
    with tf.variable_scope('SemanticLoss'):
        if groundtruth_list is None:
            raise ValueError('No label for softmax cross entropy loss.')
        if upsample_logits:
            # Label is not downsampled, and instead we upsample logits.
            # logits = slim.conv2d_transpose(
            #     logits,
            #     num_outputs=self.num_semantic_classes,
            #     kernel_size=[2, 2],
            #     stride=2)
            # logits = slim.conv2d_transpose(
            #     logits,
            #     num_outputs=self.num_semantic_classes,
            #     kernel_size=[2, 2],
            #     stride=2)
            # TODO: we try to prevent image resize for MPI
            if fp_dtype != tf.float32:
                def cond(_labels, _logits):
                    return tf.not_equal(tf.reduce_sum(tf.shape(_logits)[1:3]),
                                        tf.reduce_sum(tf.shape(_labels)[1:3]))
                def body(_labels, _logits):
                    _logits = ops.nearest_neighbor_upsampling(_logits, 2)
                    shape_assert = tf.Assert(
                        tf.less_equal(tf.reduce_sum(tf.shape(_logits)[1:3]),
                                      tf.reduce_sum(tf.shape(_labels)[1:3])),
                        ['Shape is too big for logits compared to labels.'])
                    with tf.control_dependencies([shape_assert]):
                        return [_labels, _logits]
                loop_vars = [labels, logits]
                labels_shape = labels.get_shape()
                logits_shape = logits.get_shape().as_list()
                logits_shape = tf.TensorShape(
                    [logits_shape[0], None, None, logits_shape[3]])
                output = tf.while_loop(
                    cond, body, loop_vars,
                    shape_invariants=[labels_shape, logits_shape])
                logits = output[1]
            else:
                logits = tf.cast(tf.image.resize_bilinear(
                    logits, tf.shape(labels)[1:3], align_corners=True),
                    dtype=logits.dtype)
            scaled_labels = labels
        else:
            # Label is downsampled to the same size as logits.
            logits = logits
            scaled_labels = tf.image.resize_nearest_neighbor(
                labels, tf.shape(logits)[1:3], align_corners=True)
        original_shape = tf.shape(scaled_labels)
        scaled_labels = tf.reshape(scaled_labels, shape=[-1])
        one_hot_labels = slim.one_hot_encoding(
            scaled_labels, self.num_semantic_classes,
            on_value=1.0, off_value=0.0)
        logits = tf.reshape(logits, shape=[-1, self.num_semantic_classes])
        not_ignore_mask = tf.cast(
            tf.not_equal(scaled_labels, ignore_label), dtype=tf.float32)
        semantic_loss = tf.losses.softmax_cross_entropy(
            one_hot_labels, logits,
            weights=not_ignore_mask*self._semantic_loss_weight,
            loss_collection=None)
        semantic_loss = tf.identity(semantic_loss, name='semantic_loss')
        loss_dict = {semantic_loss.op.name: semantic_loss}
        not_ignore_mask = tf.reshape(not_ignore_mask, shape=original_shape)
        return loss_dict, logits, not_ignore_mask

  @staticmethod
  def _loss_instance_cluster(predictions,
                             groundtruth_list,
                             true_image_shape,
                             upsample_predictions=True):
      """Computes scalar semantic loss tensors.

      Args:
          predictions: A list of predictions with different dimensions/depth.
          groundtruth_list: A list (batch_size) of 4-D tensor of shape
              [K x W x H x 1] of pixel-wise binary label with the same shape
              as the input image. We use a list
              to make it compatible with the rest of the ground truths.
          true_image_shape: Tensor [W,H,C]
      """

      def adding_count_to_instance(args):
          return args[0] * args[1]

      # While loop condition for upsampling.
      def cond(_labels, _prediction):
          return tf.not_equal(tf.reduce_sum(tf.shape(_prediction)[1:3]),
                              tf.reduce_sum(tf.shape(_labels)[1:3]))

      # While loop body for upsampling.
      def body(_labels, _prediction):
          _prediction = ops.nearest_neighbor_upsampling(_prediction, 2)
          shape_assert = tf.Assert(
              tf.less_equal(tf.reduce_sum(tf.shape(_prediction)[1:3]),
                            tf.reduce_sum(tf.shape(_labels)[1:3])),
              ['Shape is too big for prediction compared to labels.'])
          with tf.control_dependencies([shape_assert]):
              return [_labels, _prediction]

      # [batch_size x W x H]
      zeros_gt_batch = tf.zeros([predictions[0].shape[0],
                                 true_image_shape[0],
                                 true_image_shape[1]],
                                dtype=tf.float32)

      # We need to add zero/empty label if the groundtruth has no instance.
      labels = []
      for gt, zeros_gt in zip(groundtruth_list, tf.unstack(zeros_gt_batch)):
          gt = tf.cond(tf.equal(tf.size(gt), 0),
                       lambda: tf.expand_dims(zeros_gt, axis=0),
                       lambda: gt)
          counts = tf.cast(tf.range(tf.shape(gt)[0]) + 1, dtype=tf.float32)
          gt_batch = tf.map_fn(
              adding_count_to_instance, elems=[gt, counts], dtype=gt.dtype)
          labels.append(tf.reduce_sum(gt_batch, axis=0))
      labels = tf.stack(labels, axis=0)

      loss_dict = {}
      with tf.variable_scope('InstanceLoss'):
          for prediction in predictions:
              if upsample_predictions:
                  # TODO: we try to prevent image resize for MPI
                  if fp_dtype != tf.float32:
                      loop_vars = [labels, prediction]
                      labels_shape = labels.get_shape()
                      logits_shape = prediction.get_shape().as_list()
                      logits_shape = tf.TensorShape(
                          [logits_shape[0], None, None, logits_shape[3]])
                      output = tf.while_loop(
                          cond, body, loop_vars,
                          shape_invariants=[labels_shape, logits_shape])
                      prediction = output[1]
                  else:
                      prediction = tf.cast(
                          tf.image.resize_bilinear(
                              prediction, tf.shape(labels)[1:3],
                              align_corners=True),
                          dtype=prediction.dtype)
                  scaled_labels = labels
              else:
                  prediction = prediction
                  scaled_labels = tf.image.resize_nearest_neighbor(
                      labels, tf.shape(prediction)[1:3], align_corners=True)
              disc_loss, l_var, l_dist, l_reg = \
                  losses.discriminative_loss(
                      prediction, scaled_labels,
                      tf.shape(prediction)[-1], tf.shape(prediction)[1:3],
                      DELTA_V, DELTA_D, PARAM_VAR, PARAM_DIST, PARAM_REG)
              disc_loss = tf.reduce_sum(disc_loss, name='instance_disc_loss')
              l_var = tf.reduce_sum(l_var, name='instance_l_var')
              l_dist = tf.reduce_sum(l_dist, name='instance_l_dist')
              l_reg = tf.reduce_sum(disc_loss, name='instance_l_reg')
              tf.summary.scalar('Loss/instance_l_var', l_var)
              tf.summary.scalar('Loss/instance_l_dist', l_dist)
              tf.summary.scalar('Loss/instance_l_reg', l_reg)
              loss_dict.update({disc_loss.op.name: disc_loss})
          return loss_dict

  def _loss_instance(self,
                     predictions,
                     groundtruth_list,
                     max_num_instances=1,
                     iou_threshold=0.5,
                     upsample_logits=True):
    """Computes scalar semantic loss tensors.

    We use the implementation used in deeplab.

    Args:
        predictions: A 4-D float tensor of shape
            [batch_size x W x H x num_semantic_classes] containing
            predicted semantic labels.
        groundtruth_list: a list of 3-D tensor of shape
            [batch_size x W x H x 1] element list of pixel-wise semantic
            label with the same shape as the input image. We use a list
            to make it compatible with the rest of the ground truths.
        max_num_instances: Integer, maximum number of instances.
        upsample_logits: to determine if the label or the logits are
            upsampled / downsampled.

    Returns:
        a dictionary mapping loss keys (`first_stage_localization_loss`,
        `first_stage_objectness_loss`) to scalar tensors representing
        corresponding loss values.
    """
    # def similarity_match(args):
    #   (gt_instances, predictions_one_hot, zeros_gt) = args
    #   gt_instances = tf.cond(
    #       tf.equal(tf.size(gt_instances), 0),
    #       lambda: tf.expand_dims(zeros_gt, axis=0),
    #       lambda: tf.reshape(gt_instances,
    #                          [tf.shape(gt_instances)[0], -1]))
    #   similarity_matrix = target_assigner.assign_instance_targets(
    #       predictions_one_hot, gt_instances)
    #   targets = tf.argmax(similarity_matrix, axis=-1)
    #   targets_probability = tf.reduce_max(similarity_matrix, axis=-1)
    #   return tf.gather(gt_instances, targets), targets_probability

    logits = predictions
    batch_size = len(groundtruth_list)
    # gt_batch = tf.convert_to_tensor(groundtruth_list)
    with tf.variable_scope('InstanceLoss'):
        if upsample_logits:
            logits = tf.image.resize_bilinear(
                logits, tf.shape(groundtruth_list[0])[1:3], align_corners=True)
        softmax_batch = tf.nn.softmax(logits, axis=-1)
        # probability = tf.reduce_max(softmax_batch, axis=3, keepdims=True)
        predictions_batch = tf.argmax(
            softmax_batch, axis=3, output_type=tf.int32)
        predictions_batch = tf.reshape(
            predictions_batch, shape=[batch_size, -1])
        predictions_one_hot_batch = tf.one_hot(
            predictions_batch, depth=max_num_instances, axis=1)

        # # Here we match the instances to the groundtruths using iou.
        # zeros_gt_batch = tf.zeros_like(predictions_batch, dtype=tf.float32)
        # elems = (gt_batch, predictions_one_hot_batch, zeros_gt_batch)
        # output_dtype = (gt_batch.dtype, tf.float32)
        # gt_targets_batch, targets_probability_batch =\
        #     tf.map_fn(similarity_match, elems=elems, dtype=output_dtype)
        gt_targets_batch_list = []
        targets_probability_batch_list = []
        zeros_gt_batch = tf.zeros_like(predictions_batch, dtype=tf.float32)
        for gt_instances, predictions_one_hot, zeros_gt in \
                zip(groundtruth_list,
                    tf.unstack(predictions_one_hot_batch),
                    tf.unstack(zeros_gt_batch)):
            gt_instances = tf.cond(
                tf.equal(tf.size(gt_instances), 0),
                lambda: tf.expand_dims(zeros_gt, axis=0),
                lambda: tf.reshape(gt_instances,
                                   [tf.shape(gt_instances)[0], -1]))
            gt_background = tf.reduce_sum(gt_instances, axis=0, keepdims=True)
            gt_background = \
                tf.ones_like(gt_background,
                             dtype=gt_background.dtype) - gt_background
            gt_instances = tf.concat([gt_background, gt_instances], axis=0)
            similarity_matrix = target_assigner.assign_instance_targets(
                predictions_one_hot, gt_instances)
            targets = tf.argmax(similarity_matrix, axis=-1)
            targets_probability = tf.reduce_max(similarity_matrix, axis=-1)
            gt_targets_batch_list.append(tf.gather(gt_instances, targets))
            targets_probability_batch_list.append(targets_probability)
        gt_targets_batch = tf.stack(gt_targets_batch_list)
        targets_probability_batch = tf.stack(targets_probability_batch_list)

        # Here we filter out matches with iou less than a threshold.
        target_weights = tf.where(
            tf.greater(targets_probability_batch, iou_threshold),
            tf.ones_like(targets_probability_batch),
            tf.zeros_like(targets_probability_batch))

        normalizer = \
            tf.cast(tf.shape(gt_targets_batch)[-1], tf.float32) * \
            tf.maximum(
                tf.reduce_sum(target_weights, axis=1, keepdims=True),
                tf.ones((batch_size, 1), dtype=tf.float32))

        loss_batch = self._second_stage_mask_loss(
                predictions_one_hot_batch,
                gt_targets_batch,
                weights=target_weights*self._instance_loss_weight)

        # TODO: Still need to check what happened.
        instance_losses = tf.reduce_sum(loss_batch, axis=2) / normalizer

        instance_loss = tf.reduce_sum(instance_losses, name='instance_loss')
        loss_dict = {instance_loss.op.name: instance_loss}
        return loss_dict

  def restore_map(self,
                  fine_tune_checkpoint_type='detection',
                  load_all_detection_checkpoint_vars=False,
                  scopes_to_ignore=None):
    """Returns a map of variables to load from a foreign checkpoint.

    See parent class for details.

    Args:
      fine_tune_checkpoint_type: whether to restore from a full detection
        checkpoint (with compatible variable names) or to restore from a
        classification checkpoint for initialization prior to training.
        Valid values: `detection`, `classification`. Default 'detection'.
      load_all_detection_checkpoint_vars: whether to load all variables (when
         `fine_tune_checkpoint_type` is `detection`). If False, only variables
         within the feature extractor scopes are included. Default False.
      scopes_to_ignore: a list of scopes to ignore

    Returns:
      A dict mapping variable names (to load from a checkpoint) to variables in
      the model graph.
    Raises:
      ValueError: if fine_tune_checkpoint_type is neither `classification`
        nor `detection`.
    """
    if fine_tune_checkpoint_type not in ['detection', 'classification']:
      raise ValueError('Not supported fine_tune_checkpoint_type: {}'.format(
          fine_tune_checkpoint_type))
    if fine_tune_checkpoint_type == 'classification':
      default_scopes = [self.image_feature_extractor_scope]
      if scopes_to_ignore is None:
          scopes_to_ignore = default_scopes
      else:
          if not isinstance(scopes_to_ignore, list):
              scopes_to_ignore = [scopes_to_ignore]
          scopes_to_ignore += default_scopes
      return self._feature_extractor.restore_from_classification_checkpoint_fn(
          scopes_to_ignore=scopes_to_ignore)

    variables_to_restore = tf.global_variables()
    variables_to_restore.append(slim.get_or_create_global_step())
    # Only load feature extractor variables to be consistent with loading from
    # a classification checkpoint.
    include_patterns = None
    if not load_all_detection_checkpoint_vars:
      include_patterns = []
    feature_extractor_variables = tf.contrib.framework.filter_variables(
        variables_to_restore, include_patterns=include_patterns)
    return {var.op.name: var for var in feature_extractor_variables}

  def _add_image_to_summary(self, raw_prediction, mode, semantic_image=None):
    """ Drawing images for verification.

    Args:
        raw_prediction: A [1, H, W, num_classes] tensor.

    Return:
        An image Tensor.
    """
    if mode == 'rgb':
        tf.summary.image('Original',
                         tf.expand_dims(raw_prediction[0], axis=0))
        return raw_prediction
    elif mode == 'semantic':
        groundtruth_image = tf.expand_dims(
            self.groundtruth_lists('semantic')[0], axis=0)
        raw_prediction = tf.expand_dims(raw_prediction[0], axis=0)
        return lib.add_semantic_image_to_summary(
            raw_prediction, groundtruth_image, self.num_classes)
    elif mode == 'instance_clusters':
        groundtruth_images = tf.expand_dims(
            self.groundtruth_lists('masks')[0], axis=-1)
        groundtruth_image = tf.zeros(tf.shape(groundtruth_images)[1:3])
        groundtruth_image = tf.expand_dims(groundtruth_image, axis=0)
        groundtruth_image = tf.expand_dims(groundtruth_image, axis=-1)
        groundtruth_image = tf.concat([groundtruth_image, groundtruth_images],
                                      axis=0)
        groundtruth_image = tf.reduce_sum(groundtruth_image, axis=0,
                                          keepdims=True)
        raw_prediction = tf.concat(raw_prediction, axis=-1)
        raw_prediction = tf.expand_dims(raw_prediction[0], axis=0)
        cluster_prediction = tf.image.resize_nearest_neighbor(
            raw_prediction, tf.shape(groundtruth_image)[1:3],
            align_corners=True)
        instances = self._clustering_batch(cluster_prediction, semantic_image)
        lib.add_cluster_instance_image_to_summary(instances, groundtruth_image)
        return None
    elif mode == 'instance_logits':
        groundtruth_images = tf.expand_dims(
            self.groundtruth_lists('masks')[0], axis=-1)
        groundtruth_image = tf.zeros(tf.shape(groundtruth_images)[1:3])
        groundtruth_image = tf.expand_dims(groundtruth_image, axis=0)
        groundtruth_image = tf.expand_dims(groundtruth_image, axis=-1)
        groundtruth_image = tf.concat([groundtruth_image, groundtruth_images],
                                      axis=0)
        groundtruth_image = tf.reduce_sum(groundtruth_image, axis=0,
                                          keepdims=True)
        raw_prediction = tf.expand_dims(raw_prediction[0], axis=0)
        return lib.add_logit_instance_image_to_summary(
            raw_prediction, groundtruth_image)
    else:
        raise ValueError("Unknown mode...")
