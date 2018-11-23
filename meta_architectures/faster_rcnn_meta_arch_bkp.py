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
"""Faster R-CNN meta-architecture definition.

General tensorflow implementation of Faster R-CNN detection models.

See Faster R-CNN: Ren, Shaoqing, et al.
"Faster R-CNN: Towards real-time object detection with region proposal
networks." Advances in neural information processing systems. 2015.

"""
from abc import abstractmethod
from functools import partial
import tensorflow as tf
import math

from object_detection import Options
from object_detection import fp_dtype
from object_detection.builders import box_predictor_builder
from object_detection.core import box_list
from object_detection.core import box_list_ops
from object_detection.core import box_predictor
from object_detection.core import losses
from object_detection.core import model
from object_detection.core import post_processing
from object_detection.core import standard_fields as fields
from object_detection.core import target_assigner
from object_detection.utils import ops
from object_detection.utils import shape_utils
from object_detection.utils import visualization_utils as vis
from object_detection.meta_architectures import meta_arch_lib as lib

# For debugging purposes
# from nets import resnet_utils
# from nets import resnet_v1

slim = tf.contrib.slim


class FasterRCNNFeatureExtractor(object):
  """Faster R-CNN Feature Extractor definition."""

  def __init__(self,
               is_training,
               first_stage_features_stride,
               batch_norm_trainable=False,
               reuse_weights=None,
               weight_decay=0.0):
    """Constructor.

    Args:
      is_training: A boolean indicating whether the training version of the
        computation graph should be constructed.
      first_stage_features_stride: Output stride of extracted RPN feature map.
      batch_norm_trainable: Whether to update batch norm parameters during
        training or not. When training with a relative large batch size
        (e.g. 8), it could be desirable to enable batch norm update.
      reuse_weights: Whether to reuse variables. Default is None.
      weight_decay: float weight decay for feature extractor (default: 0.0).
    """
    self._is_training = is_training
    self._first_stage_features_stride = first_stage_features_stride
    self._train_batch_norm = (batch_norm_trainable and is_training)
    self._reuse_weights = reuse_weights
    self._weight_decay = weight_decay

  @abstractmethod
  def preprocess(self, resized_inputs):
    """Feature-extractor specific preprocessing (minus image resizing)."""
    pass

  @abstractmethod
  def _extract_decoder_features(self, preprocessed_inputs, scope):
    """Extracts first stage decoder features, to be overridden."""
    pass

  @abstractmethod
  def _extract_proposal_features(self, decoder_feature_maps, scope):
    """Extracts first stage RPN features, to be overridden."""
    pass

  @abstractmethod
  def _extract_box_classifier_features(self, proposal_feature_maps, scope):
    """Extracts second stage box classifier features, to be overridden."""
    pass

  def extract_decoder_features(self, preprocessed_inputs, scope):
    """Extracts first stage decoder features based on FPN.

    Args:
      preprocessed_inputs: A [batch, height, width, channels] float tensor
        representing a batch of images.
      scope: A scope name.

    Returns:
      decoder_feature_map: A tensor with shape [batch, height, width, depth]
      activations: A dictionary mapping activation tensor names to tensors.
    """
    with tf.variable_scope(scope, values=[preprocessed_inputs]):
      return self._extract_decoder_features(preprocessed_inputs, scope)

  def extract_proposal_features(self, decoder_feature_maps, scope):
    """Extracts first stage RPN features.

    This function is responsible for extracting feature maps from preprocessed
    images.  These features are used by the region proposal network (RPN) to
    predict proposals.

    Args:
      decoder_feature_maps: A tensor list containing the feature maps
      extracted from 'extract_decoder_features'.
      scope: A scope name.

    Returns:
      rpn_feature_map: A tensor with shape [batch, height, width, depth]
      activations: A dictionary mapping activation tensor names to tensors.
    """
    with tf.variable_scope(scope, values=[decoder_feature_maps]):
      return self._extract_proposal_features(decoder_feature_maps, scope)

  def extract_box_classifier_features(self, proposal_feature_maps, scope):
    """Extracts second stage box classifier features.

    Args:
      proposal_feature_maps: A 4-D float tensor with shape
        [batch_size * self.max_num_proposals, crop_height, crop_width, depth]
        representing the feature map cropped to each proposal.
      scope: A scope name.

    Returns:
      proposal_classifier_features: A 4-D float tensor with shape
        [batch_size * self.max_num_proposals, height, width, depth]
        representing box classifier features for each proposal.
    """
    with tf.variable_scope(
            scope, values=[proposal_feature_maps], reuse=tf.AUTO_REUSE):
      return self._extract_box_classifier_features(proposal_feature_maps, scope)

  def restore_from_classification_checkpoint_fn(
          self,
          **kwargs):
    """Returns a map of variables to load from a foreign checkpoint.

    Returns:
      A dict mapping variable names (to load from a checkpoint) to variables in
      the model graph.
    """
    scopes_to_ignore = [v for _, v in kwargs.items()]
    # This is due to distributed training.
    replica_scopes_to_ignore = [
        'replica_1',
        'replica_2',
        'replica_3',
        'replica_4',
        'replica_5',
        'replica_6',
        'replica_7',
    ]
    variables_to_restore = {}
    for variable in tf.global_variables():
      include_flag = False
      var_name = variable.op.name
      for scope_name in scopes_to_ignore:
        if scope_name in var_name:
          var_name = var_name.replace(scope_name + '/', '')
          include_flag = True
      for scope_name in replica_scopes_to_ignore:
        if scope_name in var_name:
          var_name = var_name.replace('/' + scope_name, '')
          include_flag = True
      if include_flag:
        variables_to_restore[var_name] = variable
    return variables_to_restore


class FasterRCNNMetaArch(model.DetectionModel):
  """Faster R-CNN Meta-architecture definition."""

  def __init__(self,
               is_training,
               is_evaluating,
               num_classes,
               image_resizer_fn,
               feature_extractor,
               number_of_stages,
               first_stage_fpn_min_level,
               first_stage_fpn_max_level,
               first_stage_anchor_generator,
               first_stage_target_assigner,
               first_stage_atrous_rate,
               first_stage_box_predictor_arg_scope_fn,
               first_stage_box_predictor_kernel_size,
               first_stage_box_predictor_depth,
               first_stage_minibatch_size,
               first_stage_sampler,
               first_stage_non_max_suppression_fn,
               first_stage_max_proposals,
               first_stage_localization_loss_weight,
               first_stage_objectness_loss_weight,
               crop_and_resize_fn,
               initial_crop_size,
               maxpool_kernel_size,
               maxpool_stride,
               second_stage_target_assigner,
               second_stage_mask_rcnn_box_predictor,
               second_stage_batch_size,
               second_stage_sampler,
               second_stage_non_max_suppression_fn,
               second_stage_score_conversion_fn,
               second_stage_localization_loss_weight,
               second_stage_classification_loss_weight,
               second_stage_classification_loss,
               second_stage_mask_prediction_loss_weight=1.0,
               hard_example_miner=None,
               parallel_iterations=16,
               add_summaries=True,
               clip_anchors_to_image=False,
               use_static_shapes=False,
               resize_masks=True,
               num_semantic_classes=0,
               first_stage_semantic_arg_scope_fn=None,
               first_stage_semantic_kernel_size=0,
               first_stage_semantic_depth=0,
               first_stage_semantic_loss_weight=0.0,
               dataset_name=None):
    """FasterRCNNMetaArch Constructor.

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
      feature_extractor: A FasterRCNNFeatureExtractor object.
      number_of_stages:  An integer values taking values in {1, 2, 3}. If
        1, the function will construct only the Region Proposal Network (RPN)
        part of the model. If 2, the function will perform box refinement and
        other auxiliary predictions all in the second stage. If 3, it will
        extract features from refined boxes and perform the auxiliary
        predictions on the non-maximum suppressed refined boxes.
        If is_training is true and the value of number_of_stages is 3, it is
        reduced to 2 since all the model heads are trained in parallel in second
        stage during training.
      first_stage_anchor_generator: An anchor_generator.AnchorGenerator object
        (note that currently we only support
        grid_anchor_generator.GridAnchorGenerator objects)
      first_stage_target_assigner: Target assigner to use for first stage of
        Faster R-CNN (RPN).
      first_stage_atrous_rate: A single integer indicating the atrous rate for
        the single convolution op which is applied to the `rpn_features_to_crop`
        tensor to obtain a tensor to be used for box prediction. Some feature
        extractors optionally allow for producing feature maps computed at
        denser resolutions.  The atrous rate is used to compensate for the
        denser feature maps by using an effectively larger receptive field.
        (This should typically be set to 1).
      first_stage_box_predictor_arg_scope_fn: A function to construct tf-slim
        arg_scope for conv2d, separable_conv2d and fully_connected ops for the
        RPN box predictor.
      first_stage_box_predictor_kernel_size: Kernel size to use for the
        convolution op just prior to RPN box predictions.
      first_stage_box_predictor_depth: Output depth for the convolution op
        just prior to RPN box predictions.
      first_stage_minibatch_size: The "batch size" to use for computing the
        objectness and location loss of the region proposal network. This
        "batch size" refers to the number of anchors selected as contributing
        to the loss function for any given image within the image batch and is
        only called "batch_size" due to terminology from the Faster R-CNN paper.
      first_stage_sampler: Sampler to use for first stage loss (RPN loss).
      first_stage_non_max_suppression_fn: batch_multiclass_non_max_suppression
        callable that takes `boxes`, `scores` and optional `clip_window`(with
        all other inputs already set) and returns a dictionary containing
        tensors with keys: `detection_boxes`, `detection_scores`,
        `detection_classes`, `num_detections`. This is used to perform non max
        suppression  on the boxes predicted by the Region Proposal Network
        (RPN).
        See `post_processing.batch_multiclass_non_max_suppression` for the type
        and shape of these tensors.
      first_stage_max_proposals: Maximum number of boxes to retain after
        performing Non-Max Suppression (NMS) on the boxes predicted by the
        Region Proposal Network (RPN).
      first_stage_localization_loss_weight: A float
      first_stage_objectness_loss_weight: A float
      crop_and_resize_fn: A differentiable resampler to use for cropping RPN
        proposal features.
      initial_crop_size: A single integer indicating the output size
        (width and height are set to be the same) of the initial bilinear
        interpolation based cropping during ROI pooling.
      maxpool_kernel_size: A single integer indicating the kernel size of the
        max pool op on the cropped feature map during ROI pooling.
      maxpool_stride: A single integer indicating the stride of the max pool
        op on the cropped feature map during ROI pooling.
      second_stage_target_assigner: Target assigner to use for second stage of
        Faster R-CNN. If the model is configured with multiple prediction heads,
        this target assigner is used to generate targets for all heads (with the
        correct `unmatched_class_label`).
      second_stage_mask_rcnn_box_predictor: Mask R-CNN box predictor to use for
        the second stage.
      second_stage_batch_size: The batch size used for computing the
        classification and refined location loss of the box classifier.  This
        "batch size" refers to the number of proposals selected as contributing
        to the loss function for any given image within the image batch and is
        only called "batch_size" due to terminology from the Faster R-CNN paper.
      second_stage_sampler:  Sampler to use for second stage loss (box
        classifier loss).
      second_stage_non_max_suppression_fn: batch_multiclass_non_max_suppression
        callable that takes `boxes`, `scores`, optional `clip_window` and
        optional (kwarg) `mask` inputs (with all other inputs already set)
        and returns a dictionary containing tensors with keys:
        `detection_boxes`, `detection_scores`, `detection_classes`,
        `num_detections`, and (optionally) `detection_masks`. See
        `post_processing.batch_multiclass_non_max_suppression` for the type and
        shape of these tensors.
      second_stage_score_conversion_fn: Callable elementwise nonlinearity
        (that takes tensors as inputs and returns tensors).  This is usually
        used to convert logits to probabilities.
      second_stage_localization_loss_weight: A float indicating the scale factor
        for second stage localization loss.
      second_stage_classification_loss_weight: A float indicating the scale
        factor for second stage classification loss.
      second_stage_classification_loss: Classification loss used by the second
        stage classifier. Either losses.WeightedSigmoidClassificationLoss or
        losses.WeightedSoftmaxClassificationLoss.
      second_stage_mask_prediction_loss_weight: A float indicating the scale
        factor for second stage mask prediction loss. This is applicable only if
        second stage box predictor is configured to predict masks.
      hard_example_miner:  A losses.HardExampleMiner object (can be None).
      parallel_iterations: (Optional) The number of iterations allowed to run
        in parallel for calls to tf.map_fn.
      add_summaries: boolean (default: True) controlling whether summary ops
        should be added to tensorflow graph.
      clip_anchors_to_image: Normally, anchors generated for a given image size
      are pruned during training if they lie outside the image window. This
      option clips the anchors to be within the image instead of pruning.
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
    super(FasterRCNNMetaArch, self).__init__(
        num_classes=num_classes,
        num_semantic_classes=num_semantic_classes)

    if (dataset_name is None) or (dataset_name not in
                                  lib.DATASET_DICT.keys()):
        raise ValueError(
            'dataset_name is empty or does not exist, we need to use a '
            'known dataset category index dict.')
    self._category_index = lib.DATASET_DICT[dataset_name]

    # TODO: We assume that at least the first 2 layers will give us
    # enough proposals
    if is_training and \
            second_stage_batch_size > (2 * first_stage_max_proposals):
        tf.logging.warning("second_stage_batch_size should be no greater than "
                           "2 x first_stage_max_proposals.")
        tf.logging.warning("However we are still proceeding with "
                           "second_stage_batch_size = {}.".format(
                            second_stage_batch_size))

    self._is_training = is_training
    self._is_evaluating = is_evaluating
    self._is_predicting = is_training is False and is_evaluating is False

    self._image_resizer_fn = image_resizer_fn
    self._resize_masks = resize_masks
    self._feature_extractor = feature_extractor
    self._number_of_stages = number_of_stages

    self._proposal_target_assigner = first_stage_target_assigner
    self._detector_target_assigner = second_stage_target_assigner
    # Both proposal and detector target assigners use the same box coder
    self._box_coder = self._proposal_target_assigner.box_coder

    # (First stage) Semantic segmentation output layer
    self._first_stage_semantic_arg_scope_fn = (
        first_stage_semantic_arg_scope_fn)
    self._first_stage_semantic_depth = first_stage_semantic_depth
    self._first_stage_semantic_kernel_size = \
        first_stage_semantic_kernel_size

    # (First stage) Region proposal network parameters
    self._first_stage_anchor_generator = first_stage_anchor_generator
    self._first_stage_atrous_rate = first_stage_atrous_rate
    self._first_stage_box_predictor_arg_scope_fn = (
        first_stage_box_predictor_arg_scope_fn)
    self._first_stage_box_predictor_kernel_size = (
        first_stage_box_predictor_kernel_size)
    self._first_stage_box_predictor_depth = first_stage_box_predictor_depth
    self._first_stage_minibatch_size = first_stage_minibatch_size
    self._first_stage_sampler = first_stage_sampler
    self._first_stage_box_predictor = (
        box_predictor_builder.build_convolutional_box_predictor(
            is_training=self._is_training,
            num_classes=1,
            conv_hyperparams_fn=self._first_stage_box_predictor_arg_scope_fn,
            use_dropout=False,
            dropout_keep_prob=1.0,
            box_code_size=self._box_coder.code_size,
            kernel_size=1,
            num_layers_before_predictor=0,
            min_depth=0,
            max_depth=0))

    self._first_stage_nms_fn = first_stage_non_max_suppression_fn
    self._first_stage_max_proposals = first_stage_max_proposals
    self._use_static_shapes = use_static_shapes

    self._first_stage_localization_loss = (
        losses.WeightedSmoothL1LocalizationLoss())
    self._first_stage_objectness_loss = (
        losses.WeightedSoftmaxClassificationLoss())
    self._first_stage_loc_loss_weight = first_stage_localization_loss_weight
    self._first_stage_obj_loss_weight = first_stage_objectness_loss_weight
    self._first_stage_sem_loss_weight = first_stage_semantic_loss_weight

    self._first_stage_fpn_min_level = first_stage_fpn_min_level
    self._first_stage_fpn_max_level = first_stage_fpn_max_level

    # Per-region cropping parameters
    self._crop_and_resize_fn = crop_and_resize_fn
    self._initial_crop_size = initial_crop_size
    self._maxpool_kernel_size = maxpool_kernel_size
    self._maxpool_stride = maxpool_stride

    self._mask_rcnn_box_predictor = second_stage_mask_rcnn_box_predictor

    self._second_stage_batch_size = second_stage_batch_size
    self._second_stage_sampler = second_stage_sampler

    self._second_stage_nms_fn = second_stage_non_max_suppression_fn
    self._second_stage_score_conversion_fn = second_stage_score_conversion_fn

    self._second_stage_localization_loss = (
        losses.WeightedSmoothL1LocalizationLoss())
    self._second_stage_classification_loss = second_stage_classification_loss
    self._second_stage_mask_loss = (
        losses.WeightedSigmoidClassificationLoss())
    self._second_stage_loc_loss_weight = second_stage_localization_loss_weight
    self._second_stage_cls_loss_weight = second_stage_classification_loss_weight
    self._second_stage_mask_loss_weight = (
        second_stage_mask_prediction_loss_weight)
    self._hard_example_miner = hard_example_miner
    self._parallel_iterations = parallel_iterations

    self.clip_anchors_to_image = clip_anchors_to_image

    self._anchors = None

    self._add_summaries = add_summaries

    # TODO: Need to be in config proto file.
    # _panoptic_score_threshold = threshold to consider a prediction valid.
    # _panoptic_mask_threshold = threshold to create binary mask.
    self._panoptic_score_threshold = 0.7
    self._panoptic_mask_threshold = 0.5

    if self._number_of_stages <= 0 or self._number_of_stages > 3:
      raise ValueError('Number of stages should be a value in {1, 2, 3}.')

  @property
  def first_stage_feature_extractor_scope(self):
    return 'FirstStageFeatureExtractor'

  @property
  def second_stage_feature_extractor_scope(self):
    return 'SecondStageFeatureExtractor'

  @property
  def first_stage_semantic_predictor_scope(self):
    return 'FirstStageSemanticPredictor'

  @property
  def first_stage_box_predictor_scope(self):
    return 'FirstStageBoxPredictor'

  @property
  def second_stage_box_predictor_scope(self):
    return 'SecondStageBoxPredictor'

  @property
  def max_num_proposals(self):
    """Max number of proposals (to pad to) for each image in the input batch.

    At training time, this is set to be the `second_stage_batch_size` if hard
    example miner is not configured, else it is set to
    `first_stage_max_proposals`. At inference time, this is always set to
    `first_stage_max_proposals`.

    Returns:
      A positive integer.
    """
    if self._is_training and not self._hard_example_miner:
      return self._second_stage_batch_size
    return self._first_stage_max_proposals

  @property
  def anchors(self):
    if not self._anchors:
      raise RuntimeError('anchors have not been constructed yet!')
    if not isinstance(self._anchors, box_list.BoxList):
      raise RuntimeError('anchors should be a BoxList object, but is not.')
    return self._anchors

  def preprocess(self, inputs):
    """Feature-extractor specific preprocessing.

    See base class.

    For Faster R-CNN, we perform image resizing in the base class --- each
    class subclassing FasterRCNNMetaArch is responsible for any additional
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
    with tf.name_scope('Preprocessor'):
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

    This function takes an input batch of images and runs it through the
    forward pass of the network to yield "raw" un-postprocessed predictions.
    If `number_of_stages` is 1, this function only returns first stage
    RPN predictions (un-postprocessed).  Otherwise it returns both
    first stage RPN predictions as well as second stage box classifier
    predictions.

    Other remarks:
    + Anchor pruning vs. clipping: following the recommendation of the Faster
    R-CNN paper, we prune anchors that venture outside the image window at
    training time and clip anchors to the image window at inference time.
    + Proposal padding: as described at the top of the file, proposals are
    padded to self._max_num_proposals and flattened so that proposals from all
    images within the input batch are arranged along the same batch dimension.

    Args:
      preprocessed_inputs: a [batch, height, width, channels] float tensor
        representing a batch of images.
      true_image_shapes: int32 tensor of shape [batch, 3] where each row is
        of the form [height, width, channels] indicating the shapes
        of true images in the resized images, as resized images can be padded
        with zeros.

    Returns:
      prediction_dict: a dictionary holding "raw" prediction tensors:
        1) rpn_box_predictor_features: A 4-D float32 tensor with shape
          [batch_size, height, width, depth] to be used for predicting proposal
          boxes and corresponding objectness scores.
        2) rpn_features_to_crop: A 4-D float32 tensor with shape
          [batch_size, height, width, depth] representing image features to crop
          using the proposal boxes predicted by the RPN.
        3) image_shape: a 1-D tensor of shape [4] representing the input
          image shape.
        4) rpn_box_encodings:  3-D float tensor of shape
          [batch_size, num_anchors, self._box_coder.code_size] containing
          predicted boxes.
        5) rpn_objectness_predictions_with_background: 3-D float tensor of shape
          [batch_size, num_anchors, 2] containing class
          predictions (logits) for each of the anchors.  Note that this
          tensor *includes* background class predictions (at class index 0).
        6) anchors: A 2-D tensor of shape [num_anchors, 4] representing anchors
          for the first stage RPN (in absolute coordinates).  Note that
          `num_anchors` can differ depending on whether the model is created in
          training or inference mode.

        (and if number_of_stages > 1):
        7) refined_box_encodings: a 3-D tensor with shape
          [total_num_proposals, num_classes, self._box_coder.code_size]
          representing predicted (final) refined box encodings, where
          total_num_proposals=batch_size*self._max_num_proposals. If using
          a shared box across classes the shape will instead be
          [total_num_proposals, 1, self._box_coder.code_size].
        8) class_predictions_with_background: a 3-D tensor with shape
          [total_num_proposals, num_classes + 1] containing class
          predictions (logits) for each of the anchors, where
          total_num_proposals=batch_size*self._max_num_proposals.
          Note that this tensor *includes* background class predictions
          (at class index 0).
        9) num_proposals: An int32 tensor of shape [batch_size] representing the
          number of proposals generated by the RPN.  `num_proposals` allows us
          to keep track of which entries are to be treated as zero paddings and
          which are not since we always pad the number of proposals to be
          `self.max_num_proposals` for each image.
        10) proposal_boxes: A float32 tensor of shape
          [batch_size, self.max_num_proposals, 4] representing
          decoded proposal bounding boxes in absolute coordinates.
        11) mask_predictions: (optional) a 4-D tensor with shape
          [total_num_padded_proposals, num_classes, mask_height, mask_width]
          containing instance mask predictions.

    Raises:
      ValueError: If `predict` is called before `preprocess`.
    """
    ############################################################################
    # START HERE                                                               #
    ############################################################################
    # [DummyCode] **************************************************************
    # DEBUG: Dummy code to test resnet 101 backbone.
    # def _filter_features(image_features):
    #     _keymap = {
    #         'block1': 'C2',
    #         'block2': 'C3',
    #         'block3': 'C4',
    #         'block4': 'C5',
    #     }
    #     filtered_image_features = []
    #     for key, feature in image_features.items():
    #         feature_name = key.split('/')[-1]
    #         if feature_name in ['block1', 'block2', 'block3', 'block4']:
    #             filtered_image_features.append(
    #                 [_keymap[feature_name], feature])
    #     return filtered_image_features
    #
    # with slim.arg_scope(
    #         resnet_utils.resnet_arg_scope(
    #             batch_norm_epsilon=1e-5,
    #             batch_norm_scale=True,
    #             weight_decay=0.0)):
    #     with tf.variable_scope('resnet_v1_101', reuse=False) as var_scope:
    #         _, image_features_btm_up = resnet_v1.resnet_v1_101(
    #             preprocessed_inputs,
    #             num_classes=None,
    #             is_training=False,  # Is used for BN.
    #             global_pool=False,
    #             output_stride=32,
    #             spatial_squeeze=False,
    #             scope=var_scope)
    # image_features_btm_up = _filter_features(image_features_btm_up)
    # dummy_predictor = slim.fully_connected(
    #     image_features_btm_up[-1][1],
    #     self.num_classes,
    #     activation_fn=None,
    #     scope='Dummy_Predictor')
    #
    # prediction_dict = {'final_top_down_feature': image_features_btm_up,
    #                    'dummy_predictor': dummy_predictor}
    # return prediction_dict
    # ************************************************************** [DummyCode]

    # ======================================================================== #
    # (1.) BASE AND RPN FEATURE EXTRACTOR BACKBONE                             #
    # ======================================================================== #
    # ANCHORS ARE IN ABSOLUTE COORDINATES.
    (rpn_box_predictor_features, rpn_features_to_crop, anchors_boxlist,
     image_shape, final_top_down_feature
     ) = self._extract_rpn_feature_maps(preprocessed_inputs)

    prediction_dict = {
        'final_top_down_feature': final_top_down_feature,
        'rpn_box_predictor_features': rpn_box_predictor_features,
        'rpn_features_to_crop': rpn_features_to_crop,
        'image_shape': image_shape,
    }

    # TODO: Hack to get original image. VGG style mean.
    summary_image_ori = preprocessed_inputs + [[123.68, 116.779, 103.939]]
    prediction_dict['summary_images'] = summary_image_ori

    if fields.InputDataFields.original_image in params:
        prediction_dict[fields.InputDataFields.original_image] = \
            params[fields.InputDataFields.original_image]

    # ======================================================================== #
    # (2.) Semantic label                                                      #
    # ======================================================================== #
    if Options.semantic_segmentation:
        prediction_dict['semantic_predictions'] = \
            self._predict_semantic_labels(final_top_down_feature)
        # To add into summary only.
        prediction_dict['semantic_predictions'] = tf.identity(
            prediction_dict['semantic_predictions'],
            name='summary_logits_prediction')

    # ======================================================================== #
    # (3.) RPN -> 4*k REG + 2*k CLS                                            #
    # ======================================================================== #
    # 'rpn_box_obj' = ['rpn_box_encodings' ,
    #                  'rpn_objectness_predictions_with_background']
    rpn_box_obj = self._predict_rpn_proposals(rpn_box_predictor_features)

    if self._mask_rcnn_box_predictor.get_prediction_head_backbone() is None \
            or self._mask_rcnn_box_predictor.get_prediction_head_backbone() \
            == 'C4':
        rpn_objectness_predictions_with_background = []
        rpn_box_encodings = []
        # The Faster R-CNN paper recommends pruning anchors that venture outside
        # the image window at training time and clipping at inference time.
        clip_window = tf.to_float(
            tf.stack([0, 0, image_shape[1], image_shape[2]]))
        if self._is_training:
            if self.clip_anchors_to_image:
                anchors_boxlist = box_list_ops.clip_to_window(
                    anchors_boxlist, clip_window, filter_nonoverlapping=False)
            else:
                (rpn_box_encodings, rpn_objectness_predictions_with_background,
                 anchors_boxlist) = self._remove_invalid_anchors_and_predictions(
                    rpn_box_obj[0],
                    rpn_box_obj[1],
                    anchors_boxlist, clip_window)
        else:
            anchors_boxlist = box_list_ops.clip_to_window(
                anchors_boxlist, clip_window)

        self._anchors = anchors_boxlist
        prediction_dict['rpn_box_encodings'] = rpn_box_encodings
        prediction_dict['rpn_objectness_predictions_with_background'] = \
            rpn_objectness_predictions_with_background
        prediction_dict['anchors'] = anchors_boxlist.get()

    elif self._mask_rcnn_box_predictor.get_prediction_head_backbone() == 'FPN':
        rpn_objectness_predictions_with_background = []
        rpn_box_encodings = []
        _anchors_boxlist = []
        for box_obj, anchor in zip(rpn_box_obj, anchors_boxlist):
            # The Faster R-CNN paper recommends pruning anchors that venture
            # outside the image window at training time and clipping at
            # inference time.
            clip_window = tf.cast(
                tf.stack([0, 0, image_shape[1], image_shape[2]]),
                dtype=tf.float32)
            single_level_rpn_box_encodings = box_obj[0]
            single_level_rpn_objectness_predictions_with_background = box_obj[1]
            if self._is_training:
                if self.clip_anchors_to_image:
                    anchor = box_list_ops.clip_to_window(
                        anchor, clip_window, filter_nonoverlapping=False)
                else:
                    (single_level_rpn_box_encodings,
                     single_level_rpn_objectness_predictions_with_background,
                     anchor
                     ) = self._remove_invalid_anchors_and_predictions(
                        single_level_rpn_box_encodings,
                        single_level_rpn_objectness_predictions_with_background,
                        anchor, clip_window)
            else:
                anchor = box_list_ops.clip_to_window(anchor, clip_window)
            _anchors_boxlist.append(anchor)
            rpn_box_encodings.append(single_level_rpn_box_encodings)
            rpn_objectness_predictions_with_background.append(
                single_level_rpn_objectness_predictions_with_background)

        anchors = []
        for anchor in _anchors_boxlist:
            anchors.append(anchor.get())
        self._anchors = anchors

        # # Stacking predictions, encodings, anchors into a single list
        # # respectively to be used by the F-RCNN/M-RCNN heads.
        # anchors = tf.concat(anchors, axis=0)
        # rpn_box_encodings = tf.concat(rpn_box_encodings, axis=1)
        # rpn_objectness_predictions_with_background = tf.concat(
        #     rpn_objectness_predictions_with_background, axis=1)

        prediction_dict['rpn_box_encodings'] = rpn_box_encodings
        prediction_dict['rpn_objectness_predictions_with_background'] = \
            rpn_objectness_predictions_with_background
        prediction_dict['anchors'] = anchors
        prediction_dict['anchors_boxlist'] = _anchors_boxlist
    else:
      raise ValueError("Backbone is not supported...")

    # ======================================================================== #
    # (4.) F-RCNN / M-RCNN PREDICTION HEAD                                     #
    # ======================================================================== #
    # FPN MODIFICATION:
    # - box encoding = x,y,dx,dy for each level is fused to form a list.
    # - objectness prediction and anchors also.
    # - features to crop is not fused into a single list.
    if self._number_of_stages >= 2:
        prediction_dict.update(self._predict_second_stage(
            summary_image_ori,
            rpn_box_encodings,
            rpn_objectness_predictions_with_background,
            rpn_features_to_crop,
            self._anchors, image_shape, true_image_shapes))

    # Training: Sub-sampling to obtain the max_batch_size bb to compute
    # the ROI needed to perform the mask prediction
    # Inference: Post-processing to obtain the top k bb to compute the ROI
    # needed to perform the mask prediction
    if self._number_of_stages == 3:
        prediction_dict = self._predict_third_stage(
            prediction_dict, image_shape, true_image_shapes)

    # ======================================================================== #
    # (5.) Viewing Results (Taken from the loss_box_classifier(...))           #
    # ======================================================================== #
    # if self._is_training and True:
    if self._is_training or self._is_evaluating:
        (groundtruth_boxlists, groundtruth_classes_with_background_list,
         groundtruth_masks_list, groundtruth_weights_list) = \
            self._format_groundtruth_data(true_image_shapes)

        images = prediction_dict['summary_images']
        refined_box_encodings = prediction_dict['refined_box_encodings']
        class_predictions_with_background = prediction_dict[
            'class_predictions_with_background']
        proposal_boxes = prediction_dict['proposal_boxes']
        num_proposals = prediction_dict['num_proposals']
        groundtruth_boxlists = groundtruth_boxlists
        groundtruth_classes_with_background_list = \
            groundtruth_classes_with_background_list
        image_shape = prediction_dict['image_shape']
        prediction_masks = prediction_dict.get(box_predictor.MASK_PREDICTIONS)
        groundtruth_masks_list = groundtruth_masks_list

        if prediction_masks is None:
            tf.logging.info("prediction_masks is not found...")
            return prediction_dict

        if self._is_training:
            max_num_proposals = proposal_boxes.shape[1].value
        else:
            max_num_proposals = tf.shape(proposal_boxes)[1]
        proposal_boxlists = [
            box_list.BoxList(proposal_boxes_single_image)
            for proposal_boxes_single_image in tf.unstack(proposal_boxes)]
        batch_size = len(proposal_boxlists)

        targets = target_assigner.batch_assign_targets(
            target_assigner=self._detector_target_assigner,
            anchors_batch=proposal_boxlists, gt_box_batch=groundtruth_boxlists,
            gt_class_targets_batch=groundtruth_classes_with_background_list,
            unmatched_class_label=tf.constant([1] + self._num_classes * [0],
                                              dtype=tf.float32),
            gt_weights_batch=groundtruth_weights_list)
        (batch_cls_targets_with_background, batch_cls_weights,
         batch_reg_targets, batch_reg_weights, _) = targets

        class_predictions_with_background = tf.reshape(
            class_predictions_with_background,
            [batch_size, max_num_proposals, self.num_classes + 1])

        flat_cls_targets_with_background = tf.reshape(
            batch_cls_targets_with_background,
            [batch_size * max_num_proposals, -1])
        one_hot_flat_cls_targets_with_background = tf.argmax(
            flat_cls_targets_with_background, axis=1)
        if self._is_training:
            one_hot_flat_cls_targets_with_background = tf.one_hot(
                one_hot_flat_cls_targets_with_background,
                flat_cls_targets_with_background.get_shape()[1])
        else:
            one_hot_flat_cls_targets_with_background = tf.one_hot(
                one_hot_flat_cls_targets_with_background,
                tf.shape(flat_cls_targets_with_background)[1])

        # If using a shared box across classes use directly
        if refined_box_encodings.shape[1] == 1:
            reshaped_refined_box_encodings = tf.reshape(
                refined_box_encodings,
                [batch_size, max_num_proposals, self._box_coder.code_size])
        # For anchors with multiple labels, picks refined_location_encodings
        # for just one class to avoid over-counting for regression loss and
        # (optionally) mask loss.
        else:
            # We only predict refined location encodings for the non
            # background classes, but we now pad it to make it compatible
            # with the class predictions
            refined_box_encodings_with_background = tf.pad(
                refined_box_encodings, [[0, 0], [1, 0], [0, 0]])
            refined_box_encodings_masked_by_class_targets = tf.boolean_mask(
                refined_box_encodings_with_background,
                tf.greater(one_hot_flat_cls_targets_with_background, 0))
            reshaped_refined_box_encodings = tf.reshape(
                refined_box_encodings_masked_by_class_targets,
                [batch_size, max_num_proposals, self._box_coder.code_size])

        # For the mask
        unmatched_mask_label = tf.zeros(image_shape[1:3], dtype=tf.float32)
        (batch_mask_targets, _, _, batch_mask_target_weights,
         _) = target_assigner.batch_assign_targets(
             target_assigner=self._detector_target_assigner,
             anchors_batch=proposal_boxlists,
             gt_box_batch=groundtruth_boxlists,
             gt_class_targets_batch=groundtruth_masks_list,
             unmatched_class_label=unmatched_mask_label,
             gt_weights_batch=groundtruth_weights_list)

        num_classes = prediction_masks.shape[1].value
        mask_height = prediction_masks.shape[2].value
        mask_width = prediction_masks.shape[3].value
        # Pad the prediction_masks with to add zeros for background
        # class to be consistent with class predictions.
        if num_classes == 1:
            # Class agnostic masks or masks for one-class prediction.
            # Logic for both cases is the same since background
            # predictions are ignored through the batch_mask_target
            # weights.
            prediction_masks_masked_by_class_targets = prediction_masks
        else:
            prediction_masks_with_background = tf.pad(
                prediction_masks, [[0, 0], [1, 0], [0, 0], [0, 0]])
            prediction_masks_masked_by_class_targets = tf.boolean_mask(
                prediction_masks_with_background,
                tf.greater(one_hot_flat_cls_targets_with_background, 0))
            if Options.masks_prediction_full:
                _img = prediction_masks_with_background
                _img = tf.reshape(_img,
                                  [-1, mask_height,
                                   mask_width * (num_classes + 1)])
                _img = tf.reshape(_img,
                                  [-1, mask_width * (num_classes + 1)])
                _img = tf.expand_dims(_img, axis=-1)
                _img = tf.expand_dims(_img, axis=0)
                tf.summary.image("masks_prediction_full",
                                 tf.cast(_img, dtype=tf.float32))

        reshaped_prediction_masks = tf.reshape(
            prediction_masks_masked_by_class_targets,
            [batch_size, -1, mask_height * mask_width])

        batch_mask_targets_shape = tf.shape(batch_mask_targets)
        flat_gt_masks = tf.reshape(batch_mask_targets,
                                   [-1, batch_mask_targets_shape[2],
                                    batch_mask_targets_shape[3]])

        # Use normalized proposals to crop mask targets from image masks.
        flat_normalized_proposals = box_list_ops.to_normalized_coordinates(
            box_list.BoxList(tf.reshape(proposal_boxes, [-1, 4])),
            image_shape[1], image_shape[2]).get()

        if self._is_training:
            total_proposals = flat_normalized_proposals.shape[0].value
        else:
            total_proposals = tf.shape(flat_normalized_proposals)[0]
        flat_cropped_gt_mask = tf.image.crop_and_resize(
            tf.expand_dims(flat_gt_masks, -1),
            tf.cast(flat_normalized_proposals, dtype=tf.float32),
            tf.range(total_proposals),
            [mask_height, mask_width])

        batch_cropped_gt_mask = tf.reshape(
            flat_cropped_gt_mask,
            [batch_size, -1, mask_height * mask_width])

        with tf.name_scope('Predictions'):
            params = (images, image_shape,
                      batch_reg_targets, reshaped_refined_box_encodings,
                      batch_cls_targets_with_background,
                      class_predictions_with_background,
                      batch_cropped_gt_mask,
                      batch_mask_targets,  # Uncropped mask.
                      reshaped_prediction_masks,
                      groundtruth_boxlists, proposal_boxlists,
                      num_proposals,
                      flat_gt_masks, prediction_masks_masked_by_class_targets,
                      mask_width)
            self._add_summary_prediction_image(params)

        # Viewing the segmentation results.
        if Options.semantic_segmentation:
            with tf.name_scope('Segmentation'):
                params = (prediction_dict['semantic_predictions'])
                self._add_summary_semantic_image(params)

    return prediction_dict

  def _predict_second_stage(self,
                            images,
                            rpn_box_encodings,
                            rpn_objectness_predictions_with_background,
                            rpn_features_to_crop,
                            anchors,
                            image_shape,
                            true_image_shapes):
    """Predicts the output tensors from second stage of Faster R-CNN.

    Args:
      rpn_box_encodings: 4-D float tensor of shape
        [batch_size, num_valid_anchors, self._box_coder.code_size] containing
        predicted boxes.
      rpn_objectness_predictions_with_background: 2-D float tensor of shape
        [batch_size, num_valid_anchors, 2] containing class
        predictions (logits) for each of the anchors.  Note that this
        tensor *includes* background class predictions (at class index 0).
      rpn_features_to_crop: A 4-D float32 tensor with shape
        [batch_size, height, width, depth] representing image features to crop
        using the proposal boxes predicted by the RPN.
      anchors: 2-D float tensor of shape
        [num_anchors, self._box_coder.code_size].
      image_shape: A 1D int32 tensors of size [4] containing the image shape.
      true_image_shapes: int32 tensor of shape [batch, 3] where each row is
        of the form [height, width, channels] indicating the shapes
        of true images in the resized images, as resized images can be padded
        with zeros.

    Returns:
      prediction_dict: a dictionary holding "raw" prediction tensors:
        1) refined_box_encodings: a 3-D tensor with shape
          [total_num_proposals, num_classes, self._box_coder.code_size]
          representing predicted (final) refined box encodings, where
          total_num_proposals=batch_size*self._max_num_proposals. If using a
          shared box across classes the shape will instead be
          [total_num_proposals, 1, self._box_coder.code_size].
        2) class_predictions_with_background: a 3-D tensor with shape
          [total_num_proposals, num_classes + 1] containing class
          predictions (logits) for each of the anchors, where
          total_num_proposals=batch_size*self._max_num_proposals.
          Note that this tensor *includes* background class predictions
          (at class index 0).
        3) num_proposals: An int32 tensor of shape [batch_size] representing the
          number of proposals generated by the RPN.  `num_proposals` allows us
          to keep track of which entries are to be treated as zero paddings and
          which are not since we always pad the number of proposals to be
          `self.max_num_proposals` for each image.
        4) proposal_boxes: A float32 tensor of shape
          [batch_size, self.max_num_proposals, 4] representing
          decoded proposal bounding boxes in absolute coordinates.
        5) proposal_boxes_normalized: A float32 tensor of shape
          [batch_size, self.max_num_proposals, 4] representing decoded proposal
          bounding boxes in normalized coordinates. Can be used to override the
          boxes proposed by the RPN, thus enabling one to extract features and
          get box classification and prediction for externally selected areas
          of the image.
        6) box_classifier_features: a 4-D float32 tensor representing the
          features for each proposal.
    """
    # ======================================================================== #
    # (3.1.) NORM + NMS                                                        #
    # Normalize and run NMS on the raw proposal prediction to scale down       #
    # the number of proposal that will be used.                                #
    # ======================================================================== #
    image_shape_2d = lib.image_batch_shape_2d(image_shape)
    proposal_boxes_normalized, proposal_scores, num_proposals = \
        self._postprocess_rpn(images, rpn_box_encodings,
                              rpn_objectness_predictions_with_background,
                              anchors, image_shape_2d, true_image_shapes)

    if Options.rpn_proposals:
        lib.draw_results(images,
                         proposal_boxes_normalized,
                         Options.object_index,
                         classes=None,
                         scores=None,
                         instance_masks=None,
                         min_score_thresh=0.1,
                         max_boxes_to_draw=2000,
                         use_normalized_coordinates=True,
                         name='rpn_proposals')

    # The proposals are concatenated over all the batches, causing the shape
    # here to become [batch x max_num_proposals, height, width, depth]
    flattened_proposal_feature_maps = (
        self._compute_second_stage_input_feature_maps(
            rpn_features_to_crop, proposal_boxes_normalized,
            image_shape_2d))

    # ======================================================================== #
    # (3.2.) ROI + Realignment + Assignment                                    #
    # ======================================================================== #
    # ======================================================================== #
    # (3.3.) ResNetV1101 Block4 or None for FPN                                #
    # ======================================================================== #
    if self._mask_rcnn_box_predictor.get_prediction_head_backbone() is None \
            or self._mask_rcnn_box_predictor.get_prediction_head_backbone() \
            == 'C4':
      box_classifier_features = (
          self._feature_extractor.extract_box_classifier_features(
              flattened_proposal_feature_maps,
              scope=self.second_stage_feature_extractor_scope))
      pre_split_box_classifier_features = box_classifier_features
    elif self._mask_rcnn_box_predictor.get_prediction_head_backbone() == 'FPN':
      box_classifier_features = slim.max_pool2d(
          flattened_proposal_feature_maps,
          [self._maxpool_kernel_size, self._maxpool_kernel_size],
          stride=self._maxpool_stride)
      pre_split_box_classifier_features = flattened_proposal_feature_maps
      # tmp_shape = tf.shape(box_classifier_features)
      # box_classifier_features = tf.reshape(
      #     box_classifier_features,
      #     [image_shape[0], -1, tmp_shape[1], tmp_shape[2], tmp_shape[3]])
    else:
        raise ValueError("Backbone is not supported...")

    # ======================================================================== #
    # (3.4.) PREDICTION CLS + BB + MASK                                        #
    #  - size is equal to the max proposal for stage 2 in config.              #
    # ======================================================================== #

    if self._mask_rcnn_box_predictor.is_keras_model:
      box_predictions = self._mask_rcnn_box_predictor(
          [box_classifier_features],
          prediction_stage=2)
    else:
      box_predictions = self._mask_rcnn_box_predictor.predict(
          [box_classifier_features],
          num_predictions_per_location=[1],
          scope=self.second_stage_box_predictor_scope,
          prediction_stage=2)

    refined_box_encodings = tf.squeeze(
        box_predictions[box_predictor.BOX_ENCODINGS],
        axis=1, name='all_refined_box_encodings')
    class_predictions_with_background = tf.squeeze(
        box_predictions[box_predictor.CLASS_PREDICTIONS_WITH_BACKGROUND],
        axis=1, name='all_class_predictions_with_background')

    absolute_proposal_boxes = ops.normalized_to_image_coordinates(
        proposal_boxes_normalized, image_shape, self._parallel_iterations)

    # MixPrecision Box and Class
    refined_box_encodings = tf.cast(refined_box_encodings, dtype=tf.float32)
    class_predictions_with_background = tf.cast(
        class_predictions_with_background, dtype=tf.float32)
    prediction_dict = {
        'refined_box_encodings': refined_box_encodings,
        'class_predictions_with_background':
        class_predictions_with_background,
        'num_proposals': num_proposals,
        'proposal_boxes': absolute_proposal_boxes,
        'box_classifier_features': pre_split_box_classifier_features,
        'proposal_boxes_normalized': proposal_boxes_normalized,
    }

    return prediction_dict

  def _predict_third_stage(self, prediction_dict,
                           image_shapes, true_image_shapes):
    """Predicts non-box, non-class outputs using refined detections.

    For training, masks as predicted directly on the box_classifier_features,
    which are region-features from the initial anchor boxes.
    For inference, this happens after calling the post-processing stage, such
    that masks are only calculated for the top scored boxes.

    Args:
     prediction_dict: a dictionary holding "raw" prediction tensors:
        1) refined_box_encodings: a 3-D tensor with shape
          [total_num_proposals, num_classes, self._box_coder.code_size]
          representing predicted (final) refined box encodings, where
          total_num_proposals=batch_size*self._max_num_proposals. If using a
          shared box across classes the shape will instead be
          [total_num_proposals, 1, self._box_coder.code_size].
        2) class_predictions_with_background: a 3-D tensor with shape
          [total_num_proposals, num_classes + 1] containing class
          predictions (logits) for each of the anchors, where
          total_num_proposals=batch_size*self._max_num_proposals.
          Note that this tensor *includes* background class predictions
          (at class index 0).
        3) num_proposals: An int32 tensor of shape [batch_size] representing the
          number of proposals generated by the RPN.  `num_proposals` allows us
          to keep track of which entries are to be treated as zero paddings and
          which are not since we always pad the number of proposals to be
          `self.max_num_proposals` for each image.
        4) proposal_boxes: A float32 tensor of shape
          [batch_size, self.max_num_proposals, 4] representing
          decoded proposal bounding boxes in absolute coordinates.
        5) box_classifier_features: a 4-D float32 tensor representing the
          features for each proposal.
      image_shapes: A 2-D int32 tensors of shape [batch_size, 3] containing
        shapes of images in the batch.

    Returns:
      prediction_dict: a dictionary that in addition to the input predictions
      does hold the following predictions as well:
        1) mask_predictions: a 4-D tensor with shape
          [batch_size, max_detection, mask_height, mask_width] containing
          instance mask predictions.
    """
    image_shape_2d = lib.image_batch_shape_2d(image_shapes)
    if self._is_training:
      curr_box_classifier_features = prediction_dict['box_classifier_features']
      # detection_classes = prediction_dict['class_predictions_with_background']
      if self._mask_rcnn_box_predictor.is_keras_model:
        mask_predictions = self._mask_rcnn_box_predictor(
            [curr_box_classifier_features],
            prediction_stage=3)
      else:
        mask_predictions = self._mask_rcnn_box_predictor.predict(
            [curr_box_classifier_features],
            num_predictions_per_location=[1],
            scope=self.second_stage_box_predictor_scope,
            prediction_stage=3)
      # MixPrecision
      prediction_dict[box_predictor.MASK_PREDICTIONS] = tf.cast(
          tf.squeeze(
              mask_predictions[box_predictor.MASK_PREDICTIONS], axis=1),
          dtype=tf.float32)
    else:
      # prediction_dicts are a list of different pyramid levels.
      detections_dict = self._postprocess_box_classifier(
          prediction_dict['refined_box_encodings'],
          prediction_dict['class_predictions_with_background'],
          prediction_dict['proposal_boxes'],
          prediction_dict['num_proposals'],
          true_image_shapes)
      prediction_dict.update(detections_dict)
      detection_boxes = detections_dict[
          fields.DetectionResultFields.detection_boxes]
      detection_classes = detections_dict[
          fields.DetectionResultFields.detection_classes]
      rpn_features_to_crop = prediction_dict['rpn_features_to_crop']
      batch_size = tf.shape(detection_boxes)[0]
      # max_detection = tf.shape(detection_boxes)[1]
      flattened_detected_feature_maps = (
          self._compute_second_stage_input_feature_maps(
              rpn_features_to_crop, detection_boxes, image_shape_2d))

      if self._mask_rcnn_box_predictor.get_prediction_head_backbone() is None \
              or self._mask_rcnn_box_predictor.get_prediction_head_backbone() \
              == 'C4':
        # For C4 backbone only
        curr_box_classifier_features = (
            self._feature_extractor.extract_box_classifier_features(
                flattened_detected_feature_maps,
                scope=self.second_stage_feature_extractor_scope))
      elif self._mask_rcnn_box_predictor.get_prediction_head_backbone() \
              == 'FPN':
        curr_box_classifier_features = flattened_detected_feature_maps
      else:
          raise ValueError("Backbone is not supported...")

      if self._mask_rcnn_box_predictor.is_keras_model:
        mask_predictions = self._mask_rcnn_box_predictor(
            [curr_box_classifier_features],
            prediction_stage=3)
      else:
        mask_predictions = self._mask_rcnn_box_predictor.predict(
            [curr_box_classifier_features],
            num_predictions_per_location=[1],
            scope=self.second_stage_box_predictor_scope,
            prediction_stage=3)

      prediction_masks = tf.squeeze(
          mask_predictions[box_predictor.MASK_PREDICTIONS], axis=1)

      _, num_classes, mask_height, mask_width = (
          prediction_masks.get_shape().as_list())
      _, max_detection = detection_classes.get_shape().as_list()
      if num_classes > 1:
        prediction_masks = lib.gather_instance_masks(
            prediction_masks, detection_classes)

      prediction_masks = tf.reshape(
          prediction_masks, [batch_size, max_detection, mask_height,
                             mask_width])

      # MixPrecision
      prediction_dict[box_predictor.MASK_PREDICTIONS] = tf.cast(
          prediction_masks, dtype=tf.float32)

    return prediction_dict

  def _extract_rpn_feature_maps(self, preprocessed_inputs):
    """Extracts RPN features.

    This function extracts two feature maps: a feature map to be directly
    fed to a box predictor (to predict location and objectness scores for
    proposals) and a feature map from which to crop regions which will then
    be sent to the second stage box classifier.

    Args:
      preprocessed_inputs: a [batch, height, width, channels] image tensor.

    Returns:
      rpn_box_predictor_features: A 4-D float32 tensor with shape
        [batch, height, width, depth] to be used for predicting proposal boxes
        and corresponding objectness scores.
      rpn_features_to_crop: A 4-D float32 tensor with shape
        [batch, height, width, depth] representing image features to crop using
        the proposals boxes.
      anchors: A BoxList representing anchors (for the RPN) in
        absolute coordinates.
      image_shape: A 1-D tensor representing the input image shape.
    """
    image_shape = tf.shape(preprocessed_inputs)

    # ======================================================================== #
    # (1.1.) IMAGE FEATURE GENERATOR                                           #
    # ======================================================================== #
    # Here the 'rpn_features_to_crop' is a list of features from different
    # pyramid levels. P2 - P6
    top_down_feature = self._feature_extractor.extract_decoder_features(
        preprocessed_inputs,
        scope=self.first_stage_feature_extractor_scope)
    (rpn_features_to_crop, final_top_down_feature) = \
        self._feature_extractor.extract_proposal_features(
            top_down_feature,
            scope=self.first_stage_feature_extractor_scope)

    # ======================================================================== #
    # (1.2.) RPN FEATURE GENERATOR                                             #
    # ======================================================================== #
    if self._mask_rcnn_box_predictor.get_prediction_head_backbone() is None \
            or self._mask_rcnn_box_predictor.get_prediction_head_backbone() \
            == 'C4':
      feature_map_shape = tf.shape(rpn_features_to_crop)
      anchors = box_list_ops.concatenate(
          self._first_stage_anchor_generator.generate(
              [(feature_map_shape[1], feature_map_shape[2])]))
      with slim.arg_scope(self._first_stage_box_predictor_arg_scope_fn()):
          kernel_size = self._first_stage_box_predictor_kernel_size
          reuse = tf.get_variable_scope().reuse
          rpn_box_predictor_features = slim.conv2d(
              rpn_features_to_crop,
              self._first_stage_box_predictor_depth,
              kernel_size=[kernel_size, kernel_size],
              rate=self._first_stage_atrous_rate,
              activation_fn=tf.nn.relu6,
              scope='Conv',
              reuse=reuse)
      return (rpn_box_predictor_features, rpn_features_to_crop,
              anchors, image_shape)
    elif self._mask_rcnn_box_predictor.get_prediction_head_backbone() == 'FPN':
      rpn_box_predictor_features = []
      feature_map_shape_list = []
      kernel_size = self._first_stage_box_predictor_kernel_size
      # Base RPN conv is shared among the levels.
      with tf.variable_scope('ConvBaseRPN', reuse=tf.AUTO_REUSE):
        with slim.arg_scope(self._first_stage_box_predictor_arg_scope_fn()):
          for rpn_feature in rpn_features_to_crop:
            # PRN feature maps for prediction.
            _base_rpn_feature = slim.conv2d(
                rpn_feature,
                self._first_stage_box_predictor_depth,
                kernel_size=[kernel_size, kernel_size],
                rate=self._first_stage_atrous_rate,
                activation_fn=tf.nn.relu6,
                scope="BaseRPNFeatures")
            rpn_box_predictor_features.append(_base_rpn_feature)
            # Feature map size for anchor generation.
            feature_map_shape = tf.shape(rpn_feature)
            feature_map_shape_list.append((feature_map_shape[1],
                                           feature_map_shape[2]))
      anchor_grid_list = self._first_stage_anchor_generator.generate(
          feature_map_shape_list, im_height=1, im_width=1)
      anchors = anchor_grid_list

      # anchors = []
      # for bl in anchor_grid_list:
      #     boxes = tf.cast(bl.get_field('boxes'), fp_dtype)
      #     feature_map_index = tf.cast(bl.get_field('feature_map_index'),
      #                                 fp_dtype)
      #     bl.set_field('boxes', boxes)
      #     bl.set_field('feature_map_index', feature_map_index)
      #     anchors.append(bl)

      return (rpn_box_predictor_features, rpn_features_to_crop, anchors,
              image_shape, final_top_down_feature)
    else:
      raise ValueError("Backbone is not supported...")

  def _predict_rpn_proposals(self, rpn_box_predictor_features):
    """Adds box predictors to RPN feature map to predict proposals.

    Note resulting tensors will not have been postprocessed.

    Args:
      rpn_box_predictor_features: A 4-D float32 tensor with shape
        [batch, height, width, depth] to be used for predicting proposal boxes
        and corresponding objectness scores.

    Returns:
      box_encodings: 3-D float tensor of shape
        [batch_size, num_anchors, self._box_coder.code_size] containing
        predicted boxes.
      objectness_predictions_with_background: 3-D float tensor of shape
        [batch_size, num_anchors, 2] containing class
        predictions (logits) for each of the anchors.  Note that this
        tensor *includes* background class predictions (at class index 0).

    Raises:
      RuntimeError: if the anchor generator generates anchors corresponding to
        multiple feature maps.  We currently assume that a single feature map
        is generated for the RPN.
    """
    if self._mask_rcnn_box_predictor.get_prediction_head_backbone() is None \
            or self._mask_rcnn_box_predictor.get_prediction_head_backbone() \
            == 'C4':
      num_anchors_per_location = (
          self._first_stage_anchor_generator.num_anchors_per_location())
      if len(num_anchors_per_location) != 1:
        raise RuntimeError('anchor_generator is expected to generate anchors '
                           'corresponding to a single feature map.')
      if self._first_stage_box_predictor.is_keras_model:
        box_predictions = self._first_stage_box_predictor(
            [rpn_box_predictor_features])
      else:
        box_predictions = self._first_stage_box_predictor.predict(
            [rpn_box_predictor_features],
            num_anchors_per_location,
            scope=self.first_stage_box_predictor_scope)
      box_encodings = tf.concat(
          box_predictions[box_predictor.BOX_ENCODINGS], axis=1)
      objectness_predictions_with_background = tf.concat(
          box_predictions[box_predictor.CLASS_PREDICTIONS_WITH_BACKGROUND],
          axis=1)
      return (tf.squeeze(box_encodings, axis=2),
              objectness_predictions_with_background)
    elif self._mask_rcnn_box_predictor.get_prediction_head_backbone() == 'FPN':
      num_anchors_per_location = (
          self._first_stage_anchor_generator.num_anchors_per_location())
      # RPN prediction is shared among the levels.
      # Returns a list of tuples of (cls tensor, reg tensor).
      # tensor size determines the number of anchor predictions.
      box_encodings_objectness_predictions_with_background = []
      with tf.variable_scope('PredictionRPN', reuse=tf.AUTO_REUSE):
        for feature, num_anchor in zip(
                rpn_box_predictor_features, num_anchors_per_location):
          box_predictions = self._first_stage_box_predictor.predict(
              [feature],
              [num_anchor],
              scope=self.first_stage_box_predictor_scope)
          box_encodings = tf.concat(
              box_predictions[box_predictor.BOX_ENCODINGS], axis=1)
          objectness_predictions_with_background = tf.concat(
              box_predictions[
                  box_predictor.CLASS_PREDICTIONS_WITH_BACKGROUND],
              axis=1)
          # # MixPrecision
          # box_encodings = tf.cast(box_encodings, dtype=tf.float32)
          # objectness_predictions_with_background = tf.cast(
          #     objectness_predictions_with_background, dtype=tf.float32)
          box_encodings_objectness_predictions_with_background.append(
              (tf.squeeze(box_encodings, axis=2),
               objectness_predictions_with_background))
      return box_encodings_objectness_predictions_with_background
    else:
      raise ValueError("Backbone is not supported...")

  def _predict_semantic_labels(self, semantic_predictor_features):
    """Uses the last layer lowest feature level of the FPN to predict the
    semantic labels.

    Note resulting tensors are not postprocessed.

    Args:
        semantic_predictor_features: A 4-D float32 tensor with shape
            [batch, height, width, depth] to be used for predicting
            semantic labels / scores.

    Returns:
        semantic_logits: 3-D float tensor of shape
            [batch, height, width, num_semantic_classes] containing
            semantic class predictions (logits) for each pixels.

    Raises:
        RuntimeError: if the anchor generator generates anchors
            corresponding to multiple feature maps. We currently assume
            that a single feature map is generated for the RPN.
    """
    # TODO 1: We need to define this in proto.
    # TODO 2: Maybe we need to include the 2 separable conv layers as in
    # deeplab V3+.
    with slim.arg_scope(self._first_stage_semantic_arg_scope_fn()):
        with tf.variable_scope(self.first_stage_semantic_predictor_scope,
                               self.first_stage_semantic_predictor_scope,
                               [semantic_predictor_features]):
            num_convs = 2
            decoder_features = slim.repeat(
                semantic_predictor_features,
                num_convs,
                slim.conv2d,
                self._first_stage_semantic_depth,
                self._first_stage_semantic_kernel_size,
                scope='decoder_conv')
            # with tf.name_scope('decoder_conv'):
            #     decoder_features = slim.conv2d(
            #         semantic_predictor_features,
            #         self._first_stage_semantic_depth,
            #         kernel_size=self._first_stage_semantic_kernel_size,
            #         activation_fn=tf.nn.relu,
            #         scope='decoder_conv_1')
            #     decoder_features = tf.image.resize_bilinear(
            #         decoder_features,
            #         tf.shape(decoder_features)[1:3] * 2,
            #         align_corners=True)
            #     decoder_features = slim.conv2d(
            #         decoder_features,
            #         self._first_stage_semantic_depth,
            #         kernel_size=self._first_stage_semantic_kernel_size,
            #         activation_fn=tf.nn.relu,
            #         scope='decoder_conv_2')
            semantic_logits = slim.conv2d(
                decoder_features,
                self.num_semantic_classes,
                kernel_size=1,
                activation_fn=None,
                normalizer_fn=None,
                scope='logits')

            # MixPrecision semantic logits.
            semantic_logits = tf.cast(semantic_logits, dtype=tf.float32)

    return semantic_logits

  def _remove_invalid_anchors_and_predictions(
    self,
    box_encodings,
    objectness_predictions_with_background,
    anchors_boxlist,
    clip_window):
    """Removes anchors that (partially) fall outside an image.

    Also removes associated box encodings and objectness predictions.

    Args:
      box_encodings: 3-D float tensor of shape
        [batch_size, num_anchors, self._box_coder.code_size] containing
        predicted boxes.
      objectness_predictions_with_background: 3-D float tensor of shape
        [batch_size, num_anchors, 2] containing class
        predictions (logits) for each of the anchors.  Note that this
        tensor *includes* background class predictions (at class index 0).
      anchors_boxlist: A BoxList representing num_anchors anchors (for the RPN)
        in absolute coordinates.
      clip_window: a 1-D tensor representing the [ymin, xmin, ymax, xmax]
        extent of the window to clip/prune to.

    Returns:
      box_encodings: 4-D float tensor of shape
        [batch_size, num_valid_anchors, self._box_coder.code_size] containing
        predicted boxes, where num_valid_anchors <= num_anchors
      objectness_predictions_with_background: 2-D float tensor of shape
        [batch_size, num_valid_anchors, 2] containing class
        predictions (logits) for each of the anchors, where
        num_valid_anchors <= num_anchors.  Note that this
        tensor *includes* background class predictions (at class index 0).
      anchors: A BoxList representing num_valid_anchors anchors (for the RPN) in
        absolute coordinates.
    """
    pruned_anchors_boxlist, keep_indices = box_list_ops.prune_outside_window(
        anchors_boxlist, clip_window)

    def _batch_gather_kept_indices(predictions_tensor):
      return shape_utils.static_or_dynamic_map_fn(
          partial(tf.gather, indices=keep_indices),
          elems=predictions_tensor,
          dtype=tf.float32,
          parallel_iterations=self._parallel_iterations,
          back_prop=True)
    return (_batch_gather_kept_indices(box_encodings),
            _batch_gather_kept_indices(objectness_predictions_with_background),
            pruned_anchors_boxlist)

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
    with tf.name_scope('FirstStagePostprocessor'):
      if self._number_of_stages == 1:
        proposal_boxes, proposal_scores, num_proposals = self._postprocess_rpn(
            prediction_dict['summary_images'],
            prediction_dict['rpn_box_encodings'],
            prediction_dict['rpn_objectness_predictions_with_background'],
            prediction_dict['anchors'],
            true_image_shapes,
            true_image_shapes)
        return {
            fields.DetectionResultFields.detection_boxes: proposal_boxes,
            fields.DetectionResultFields.detection_scores: proposal_scores,
            fields.DetectionResultFields.num_detections:
                tf.to_float(num_proposals),
        }

    # TODO(jrru): Remove mask_predictions from _post_process_box_classifier.
    with tf.name_scope('SecondStagePostprocessor'):
      if (self._number_of_stages == 2 or (self._number_of_stages == 3 and
                                          self._is_training)):
        mask_predictions = prediction_dict.get(box_predictor.MASK_PREDICTIONS)
        detections_dict = self._postprocess_box_classifier(
            prediction_dict['refined_box_encodings'],
            prediction_dict['class_predictions_with_background'],
            prediction_dict['proposal_boxes'],
            prediction_dict['num_proposals'],
            true_image_shapes,
            mask_predictions=mask_predictions)
        if Options.semantic_segmentation:
            semantic_prediction = prediction_dict['semantic_predictions']
            semantic_prediction = tf.image.resize_bilinear(
                semantic_prediction, true_image_shapes[0][0:2],
                align_corners=True)
            semantic_prediction = tf.argmax(
                tf.nn.softmax(semantic_prediction), axis=3)
            semantic_prediction = tf.expand_dims(semantic_prediction, axis=-1)
            detections_dict[
                fields.DetectionResultFields.detection_semantic] \
                = semantic_prediction
        return detections_dict

    with tf.name_scope('SecondStagePostprocessorInferenceWithMask'):
      if self._number_of_stages == 3:
        # Post processing is already performed in 3rd stage. We need to transfer
        # postprocessed tensors from `prediction_dict` to `detections_dict`.
        detections_dict = {}
        for key in prediction_dict:
          if key == box_predictor.MASK_PREDICTIONS:
            detections_dict[fields.DetectionResultFields.detection_masks] = \
                tf.sigmoid(prediction_dict[key])
          elif 'detection' in key:
            detections_dict[key] = prediction_dict[key]
        if Options.semantic_segmentation:
            semantic_prediction, semantic_prediction_probability = \
                self._postprocess_semantic_logits(
                    prediction_dict['semantic_predictions'], true_image_shapes)
            detections_dict[
                fields.DetectionResultFields.detection_semantic] \
                = semantic_prediction
            detections_dict[
                fields.DetectionResultFields.detection_semantic_heatmap] \
                = semantic_prediction_probability
        # [Combining Instances] ********************************************
            if Options.network_type == 'panoptic':
                boxes = detections_dict[
                    fields.DetectionResultFields.detection_boxes]
                scores = detections_dict[
                    fields.DetectionResultFields.detection_scores]
                classes = detections_dict[
                    fields.DetectionResultFields.detection_classes]
                masks = detections_dict[
                    fields.DetectionResultFields.detection_masks]
                mask_image, panoptic_image = self._postprocess_panoptic(
                    boxes, scores, classes, masks, semantic_prediction,
                    semantic_prediction_probability, true_image_shapes)
                detections_dict[
                    fields.DetectionResultFields.detection_masks_image] = \
                    mask_image
                detections_dict[
                    fields.DetectionResultFields.detection_panoptic_image] = \
                    panoptic_image
                params = (mask_image, panoptic_image)
                self._add_summary_panoptic_image(params)
        # ******************************************** [Combining Instances]
        return detections_dict

  def _postprocess_rpn(self,
                       images,
                       rpn_box_encodings_batch,
                       rpn_objectness_predictions_with_background_batch,
                       anchors,
                       image_shapes,
                       true_image_shapes):
    """Converts first stage prediction tensors from the RPN to proposals.

    This function decodes the raw RPN predictions, runs non-max suppression
    on the result.

    Note that the behavior of this function is slightly modified during
    training --- specifically, we stop the gradient from passing through the
    proposal boxes and we only return a balanced sampled subset of proposals
    with size `second_stage_batch_size`.

    Args:
      rpn_box_encodings_batch: A 3-D float32 tensor of shape
        [batch_size, num_anchors, self._box_coder.code_size] containing
        predicted proposal box encodings.
      rpn_objectness_predictions_with_background_batch: A 3-D float tensor of
        shape [batch_size, num_anchors, 2] containing objectness predictions
        (logits) for each of the anchors with 0 corresponding to background
        and 1 corresponding to object.
      anchors: A 2-D tensor of shape [num_anchors, 4] representing anchors
        for the first stage RPN.  Note that `num_anchors` can differ depending
        on whether the model is created in training or inference mode.
      image_shapes: A 2-D tensor of shape [batch, 3] containing the shapes of
        images in the batch.
      true_image_shapes: int32 tensor of shape [batch, 3] where each row is
        of the form [height, width, channels] indicating the shapes
        of true images in the resized images, as resized images can be padded
        with zeros.

    Returns:
      proposal_boxes: A float tensor with shape
        [batch_size, max_num_proposals, 4] representing the (potentially zero
        padded) proposal boxes for all images in the batch.  These boxes are
        represented as normalized coordinates.
      proposal_scores:  A float tensor with shape
        [batch_size, max_num_proposals] representing the (potentially zero
        padded) proposal objectness scores for all images in the batch.
      num_proposals: A Tensor of type `int32`. A 1-D tensor of shape [batch]
        representing the number of proposals predicted for each image in
        the batch.
    """
    # Normalize proposal boxes
    def normalize_boxes(args):
        proposal_boxes_per_image = args[0]
        image_shape = args[1]
        normalized_boxes_per_image = box_list_ops.to_normalized_coordinates(
            box_list.BoxList(proposal_boxes_per_image), image_shape[0],
            image_shape[1], check_range=False).get()
        return normalized_boxes_per_image

    if self._mask_rcnn_box_predictor.get_prediction_head_backbone() is None \
            or self._mask_rcnn_box_predictor.get_prediction_head_backbone() \
            == 'C4':
        rpn_box_encodings_batch = tf.expand_dims(rpn_box_encodings_batch,
                                                 axis=2)
        rpn_encodings_shape = shape_utils.combined_static_and_dynamic_shape(
            rpn_box_encodings_batch)
        tiled_anchor_boxes = tf.tile(
            tf.expand_dims(anchors, 0), [rpn_encodings_shape[0], 1, 1])
        proposal_boxes = self._batch_decode_boxes(rpn_box_encodings_batch,
                                                  tiled_anchor_boxes)
        proposal_boxes = tf.squeeze(proposal_boxes, axis=2)
        rpn_objectness_softmax_without_background = tf.nn.softmax(
            rpn_objectness_predictions_with_background_batch)[:, :, 1]
        clip_window = lib.compute_clip_window(image_shapes)
        (proposal_boxes, proposal_scores, _, _, _,
         num_proposals) = self._first_stage_nms_fn(
            tf.expand_dims(proposal_boxes, axis=2),
            tf.expand_dims(rpn_objectness_softmax_without_background, axis=2),
            clip_window=clip_window)
        if self._is_training:
            proposal_boxes = tf.stop_gradient(proposal_boxes)
            if not self._hard_example_miner:
                (groundtruth_boxlists, groundtruth_classes_with_background_list,
                 _, groundtruth_weights_list
                 ) = self._format_groundtruth_data(true_image_shapes)
                (proposal_boxes, proposal_scores,
                 num_proposals) = self._sample_box_classifier_batch(
                    proposal_boxes, proposal_scores, num_proposals,
                    groundtruth_boxlists,
                    groundtruth_classes_with_background_list,
                    groundtruth_weights_list)

        normalized_proposal_boxes = shape_utils.static_or_dynamic_map_fn(
            normalize_boxes, elems=[proposal_boxes, image_shapes],
            dtype=fp_dtype)
        return normalized_proposal_boxes, proposal_scores, num_proposals
    elif self._mask_rcnn_box_predictor.get_prediction_head_backbone() == 'FPN':
        ########################################################################
        # (3.0.) Stacking Proposals                                            #
        ########################################################################
        stacked_proposal_boxes = None
        stacked_proposal_scores = None
        stacked_num_proposals = None
        single_layer_max_proposals = self._first_stage_max_proposals
        # Evaluating each pyramid layer.
        for (idx,
             (single_layer_rpn_box_encodings,
              single_layer_rpn_obj_predictions_with_background,
              single_layer_anchors)) in enumerate(
            zip(reversed(rpn_box_encodings_batch),
                reversed(rpn_objectness_predictions_with_background_batch),
                reversed(anchors))
        ):
            # 1. Getting the proposal boxes.
            rpn_box = tf.expand_dims(single_layer_rpn_box_encodings, axis=2)
            rpn_box = tf.cast(rpn_box, dtype=tf.float32)
            rpn_encodings_shape = shape_utils.combined_static_and_dynamic_shape(
                rpn_box)
            tiled_anchor_boxes = tf.tile(
                tf.expand_dims(single_layer_anchors, 0),
                [rpn_encodings_shape[0], 1, 1])
            proposal_boxes = self._batch_decode_boxes(rpn_box,
                                                      tiled_anchor_boxes)
            # 2. Getting the proposal scores without background class.
            single_layer_rpn_obj_predictions_with_background = tf.cast(
                single_layer_rpn_obj_predictions_with_background,
                dtype=tf.float32)
            rpn_objectness_softmax_without_background = tf.nn.softmax(
                single_layer_rpn_obj_predictions_with_background)[:, :, 1]
            rpn_objectness_softmax_without_background = tf.expand_dims(
                rpn_objectness_softmax_without_background, axis=2)
            # 3. Clip window to clip the proposals to the image size.
            clip_window = lib.compute_clip_window(image_shapes)
            # 4. NMS
            (proposal_boxes, proposal_scores, _, _, _,
             num_proposals) = self._first_stage_nms_fn(
                proposal_boxes,
                rpn_objectness_softmax_without_background,
                clip_window=clip_window)
            # (proposal_boxes, proposal_scores, _, _, _, num_proposals) = \
            #     post_processing.batch_multiclass_non_max_suppression(
            #         proposal_boxes,
            #         rpn_objectness_softmax_without_background,
            #         self._first_stage_nms_score_threshold,
            #         self._first_stage_nms_iou_threshold,
            #         self._first_stage_max_proposals,
            #         self._first_stage_max_proposals,
            #         clip_window=clip_window)
            # 5. Evaluating each batch.
            # 5.1. For first layer we just copy the output of NMS.
            if stacked_proposal_boxes is None and \
                    stacked_proposal_scores is None and \
                    stacked_num_proposals is None:
                stacked_proposal_boxes = proposal_boxes
                stacked_proposal_scores = proposal_scores
                stacked_num_proposals = num_proposals
            # 5.2. For rest of the layers we need to concatenate the
            # positives first and the negatives after it.
            else:
                assert stacked_proposal_boxes is not None, \
                    'stacked_proposal_boxes is None.'
                assert stacked_proposal_scores is not None, \
                    'stacked_proposal_scores is None.'
                assert stacked_num_proposals is not None, \
                    'stacked_num_proposals is None.'
                _boxes_list = []
                _scores_list = []
                for (single_image_proposal_boxes,
                     single_image_proposal_scores,
                     single_image_num_proposals,
                     stacked_single_image_proposal_boxes,
                     stacked_single_image_proposal_scores,
                     stacked_single_image_num_proposals) in zip(
                    tf.unstack(proposal_boxes),
                    tf.unstack(proposal_scores),
                    tf.unstack(num_proposals),
                    tf.unstack(stacked_proposal_boxes),
                    tf.unstack(stacked_proposal_scores),
                    tf.unstack(stacked_num_proposals)
                ):
                    # Separating the positive and negative for saved tensor.
                    stacked_single_image_static_shape = \
                        stacked_single_image_proposal_boxes.get_shape()
                    stacked_single_sliced_static_shape = tf.TensorShape(
                        [tf.Dimension(None),
                         stacked_single_image_static_shape.dims[-1]])
                    valid_stacked_single_image_proposal_boxes = tf.slice(
                        stacked_single_image_proposal_boxes,
                        [0, 0],
                        [stacked_single_image_num_proposals, -1])
                    valid_stacked_single_image_proposal_boxes.set_shape(
                        stacked_single_sliced_static_shape)
                    valid_stacked_single_image_proposal_scores = tf.slice(
                        stacked_single_image_proposal_scores,
                        [0],
                        [stacked_single_image_num_proposals])
                    non_valid_stacked_single_image_proposal_boxes = tf.slice(
                        stacked_single_image_proposal_boxes,
                        [stacked_single_image_num_proposals, 0],
                        [single_layer_max_proposals * idx -
                         stacked_single_image_num_proposals, -1])
                    non_valid_stacked_single_image_proposal_boxes.set_shape(
                        stacked_single_sliced_static_shape)
                    non_valid_stacked_single_image_proposal_scores = tf.slice(
                        stacked_single_image_proposal_scores,
                        [stacked_single_image_num_proposals],
                        [single_layer_max_proposals * idx -
                         stacked_single_image_num_proposals])
                    # Adding the positive and negative of current batch to
                    # the saved tensor.
                    _boxes_list.append(tf.concat([
                        valid_stacked_single_image_proposal_boxes,
                        single_image_proposal_boxes,
                        non_valid_stacked_single_image_proposal_boxes], axis=0))
                    _scores_list.append(tf.concat([
                        valid_stacked_single_image_proposal_scores,
                        single_image_proposal_scores,
                        non_valid_stacked_single_image_proposal_scores], axis=0)
                    )
                # Stacking all the batches together.
                stacked_proposal_boxes = tf.stack(_boxes_list)
                stacked_proposal_scores = tf.stack(_scores_list)
                stacked_num_proposals = tf.add_n([stacked_num_proposals,
                                                  num_proposals])

        # Hard miner to get the 3:1 ratio of negative to positive.
        if self._is_training:
            stacked_proposal_boxes = tf.stop_gradient(
                stacked_proposal_boxes, name='Hard_Miner_SG')
            with tf.name_scope("Hard_Miner"):
                if not self._hard_example_miner:
                    (stacked_proposal_boxes, stacked_proposal_scores,
                     stacked_num_proposals) = self._hard_example_miner_fn(
                        stacked_proposal_boxes, stacked_proposal_scores,
                        stacked_num_proposals, true_image_shapes, images)

        normalized_proposal_boxes = shape_utils.static_or_dynamic_map_fn(
            normalize_boxes, elems=[stacked_proposal_boxes, image_shapes],
            dtype=tf.float32)
        return (normalized_proposal_boxes, stacked_proposal_scores,
                stacked_num_proposals)
    else:
      raise ValueError("Backbone is not supported...")

  @staticmethod
  def _postprocess_semantic_logits(logits, true_image_shapes):
    """Postprocess the raw semantic logit prediction.

    Args:
      logits: A float tensor with shape [batch, H, W, 1] representing
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

    semantic_prediction_logits = tf.map_fn(
        resize_logits, elems=[logits, true_image_shapes], dtype=logits.dtype)
    semantic_prediction_softmax = tf.nn.softmax(
      semantic_prediction_logits, axis=-1)
    semantic_prediction_probability = tf.reduce_max(
      semantic_prediction_softmax, axis=3, keepdims=True)
    semantic_prediction = tf.argmax(
      semantic_prediction_softmax, axis=3, output_type=tf.int32)
    semantic_prediction = tf.expand_dims(
      semantic_prediction, axis=-1)
    return semantic_prediction, semantic_prediction_probability

  def _postprocess_panoptic(self,
                            boxes,
                            scores,
                            classes,
                            masks,
                            semantic_prediction,
                            semantic_prediction_probability,
                            true_image_shapes):
    """Combines the results of instance and semantic segmentation.

    Args:
      boxes: a numpy array of shape [N, max_detections, 4], Decoded boxes for
        instance segmentation.
      scores: a numpy array of shape [N, max_detections], Scores of the
        decoded boxes.
      classes: [N, max_detections] int tensor of detection classes. Class
        label of the decoded boxes.
      masks: A 4D float32 tensor of shape [N, max_detection, H, W].
        Multiclass mask prediction in logits.
      semantic_prediction: int32 tensor with shape
        [batch, H, W, 1] representing the class label for each element.
      semantic_prediction_probability: int32 tensor with shape
        [batch, H, W, 1] representing the softmax probability of
        semantic_prediction.
      true_image_shapes: int32 tensor of shape [batch, 3] where each row is
        of the form [height, width, channels] indicating the shapes
        of true images in the resized images, as resized images can be padded
        with zeros.
    """
    # 1. Sort the detections.
    def rearrange_2d_tensor(args):
      [tensor, indices] = args
      return tf.gather(tensor, indices)
    sorting_index = tf.contrib.framework.argsort(
      scores, axis=-1, direction='DESCENDING')
    boxes = tf.map_fn(rearrange_2d_tensor, elems=[boxes, sorting_index],
                      dtype=boxes.dtype)
    scores = tf.map_fn(rearrange_2d_tensor, elems=[scores, sorting_index],
                       dtype=scores.dtype)
    classes += 1
    classes = tf.map_fn(rearrange_2d_tensor, elems=[classes, sorting_index],
                        dtype=classes.dtype)
    masks = tf.map_fn(rearrange_2d_tensor, elems=[masks, sorting_index],
                      dtype=masks.dtype)

    # 2. Create panoptic image.
    def create_blank_images(args):
      return tf.zeros(args, dtype=tf.uint8)

    images = tf.map_fn(create_blank_images,
                       elems=true_image_shapes, dtype=tf.uint8)
    semantic_prediction = tf.cast(semantic_prediction, dtype=tf.uint8)
    mask_image, panoptic_image = vis.create_direct_panoptic_image_tensors(
        images, boxes, scores, classes, masks,
        semantic_prediction, semantic_prediction_probability,
        self.num_classes, self.num_semantic_classes,
        self._panoptic_score_threshold, self._panoptic_mask_threshold)
    return mask_image, panoptic_image

  # FOR C4 only
  def _sample_box_classifier_batch(self,
                                   proposal_boxes,
                                   proposal_scores,
                                   num_proposals,
                                   groundtruth_boxlists,
                                   groundtruth_classes_with_background_list,
                                   groundtruth_weights_list):
    """Samples a minibatch for second stage.

    Args:
      proposal_boxes: A float tensor with shape
        [batch_size, num_proposals, 4] representing the (potentially zero
        padded) proposal boxes for all images in the batch.  These boxes are
        represented in absolute coordinates.
      proposal_scores:  A float tensor with shape
        [batch_size, num_proposals] representing the (potentially zero
        padded) proposal objectness scores for all images in the batch.
      num_proposals: A Tensor of type `int32`. A 1-D tensor of shape [batch]
        representing the number of proposals predicted for each image in
        the batch.
      groundtruth_boxlists: A list of BoxLists containing (absolute) coordinates
        of the groundtruth boxes.
      groundtruth_classes_with_background_list: A list of 2-D one-hot
        (or k-hot) tensors of shape [num_boxes, num_classes+1] containing the
        class targets with the 0th index assumed to map to the background class.
      groundtruth_weights_list: A list of 1-D tensors of shape [num_boxes]
        indicating the weight associated with the groundtruth boxes.

    Returns:
      proposal_boxes: A float tensor with shape
        [batch_size, second_stage_batch_size, 4] representing the (potentially
        zero padded) proposal boxes for all images in the batch.  These boxes
        are represented in absolute coordinates.
      proposal_scores:  A float tensor with shape
        [batch_size, second_stage_batch_size] representing the (potentially zero
        padded) proposal objectness scores for all images in the batch.
      num_proposals: A Tensor of type `int32`. A 1-D tensor of shape [batch]
        representing the number of proposals predicted for each image in
        the batch.
    """
    single_image_proposal_box_sample = []
    single_image_proposal_score_sample = []
    single_image_num_proposals_sample = []
    for (single_image_proposal_boxes,
         single_image_proposal_scores,
         single_image_num_proposals,
         single_image_groundtruth_boxlist,
         single_image_groundtruth_classes_with_background,
         single_image_groundtruth_weights) in zip(
             tf.unstack(proposal_boxes),
             tf.unstack(proposal_scores),
             tf.unstack(num_proposals),
             groundtruth_boxlists,
             groundtruth_classes_with_background_list,
             groundtruth_weights_list):
      single_image_boxlist = box_list.BoxList(single_image_proposal_boxes)
      single_image_boxlist.add_field(fields.BoxListFields.scores,
                                     single_image_proposal_scores)
      sampled_boxlist = self._sample_box_classifier_minibatch_single_image(
          single_image_boxlist,
          single_image_num_proposals,
          single_image_groundtruth_boxlist,
          single_image_groundtruth_classes_with_background,
          single_image_groundtruth_weights)
      sampled_padded_boxlist = box_list_ops.pad_or_clip_box_list(
          sampled_boxlist,
          num_boxes=self._second_stage_batch_size)
      single_image_num_proposals_sample.append(tf.minimum(
          sampled_boxlist.num_boxes(),
          self._second_stage_batch_size))
      bb = sampled_padded_boxlist.get()
      single_image_proposal_box_sample.append(bb)
      single_image_proposal_score_sample.append(
          sampled_padded_boxlist.get_field(fields.BoxListFields.scores))
    return (tf.stack(single_image_proposal_box_sample),
            tf.stack(single_image_proposal_score_sample),
            tf.stack(single_image_num_proposals_sample))

  def _hard_example_miner_fn(self,
                             proposal_boxes,
                             proposal_scores,
                             num_proposals,
                             true_image_shapes,
                             images):
    """Return a balanced sampled subset of proposals
    with size `second_stage_batch_size`.

    Args:
        See 'post_processing.batch_multiclass_non_max_suppression(...)'
        'proposal_boxes': A [batch_size, max_detections, 4] float32 tensor
            containing the non-max suppressed boxes.
        'proposal_scores': A [batch_size, max_detections] float32 tensor
            containing the scores for the boxes.
        'num_proposals': A [batch_size] int32 tensor indicating the number
            of valid detections per batch item. Only the top
            num_detections[i] entries in nms_boxes[i], nms_scores[i] and
            nms_class[i] are valid. The rest of the entries are zero
            paddings.
        true_image_shapes: int32 tensor of shape [batch, 3] where each row
            is of the form [height, width, channels] indicating the shapes
            of true images in the resized images, as resized images can be
            padded with zeros.

    Returns:
        See '_sample_box_classifier_batch_fpn(...)'
        proposal_boxes: A float tensor with shape
            [batch_size, second_stage_batch_size, 4] representing the
            (potentially zero padded) proposal boxes for all images in the
            batch. These boxes are represented in absolute coordinates.
        proposal_scores:  A float tensor with shape
            [batch_size, second_stage_batch_size] representing the
            (potentially zero padded) proposal objectness scores for all
            images in the batch.
        num_proposals: A Tensor of type `int32`. A 1-D tensor of shape
            [batch] representing the number of proposals predicted for each
            image in the batch.
    """
    (groundtruth_boxlists, groundtruth_classes_with_background_list, _,
     groundtruth_weights_list) = self._format_groundtruth_data(
        true_image_shapes)

    if Options.proposals_pre_mining:
        lib.draw_results(images,
                         proposal_boxes,
                         Options.object_index,
                         classes=None,
                         scores=None,
                         instance_masks=None,
                         min_score_thresh=0.1,
                         max_boxes_to_draw=2000,
                         use_normalized_coordinates=False,
                         name='proposals_pre_mining')

    (proposal_boxes_subsample,
     proposal_scores_subsample,
     num_proposals_subsample
     ) = self._sample_box_classifier_batch_fpn(
        images, proposal_boxes, proposal_scores, num_proposals,
        groundtruth_boxlists,
        groundtruth_classes_with_background_list,
        groundtruth_weights_list)

    if Options.proposals_gt:
        _boxes = groundtruth_boxlists[0].get()
        _boxes = tf.expand_dims(_boxes, axis=0)
        lib.draw_results(images,
                         _boxes,
                         Options.object_index,
                         classes=None,
                         scores=None,
                         instance_masks=None,
                         min_score_thresh=0.1,
                         max_boxes_to_draw=2000,
                         use_normalized_coordinates=False,
                         name='proposals_gt')

    if Options.proposals_post_mining:
        lib.draw_results(images,
                         proposal_boxes_subsample,
                         Options.object_index,
                         classes=None,
                         scores=proposal_scores_subsample,
                         instance_masks=None,
                         min_score_thresh=0.1,
                         max_boxes_to_draw=2000,
                         use_normalized_coordinates=False,
                         name='proposals_post_mining')

    return (proposal_boxes_subsample, proposal_scores_subsample,
            num_proposals_subsample)

  def _sample_box_classifier_batch_fpn(
        self,
        images,
        proposal_boxes,
        proposal_scores,
        num_proposals,
        groundtruth_boxlists,
        groundtruth_classes_with_background_list,
        groundtruth_weights_list):
    """Unpads proposals and samples a minibatch for second stage.

    Args:
        proposal_boxes: A float tensor with shape
            [batch_size, num_proposals, 4] representing the (potentially
            zero padded) proposal boxes for all images in the batch.
            These boxes are represented in absolute coordinates.
        proposal_scores:  A float tensor with shape
            [batch_size, num_proposals] representing the (potentially zero
            padded) proposal objectness scores for all images in the batch.
        num_proposals: A Tensor of type `int32`. A 1-D tensor of shape
            [batch] representing the number of proposals predicted for each
            image in the batch.
        groundtruth_boxlists: A list of BoxLists containing (absolute)
            coordinates of the groundtruth boxes.
        groundtruth_classes_with_background_list: A list of 2-D one-hot
            (or k-hot) tensors of shape [num_boxes, num_classes+1]
            containing the class targets with the 0th index assumed to map
            to the background class.

    Returns:
        proposal_boxes: A float tensor with shape
            [batch_size, second_stage_batch_size, 4] representing the
            (potentially zero padded) proposal boxes for all images in the
            batch. These boxes are represented in absolute coordinates.
        proposal_scores:  A float tensor with shape
            [batch_size, second_stage_batch_size] representing the
            (potentially zero padded) proposal objectness scores for all
            images in the batch.
        num_proposals: A Tensor of type `int32`. A 1-D tensor of shape
            [batch] representing the number of proposals predicted for each
            image in the batch.
    """
    # Each batch is evaluated as a `single_image_...`
    single_image_flag = True
    single_image_proposal_box_sample = []
    single_image_proposal_score_sample = []
    single_image_num_proposals_sample = []
    for (single_image,
         single_image_proposal_boxes,
         single_image_proposal_scores,
         single_image_num_proposals,
         single_image_groundtruth_boxlist,
         single_image_groundtruth_classes_with_background,
         single_image_groundtruth_weights_list) in zip(
        tf.unstack(images),
        tf.unstack(proposal_boxes),
        tf.unstack(proposal_scores),
        tf.unstack(num_proposals),
        groundtruth_boxlists,
        groundtruth_classes_with_background_list,
        groundtruth_weights_list
    ):
        tf.summary.scalar("Number_of_available_proposals",
                          single_image_num_proposals)
        single_image_boxlist = box_list.BoxList(single_image_proposal_boxes)
        single_image_boxlist.add_field(fields.BoxListFields.scores,
                                       single_image_proposal_scores)
        # 1. Get sub-sampling indices.
        (sampled_indices,
         positive_indices) = self._sample_box_classifier_minibatch_single_image(
            single_image_boxlist, single_image_num_proposals,
            single_image_groundtruth_boxlist,
            single_image_groundtruth_classes_with_background,
            single_image_groundtruth_weights_list)
        # 2. Get sub-sampling boxes (proposal).
        sampled_boxlist = box_list_ops.boolean_mask(
            single_image_boxlist, sampled_indices)
        sampled_padded_boxlist = box_list_ops.pad_or_clip_box_list(
            sampled_boxlist,
            num_boxes=self._second_stage_batch_size)
        positive_sampled_boxlist = box_list_ops.boolean_mask(
            single_image_boxlist, positive_indices)
        positive_sampled_padded_boxlist = box_list_ops.pad_or_clip_box_list(
            positive_sampled_boxlist,
            num_boxes=self._second_stage_batch_size)

        if Options.pre_box_classifier_minibatch_gt and single_image_flag:
            _boxes = single_image_groundtruth_boxlist.get()
            _boxes = tf.expand_dims(_boxes, axis=0)
            lib.draw_results(tf.expand_dims(single_image, axis=0),
                             _boxes,
                             Options.object_index,
                             classes=None,
                             scores=None,
                             instance_masks=None,
                             min_score_thresh=0.01,
                             max_boxes_to_draw=2000,
                             use_normalized_coordinates=False,
                             name='pre_box_classifier_minibatch_gt')

        if Options.pre_box_classifier_pre_minibatch_proposals and \
                single_image_flag:
            _boxes = single_image_boxlist.get()
            _boxes = tf.expand_dims(_boxes, axis=0)
            _scores = single_image_boxlist.get_field(
                fields.BoxListFields.scores)
            _scores = tf.expand_dims(_scores, axis=0)
            lib.draw_results(tf.expand_dims(single_image, axis=0),
                             _boxes,
                             Options.object_index,
                             classes=None,
                             scores=_scores,
                             instance_masks=None,
                             min_score_thresh=0.01,
                             max_boxes_to_draw=2000,
                             use_normalized_coordinates=False,
                             name='pre_box_classifier_pre_minibatch'
                                  '_proposals')

        if Options.pre_box_classifier_post_minibatch_proposals and \
                single_image_flag:
            _boxes = sampled_padded_boxlist.get()
            _boxes = tf.expand_dims(_boxes, axis=0)
            _scores = sampled_padded_boxlist.get_field(
                fields.BoxListFields.scores)
            _scores = tf.expand_dims(_scores, axis=0)
            lib.draw_results(tf.expand_dims(single_image, axis=0),
                             _boxes,
                             Options.object_index,
                             classes=None,
                             scores=_scores,
                             instance_masks=None,
                             min_score_thresh=0.01,
                             max_boxes_to_draw=2000,
                             use_normalized_coordinates=False,
                             name='pre_box_classifier_post_minibatch'
                                  '_proposals')

        if Options.pre_box_classifier_post_minibatch_proposals_50 and \
                single_image_flag:
            _boxes = sampled_padded_boxlist.get()
            _boxes = tf.expand_dims(_boxes, axis=0)
            _scores = sampled_padded_boxlist.get_field(
                fields.BoxListFields.scores)
            _scores = tf.expand_dims(_scores, axis=0)
            lib.draw_results(tf.expand_dims(single_image, axis=0),
                             _boxes,
                             Options.object_index,
                             classes=None,
                             scores=_scores,
                             instance_masks=None,
                             min_score_thresh=0.5,
                             max_boxes_to_draw=2000,
                             use_normalized_coordinates=False,
                             name='pre_box_classifier_post_minibatch'
                                  '_proposals_50')

        if Options.pre_box_classifier_post_minibatch_positive_proposals and \
                single_image_flag:
            _boxes = positive_sampled_padded_boxlist.get()
            _boxes = tf.expand_dims(_boxes, axis=0)
            _scores = positive_sampled_padded_boxlist.get_field(
                fields.BoxListFields.scores)
            _scores = tf.expand_dims(_scores, axis=0)
            # _max_boxes_to_draw = tf.reduce_sum(
            #     tf.cast(positive_indices, dtype=tf.int32))
            lib.draw_results(tf.expand_dims(single_image, axis=0),
                             _boxes,
                             Options.object_index,
                             classes=None,
                             scores=_scores,
                             instance_masks=None,
                             min_score_thresh=0.01,
                             max_boxes_to_draw=2000,
                             use_normalized_coordinates=False,
                             name='pre_box_classifier_post_minibatch'
                                  '_positive_proposals')

        # 3. Get number of proposals, possibly less than ss-batch-size.
        single_image_num_proposals_sample.append(tf.minimum(
            sampled_boxlist.num_boxes(),
            self._second_stage_batch_size))
        bb = sampled_padded_boxlist.get()
        tf.summary.scalar("Number_of_subsampled_proposals",
                          single_image_num_proposals_sample[-1])

        # 4. Append all data for single image/batch
        single_image_proposal_box_sample.append(bb)
        single_image_proposal_score_sample.append(
            sampled_padded_boxlist.get_field(fields.BoxListFields.scores))

        # Make sure we draw just a single image.
        single_image_flag = False

    return (tf.stack(single_image_proposal_box_sample),
            tf.stack(single_image_proposal_score_sample),
            tf.stack(single_image_num_proposals_sample))

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

  def _sample_box_classifier_minibatch_single_image(
          self, proposal_boxlist, num_valid_proposals, groundtruth_boxlist,
          groundtruth_classes_with_background, groundtruth_weights):
    """Samples a mini-batch of proposals to be sent to the box classifier.

    Helper function for self._postprocess_rpn.

    Args:
      proposal_boxlist: A BoxList containing K proposal boxes in absolute
        coordinates.
      num_valid_proposals: Number of valid proposals in the proposal boxlist.
      groundtruth_boxlist: A Boxlist containing N groundtruth object boxes in
        absolute coordinates.
      groundtruth_classes_with_background: A tensor with shape
        `[N, self.num_classes + 1]` representing groundtruth classes. The
        classes are assumed to be k-hot encoded, and include background as the
        zero-th class.
      groundtruth_weights: Weights attached to the groundtruth_boxes.

    Returns:
      a BoxList contained sampled proposals.
    """
    (cls_targets, cls_weights, _, _, _) = self._detector_target_assigner.assign(
        proposal_boxlist,
        groundtruth_boxlist,
        groundtruth_classes_with_background,
        unmatched_class_label=tf.constant(
            [1] + self._num_classes * [0], dtype=tf.float32),
        groundtruth_weights=groundtruth_weights)
    # Selects all boxes as candidates if none of them is selected according
    # to cls_weights. This could happen as boxes within certain IOU ranges
    # are ignored. If triggered, the selected boxes will still be ignored
    # during loss computation.
    positive_indicator = tf.greater(tf.argmax(cls_targets, axis=1), 0)
    valid_indicator = tf.logical_and(
        tf.range(proposal_boxlist.num_boxes()) < num_valid_proposals,
        cls_weights > 0
    )
    sampled_indices, positive_indices = self._second_stage_sampler.subsample(
        valid_indicator,
        self._second_stage_batch_size,
        positive_indicator)
    tf.summary.scalar("Valid_Num_of_Minibatch_For_SS", tf.reduce_sum(
        tf.cast(positive_indicator, dtype=tf.int32)))
    return sampled_indices, positive_indices
    # return box_list_ops.boolean_mask(proposal_boxlist, sampled_indices)

  def _compute_second_stage_input_feature_maps(self,
                                               features_to_crop,
                                               proposal_boxes_normalized,
                                               image_shape):
    """Crops to a set of proposals from the feature map for a batch of images.

    Helper function for self._postprocess_rpn. This function calls
    `tf.image.crop_and_resize` to create the feature map to be passed to the
    second stage box classifier for each proposal.

    Args:
    features_to_crop: A float32 tensor with shape
        [batch_size, height, width, depth]
    proposal_boxes_normalized: A float32 tensor with shape [batch_size,
        num_proposals, box_code_size] containing proposal boxes in
        normalized coordinates.
    image_shape: 2-D tensor of shape [batch_size, 3] were each row is of
        the form [height, width, channels].

    Returns:
      A float32 tensor with shape [K, new_height, new_width, depth].
    """

    if self._mask_rcnn_box_predictor.get_prediction_head_backbone() is None \
            or self._mask_rcnn_box_predictor.get_prediction_head_backbone() \
            == 'C4':
        cropped_regions = lib.flatten_first_two_dimensions(
            self._crop_and_resize_fn(
                features_to_crop, proposal_boxes_normalized,
                [self._initial_crop_size, self._initial_crop_size]))
        return slim.max_pool2d(
            cropped_regions,
            [self._maxpool_kernel_size, self._maxpool_kernel_size],
            stride=self._maxpool_stride)

    elif self._mask_rcnn_box_predictor.get_prediction_head_backbone() == 'FPN':
        ########################################################################
        # TAKEN FROM Mask R-CNN model implementation by Matterport.            #
        ########################################################################
        # Assign each ROI to a level in the pyramid based on the ROI area.
        y1, x1, y2, x2 = tf.split(proposal_boxes_normalized, 4, axis=2)
        roi_height = y2 - y1
        roi_width = x2 - x1
        image_height, image_width, _ = tf.split(image_shape, 3, axis=1)

        def log2_graph(x):
            """TF doesn't have a native implementation."""
            return tf.log(x) / tf.log(tf.cast(2.0, dtype=tf.float32))

        # Equation 1 in the Feature Pyramid Networks paper. Account for
        # the fact that our coordinates are normalized here.
        # e.g. a 224x224 ROI (in pixels) maps to P4
        # resnet_max_lvl = 5
        imagenet_canonical_size = tf.cast(224.0, dtype=tf.float32)

        image_area = tf.cast(image_height * image_width, dtype=tf.float32)
        roi_level = tf.fill(tf.shape(image_area), imagenet_canonical_size)
        roi_level = roi_level / tf.sqrt(image_area)
        roi_level = tf.sqrt(roi_height * roi_width) / tf.expand_dims(
            roi_level, axis=1)
        roi_level = log2_graph(roi_level)
        roi_level = 4 + tf.cast(tf.round(roi_level), dtype=tf.int32)
        roi_level = tf.maximum(self._first_stage_fpn_min_level, roi_level)
        roi_level = tf.minimum(self._first_stage_fpn_max_level, roi_level)
        roi_level = tf.squeeze(roi_level, 2)

        # Loop through levels and apply ROI pooling to each. P2 to P5.
        pooled = []
        box_to_level = []
        for i, level in enumerate(range(
                self._first_stage_fpn_min_level,
                self._first_stage_fpn_max_level + 1)):
            # Row dimension represent how many true element is present.
            # Column = [batch_i, proposal_i]
            ix = tf.where(tf.equal(roi_level, level))
            level_boxes = tf.gather_nd(proposal_boxes_normalized, ix)
            # Box indices for crop_and_resize.
            box_indices = tf.cast(ix[:, 0], tf.int32)
            # Keep track of which box is mapped to which level
            box_to_level.append(ix)
            # Stop gradient propagation to ROI proposals
            level_boxes = tf.stop_gradient(level_boxes, name='Level_Boxes_SG')
            box_indices = tf.stop_gradient(box_indices, name='Box_Indices_SG')
            # Crop and Resize
            # From Mask R-CNN paper: "We sample four regular locations, so
            # that we can evaluate either max or average pooling. In fact,
            # interpolating only a single value at each bin center (without
            # pooling) is nearly as effective."
            # Here we use the simplified approach of a single value per bin,
            # which is how it's done in tf.crop_and_resize()
            # Result: [batch * num_boxes, pool_height, pool_width, channels]
            pooled.append(
                tf.cast(
                    tf.image.crop_and_resize(
                        features_to_crop[i],
                        tf.cast(level_boxes, dtype=tf.float32),
                        box_indices,
                        (self._initial_crop_size, self._initial_crop_size),
                        method="bilinear"),
                    dtype=fp_dtype))

        # Pack pooled features into one tensor
        pooled = tf.concat(pooled, axis=0)
        # Pack box_to_level mapping into one array and add another
        # column representing the order of pooled boxes
        box_to_level = tf.concat(box_to_level, axis=0)
        box_range = tf.expand_dims(tf.range(tf.shape(box_to_level)[0]), 1)
        box_to_level = tf.concat([tf.cast(box_to_level, dtype=tf.int32),
                                  box_range], axis=1)
        # Rearrange pooled features to match the order of the original boxes
        # Sort box_to_level by batch then box index
        # TF doesn't have a way to sort by two columns, so merge them and sort.
        sorting_tensor = box_to_level[:, 0] * 100000 + box_to_level[:, 1]
        ix = tf.nn.top_k(sorting_tensor, k=tf.shape(
            box_to_level)[0]).indices[::-1]
        ix = tf.gather(box_to_level[:, 2], ix)
        pooled = tf.gather(pooled, ix)
        # # Re-add the batch dimension
        # pooled = tf.expand_dims(pooled, 0)

        # [DummyCode] **********************************************************
        # DEBUG: Trying to see if the roi features are faulty.
        # batch_size = 1
        # pooled.set_shape(
        #     [batch_size * proposal_boxes_normalized.shape[1].value,
        #      pooled.shape[1].value, pooled.shape[2].value,
        #      pooled.shape[3].value])
        # tmp_img = tf.reduce_sum(pooled, axis=3, keepdims=True)
        # for i in tf.unstack(tmp_img, axis=0):
        #     tf.summary.image("FeatureMaps", tf.expand_dims(i, axis=0))
        # ********************************************************** [DummyCode]

        return pooled

    else:
        raise ValueError("Backbone is not supported...")

  def _postprocess_box_classifier(self,
                                  refined_box_encodings,
                                  class_predictions_with_background,
                                  proposal_boxes,
                                  num_proposals,
                                  image_shapes,
                                  mask_predictions=None):
    """Converts predictions from the second stage box classifier to detections.

    Args:
      refined_box_encodings: a 3-D float tensor with shape
        [total_num_padded_proposals, num_classes, self._box_coder.code_size]
        representing predicted (final) refined box encodings. If using a shared
        box across classes the shape will instead be
        [total_num_padded_proposals, 1, 4]
      class_predictions_with_background: a 3-D tensor float with shape
        [total_num_padded_proposals, num_classes + 1] containing class
        predictions (logits) for each of the proposals.  Note that this tensor
        *includes* background class predictions (at class index 0).
      proposal_boxes: a 3-D float tensor with shape
        [batch_size, self.max_num_proposals, 4] representing decoded proposal
        bounding boxes in absolute coordinates.
      num_proposals: a 1-D int32 tensor of shape [batch] representing the number
        of proposals predicted for each image in the batch.
      image_shapes: a 2-D int32 tensor containing shapes of input image in the
        batch.
      mask_predictions: (optional) a 4-D float tensor with shape
        [total_num_padded_proposals, num_classes, mask_height, mask_width]
        containing instance mask prediction logits.

    Returns:
      A dictionary containing:
        `detection_boxes`: [batch, max_detection, 4]
        `detection_scores`: [batch, max_detections]
        `detection_classes`: [batch, max_detections]
        `num_detections`: [batch]
        `detection_masks`:
          (optional) [batch, max_detections, mask_height, mask_width]. Note
          that a pixel-wise sigmoid score converter is applied to the detection
          masks.
    """
    if self._mask_rcnn_box_predictor.get_prediction_head_backbone() is None \
            or self._mask_rcnn_box_predictor.get_prediction_head_backbone() \
            == 'C4':
        max_num_proposals = self.max_num_proposals
    elif self._mask_rcnn_box_predictor.get_prediction_head_backbone() == 'FPN':
        # self.max_num_proposals is not used here because for fpn we need to x5.
        # batch_size is better because we can infer it from number of proposals.
        max_num_proposals = tf.shape(proposal_boxes)[1]
    else:
        raise ValueError("Backbone is not supported...")

    refined_box_encodings_batch = tf.reshape(
        refined_box_encodings,
        [-1, max_num_proposals, self.num_classes,
         self._box_coder.code_size])
    class_predictions_with_background_batch = tf.reshape(
        class_predictions_with_background,
        [-1, max_num_proposals, self.num_classes + 1]
    )
    refined_decoded_boxes_batch = self._batch_decode_boxes(
        refined_box_encodings_batch, proposal_boxes)
    class_predictions_with_background_batch = (
        self._second_stage_score_conversion_fn(
            class_predictions_with_background_batch))
    class_predictions_batch = tf.reshape(
        tf.slice(class_predictions_with_background_batch,
                 [0, 0, 1], [-1, -1, -1]),
        [-1, max_num_proposals, self.num_classes])
    clip_window = lib.compute_clip_window(image_shapes)
    mask_predictions_batch = None
    if mask_predictions is not None:
        mask_height = mask_predictions.shape[-2].value
        mask_width = mask_predictions.shape[-1].value
        mask_predictions = tf.sigmoid(mask_predictions)
        mask_predictions_batch = tf.reshape(mask_predictions,
                                            [-1, max_num_proposals,
                                             self.num_classes,
                                             mask_height, mask_width])
    (nmsed_boxes, nmsed_scores, nmsed_classes, nmsed_masks, _,
     num_detections) = self._second_stage_nms_fn(
        refined_decoded_boxes_batch,
        class_predictions_batch,
        clip_window=clip_window,
        change_coordinate_frame=True,
        num_valid_boxes=num_proposals,
        masks=mask_predictions_batch)
    detections = {
        fields.DetectionResultFields.detection_boxes: nmsed_boxes,
        fields.DetectionResultFields.detection_scores: nmsed_scores,
        fields.DetectionResultFields.detection_classes: nmsed_classes,
        fields.DetectionResultFields.num_detections: tf.cast(
            num_detections, dtype=fp_dtype)
    }
    if nmsed_masks is not None:
        detections[fields.DetectionResultFields.detection_masks] = nmsed_masks
    return detections

  def _batch_decode_boxes(self, box_encodings, anchor_boxes):
    """Decodes box encodings with respect to the anchor boxes.

    Args:
      box_encodings: a 4-D tensor with shape
        [batch_size, num_anchors, num_classes, self._box_coder.code_size]
        representing box encodings.
      anchor_boxes: [batch_size, num_anchors, self._box_coder.code_size]
        representing decoded bounding boxes. If using a shared box across
        classes the shape will instead be
        [total_num_proposals, 1, self._box_coder.code_size].

    Returns:
      decoded_boxes: a
        [batch_size, num_anchors, num_classes, self._box_coder.code_size]
        float tensor representing bounding box predictions (for each image in
        batch, proposal and class). If using a shared box across classes the
        shape will instead be
        [batch_size, num_anchors, 1, self._box_coder.code_size].
    """
    combined_shape = shape_utils.combined_static_and_dynamic_shape(
        box_encodings)
    num_classes = combined_shape[2]
    tiled_anchor_boxes = tf.tile(
        tf.expand_dims(anchor_boxes, 2), [1, 1, num_classes, 1])
    tiled_anchors_boxlist = box_list.BoxList(
        tf.reshape(tiled_anchor_boxes, [-1, 4]))
    decoded_boxes = self._box_coder.decode(
        tf.reshape(box_encodings, [-1, self._box_coder.code_size]),
        tiled_anchors_boxlist)
    return tf.reshape(decoded_boxes.get(),
                      tf.stack([combined_shape[0], combined_shape[1],
                                num_classes, 4]))

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
    # [DummyCode] **************************************************************
    # DEBUG: Dummy code to test resnet 101 backbone.
    # scaled_labels = tf.ones([144], dtype=tf.int32)
    # one_hot_labels = slim.one_hot_encoding(
    #     scaled_labels, self.num_classes,
    #     on_value=1.0, off_value=0.0)
    # loss = tf.losses.softmax_cross_entropy(
    #     one_hot_labels,
    #     tf.reshape(prediction_dict['dummy_predictor'],
    #                shape=[-1, self.num_classes]),
    #     scope='dummy_loss', loss_collection=None)
    # loss = tf.reduce_sum(loss)
    # loss_dict = {'dummy_loss': loss}
    # return loss_dict
    # ************************************************************** [DummyCode]

    with tf.name_scope(scope, 'Loss', prediction_dict.values()):
      (groundtruth_boxlists, groundtruth_classes_with_background_list,
       groundtruth_masks_list, groundtruth_weights_list) = \
          self._format_groundtruth_data(true_image_shapes)

      # [RPN Loss] *************************************************************
      loss_dict = self._loss_rpn(
          prediction_dict['summary_images'],
          prediction_dict['rpn_box_encodings'],
          prediction_dict['rpn_objectness_predictions_with_background'],
          prediction_dict['anchors'],
          groundtruth_boxlists,
          groundtruth_weights_list)
      # ************************************************************* [RPN Loss]

      # [Semantic Loss] ********************************************************
      semantic_logits = None
      if Options.semantic_segmentation:
          _loss_dict, semantic_logits, not_ignore_mask = \
              self._loss_semantic(
                  prediction_dict['semantic_predictions'],
                  self.groundtruth_lists('semantic'),
                  ignore_label=0,
                  upsample_logits=True)
          loss_dict.update(_loss_dict)
      # ******************************************************** [Semantic Loss]

      if self._number_of_stages > 1:
        # [Detection Box and Class and Mask Loss] ******************************
        _loss_dict, object_prediction_masks = \
            self._loss_box_classifier(
                prediction_dict['summary_images'],
                prediction_dict['refined_box_encodings'],
                prediction_dict['class_predictions_with_background'],
                prediction_dict['proposal_boxes'],
                prediction_dict['num_proposals'],
                groundtruth_boxlists,
                groundtruth_classes_with_background_list,
                groundtruth_weights_list,
                prediction_dict['image_shape'],
                prediction_dict.get(box_predictor.MASK_PREDICTIONS),
                groundtruth_masks_list,)
        loss_dict.update(_loss_dict)
        # ****************************** [Detection Box and Class and Mask Loss]

        # [KL-Divergence Loss] *************************************************
        kl_loss_semantic_mask_weight = 0.01
        kl_loss_mask_semantic_weight = 0.01
        if Options.network_type == 'panoptic' and self._is_training \
                and False:
            kl_loss_mask_semantic = self._loss_kl_divergence(
                prediction_dict['refined_box_encodings'],
                prediction_dict['class_predictions_with_background'],
                prediction_dict['proposal_boxes'],
                prediction_dict['num_proposals'],
                prediction_dict['image_shape'],
                prediction_dict.get(box_predictor.MASK_PREDICTIONS),
                semantic_logits,
                semantic_mask=False)
            kl_loss_mask_semantic = tf.multiply(
                kl_loss_mask_semantic_weight,
                kl_loss_mask_semantic,
                name='kl_loss_mask_semantic')
            _loss_dict = {
                kl_loss_mask_semantic.op.name: kl_loss_mask_semantic}
            loss_dict.update(_loss_dict)

            kl_loss_semantic_mask = self._loss_kl_divergence(
                prediction_dict['refined_box_encodings'],
                prediction_dict['class_predictions_with_background'],
                prediction_dict['proposal_boxes'],
                prediction_dict['num_proposals'],
                prediction_dict['image_shape'],
                prediction_dict.get(box_predictor.MASK_PREDICTIONS),
                semantic_logits,
                semantic_mask=True)
            kl_loss_semantic_mask = tf.multiply(
                kl_loss_semantic_mask_weight,
                kl_loss_semantic_mask,
                name='kl_loss_semantic_mask')
            _loss_dict = {
                kl_loss_semantic_mask.op.name: kl_loss_semantic_mask}
            loss_dict.update(_loss_dict)

            # kl_loss_semantic_mask = tf.stop_gradient(
            #     kl_loss_semantic_mask, name='KL_Semantic_Mask_SG')
            # kl_loss_mask_semantic = tf.stop_gradient(
            #     kl_loss_mask_semantic, name='KL_Mask_Semantic_SG')
            # for k, v in loss_dict.items():
            #     if 'mask' in k:
            #         loss_dict[k] = tf.add(
            #             v, (kl_loss_semantic_mask *
            #                 kl_loss_semantic_mask_weight),
            #             name='BoxClassifierLoss/mask_loss_with_kl')
            #     if 'semantic' in k:
            #         loss_dict[k] = tf.add(
            #             v, (kl_loss_mask_semantic *
            #                 kl_loss_mask_semantic_weight),
            #             name='SemanticLoss/semantic_loss_with_kl')
        # ************************************************* [KL-Divergence Loss]

    return loss_dict

  def _loss_semantic(self,
                     semantic_predictions,
                     groundtruth_semantic_list,
                     ignore_label,
                     upsample_logits):
    """Computes scalar semantic loss tensors.

    We use the implementation used in deeplab.

    Args:
        semantic_predictions: A 4-D float tensor of shape
            [batch_size x W x H x num_semantic_classes] containing
            predicted semantic labels.
        groundtruth_semantic_list: a list of 3-D tensor of shape
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
    for gt in groundtruth_semantic_list:
        labels.append(
            tf.expand_dims(tf.slice(gt, [0, 0, 0], [-1, -1, 1]), axis=0))
    labels = tf.concat(labels, axis=0)
    logits = semantic_predictions
    loss_dict = None
    not_ignore_mask = None
    with tf.name_scope('SemanticLoss'):
        if groundtruth_semantic_list is None:
            raise ValueError('No label for softmax cross entropy loss.')
        if upsample_logits:
            # Label is not downsampled, and instead we upsample logits.
            logits = tf.image.resize_bilinear(
                logits, tf.shape(labels)[1:3], align_corners=True),
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
            weights=not_ignore_mask*self._first_stage_sem_loss_weight,
            loss_collection=None)
        semantic_loss = tf.identity(semantic_loss, name='semantic_loss')
        loss_dict = {semantic_loss.op.name: semantic_loss}
        not_ignore_mask = tf.reshape(not_ignore_mask, shape=original_shape)
    return loss_dict, logits, not_ignore_mask

  def _loss_rpn(self,
                images,
                rpn_box_encodings,
                rpn_objectness_predictions_with_background,
                anchors,
                groundtruth_boxlists,
                groundtruth_weights_list):
    """Computes scalar RPN loss tensors.

    Uses self._proposal_target_assigner to obtain regression and classification
    targets for the first stage RPN, samples a "minibatch" of anchors to
    participate in the loss computation, and returns the RPN losses.

    Args:
      rpn_box_encodings: A 4-D float tensor of shape
        [batch_size, num_anchors, self._box_coder.code_size] containing
        predicted proposal box encodings.
      rpn_objectness_predictions_with_background: A 2-D float tensor of shape
        [batch_size, num_anchors, 2] containing objectness predictions
        (logits) for each of the anchors with 0 corresponding to background
        and 1 corresponding to object.
      anchors: A 2-D tensor of shape [num_anchors, 4] representing anchors
        for the first stage RPN.  Note that `num_anchors` can differ depending
        on whether the model is created in training or inference mode.
      groundtruth_boxlists: A list of BoxLists containing coordinates of the
        groundtruth boxes.
      groundtruth_weights_list: A list of 1-D tf.float32 tensors of shape
        [num_boxes] containing weights for groundtruth boxes.

    Returns:
      a dictionary mapping loss keys (`first_stage_localization_loss`,
        `first_stage_objectness_loss`) to scalar tensors representing
        corresponding loss values.
    """
    if self._mask_rcnn_box_predictor.get_prediction_head_backbone() is None \
            or self._mask_rcnn_box_predictor.get_prediction_head_backbone() \
            == 'C4':
        with tf.name_scope('RPNLoss'):
          (batch_cls_targets, batch_cls_weights, batch_reg_targets,
           batch_reg_weights, _) = target_assigner.batch_assign_targets(
               target_assigner=self._proposal_target_assigner,
               anchors_batch=box_list.BoxList(anchors),
               gt_box_batch=groundtruth_boxlists,
               gt_class_targets_batch=(len(groundtruth_boxlists) * [None]),
               gt_weights_batch=groundtruth_weights_list)
          batch_cls_targets = tf.squeeze(batch_cls_targets, axis=2)

          def _minibatch_subsample_fn(inputs):
            cls_targets, cls_weights = inputs
            return self._first_stage_sampler.subsample(
                tf.cast(cls_weights, tf.bool),
                self._first_stage_minibatch_size, tf.cast(cls_targets, tf.bool))
          batch_sampled_indices = tf.to_float(
              shape_utils.static_or_dynamic_map_fn(
                  _minibatch_subsample_fn,
                  [batch_cls_targets, batch_cls_weights],
                  dtype=tf.bool,
                  parallel_iterations=self._parallel_iterations,
                  back_prop=True))

          # Normalize by number of examples in sampled minibatch
          normalizer = tf.reduce_sum(batch_sampled_indices, axis=1)
          batch_one_hot_targets = tf.one_hot(
              tf.to_int32(batch_cls_targets), depth=2)
          sampled_reg_indices = tf.multiply(batch_sampled_indices,
                                            batch_reg_weights)

          localization_losses = self._first_stage_localization_loss(
              rpn_box_encodings, batch_reg_targets, weights=sampled_reg_indices)
          objectness_losses = self._first_stage_objectness_loss(
              rpn_objectness_predictions_with_background,
              batch_one_hot_targets, weights=batch_sampled_indices)
          localization_loss = tf.reduce_mean(
              tf.reduce_sum(localization_losses, axis=1) / normalizer)
          objectness_loss = tf.reduce_mean(
              tf.reduce_sum(objectness_losses, axis=1) / normalizer)

          localization_loss = tf.multiply(self._first_stage_loc_loss_weight,
                                          localization_loss,
                                          name='localization_loss')
          objectness_loss = tf.multiply(self._first_stage_obj_loss_weight,
                                        objectness_loss, name='objectness_loss')
          loss_dict = {localization_loss.op.name: localization_loss,
                       objectness_loss.op.name: objectness_loss}

    elif self._mask_rcnn_box_predictor.get_prediction_head_backbone() == 'FPN':
        with tf.name_scope('RPNLoss'):
            localization_loss = tf.zeros(1, name='localization_loss')
            objectness_loss = tf.zeros(1, name='objectness_loss')
            for (single_level_rpn_box,
                 single_level_rpn_obj,
                 single_level_anchors) in zip(
                rpn_box_encodings,
                rpn_objectness_predictions_with_background,
                anchors
            ):
                if Options.rpn_proposals_per_layer_loss:
                    proposal_scores = tf.nn.softmax(
                        single_level_rpn_obj)[:, :, 1]
                    proposal_boxes = self._batch_decode_boxes(
                        tf.tile(tf.expand_dims(single_level_rpn_box, axis=2),
                                [1, 1, 2, 1]),
                        tf.expand_dims(single_level_anchors, axis=0))
                    proposal_boxes = tf.slice(proposal_boxes, [0, 0, 1, 0],
                                              [-1, -1, 1, -1])
                    proposal_boxes = tf.squeeze(proposal_boxes, axis=2)
                    lib.draw_results(images,
                                     proposal_boxes,
                                     Options.object_index,
                                     classes=None,
                                     scores=proposal_scores,
                                     instance_masks=None,
                                     min_score_thresh=0.5,
                                     max_boxes_to_draw=2000,
                                     use_normalized_coordinates=False,
                                     name='rpn_proposals_per_layer_loss')

                # 1. Obtains gt targets.
                (batch_cls_targets, batch_cls_weights, batch_reg_targets,
                 batch_reg_weights, _) = target_assigner.batch_assign_targets(
                    target_assigner=self._proposal_target_assigner,
                    anchors_batch=box_list.BoxList(single_level_anchors),
                    gt_box_batch=groundtruth_boxlists,
                    gt_class_targets_batch=(len(groundtruth_boxlists) * [None]),
                    gt_weights_batch=groundtruth_weights_list)
                batch_cls_targets = tf.squeeze(batch_cls_targets, axis=2)

                def _minibatch_subsample_fn(inputs):
                    cls_targets, cls_weights = inputs
                    sampled_indices, _ = self._first_stage_sampler.subsample(
                        tf.cast(cls_weights, tf.bool),
                        self._first_stage_minibatch_size,
                        tf.cast(cls_targets, tf.bool))
                    return sampled_indices

                # 2. Sub-samples a minibatch of anchors used to calculate loss.
                batch_sampled_indices = tf.cast(
                    shape_utils.static_or_dynamic_map_fn(
                        _minibatch_subsample_fn,
                        [batch_cls_targets, batch_cls_weights],
                        dtype=tf.bool,
                        parallel_iterations=self._parallel_iterations,
                        back_prop=True), dtype=tf.float32)

                # 3. Normalize by number of examples in sampled minibatch
                normalizer = tf.reduce_sum(batch_sampled_indices, axis=1)
                batch_one_hot_targets = tf.one_hot(
                    tf.to_int32(batch_cls_targets), depth=2)
                sampled_reg_indices = tf.multiply(batch_sampled_indices,
                                                  batch_reg_weights)

                # 4. Losses for each anchor.
                #  MixPrecision RPN box and objectness
                single_level_rpn_box = tf.cast(
                    single_level_rpn_box, dtype=tf.float32)
                single_level_rpn_obj = tf.cast(
                    single_level_rpn_obj, dtype=tf.float32)
                localization_losses = self._first_stage_localization_loss(
                    single_level_rpn_box, batch_reg_targets,
                    weights=sampled_reg_indices)
                objectness_losses = self._first_stage_objectness_loss(
                    single_level_rpn_obj, batch_one_hot_targets,
                    weights=batch_sampled_indices)

                # 5. Meaned anchor loss.
                single_level_localization_loss = tf.reduce_mean(
                    tf.reduce_sum(localization_losses, axis=1) / normalizer)
                single_level_objectness_loss = tf.reduce_mean(
                    tf.reduce_sum(objectness_losses, axis=1) / normalizer)

                # 6. Add loss with previous layers.
                localization_loss = tf.add(
                    localization_loss,
                    tf.multiply(self._first_stage_loc_loss_weight,
                                single_level_localization_loss),
                    name='localization_loss')
                objectness_loss = tf.add(
                    objectness_loss,
                    tf.multiply(self._first_stage_obj_loss_weight,
                                single_level_objectness_loss),
                    name='objectness_loss')
            loss_dict = {
                localization_loss.op.name:
                    tf.unstack(localization_loss, name='localization_loss')[0],
                objectness_loss.op.name:
                    tf.unstack(objectness_loss, name='objectness_loss')[0]
            }
    else:
        raise ValueError("Backbone is not supported...")

    return loss_dict

  def _loss_box_classifier(self,
                           images,
                           refined_box_encodings,
                           class_predictions_with_background,
                           proposal_boxes,
                           num_proposals,
                           groundtruth_boxlists,
                           groundtruth_classes_with_background_list,
                           groundtruth_weights_list,
                           image_shape,
                           prediction_masks=None,
                           groundtruth_masks_list=None):
    """Computes scalar box classifier loss tensors.

    Uses self._detector_target_assigner to obtain regression and classification
    targets for the second stage box classifier, optionally performs
    hard mining, and returns losses.  All losses are computed independently
    for each image and then averaged across the batch.
    Please note that for boxes and masks with multiple labels, the box
    regression and mask prediction losses are only computed for one label.

    This function assumes that the proposal boxes in the "padded" regions are
    actually zero (and thus should not be matched to).


    Args:
      refined_box_encodings: a 3-D tensor with shape
        [total_num_proposals, num_classes, box_coder.code_size] representing
        predicted (final) refined box encodings. If using a shared box across
        classes this will instead have shape
        [total_num_proposals, 1, box_coder.code_size].
      class_predictions_with_background: a 2-D tensor with shape
        [total_num_proposals, num_classes + 1] containing class
        predictions (logits) for each of the anchors.  Note that this tensor
        *includes* background class predictions (at class index 0).
      proposal_boxes: [batch_size, self.max_num_proposals, 4] representing
        decoded proposal bounding boxes.
      num_proposals: A Tensor of type `int32`. A 1-D tensor of shape [batch]
        representing the number of proposals predicted for each image in
        the batch.
      groundtruth_boxlists: a list of BoxLists containing coordinates of the
        groundtruth boxes.
      groundtruth_classes_with_background_list: a list of 2-D one-hot
        (or k-hot) tensors of shape [num_boxes, num_classes + 1] containing the
        class targets with the 0th index assumed to map to the background class.
      groundtruth_weights_list: A list of 1-D tf.float32 tensors of shape
        [num_boxes] containing weights for groundtruth boxes.
      image_shape: a 1-D tensor of shape [4] representing the image shape.
      prediction_masks: an optional 4-D tensor with shape [total_num_proposals,
        num_classes, mask_height, mask_width] containing the instance masks for
        each box.
      groundtruth_masks_list: an optional list of 3-D tensors of shape
        [num_boxes, image_height, image_width] containing the instance masks for
        each of the boxes.

    Returns:
      a dictionary mapping loss keys ('second_stage_localization_loss',
        'second_stage_classification_loss') to scalar tensors representing
        corresponding loss values.

    Raises:
      ValueError: if `predict_instance_masks` in
        second_stage_mask_rcnn_box_predictor is True and
        `groundtruth_masks_list` is not provided.
    """
    del images  # DEBUG: Keeping just in case for debugging

    if self._mask_rcnn_box_predictor.get_prediction_head_backbone() is None \
            or self._mask_rcnn_box_predictor.get_prediction_head_backbone() \
            == 'C4':
        max_num_proposals = self.max_num_proposals
    elif self._mask_rcnn_box_predictor.get_prediction_head_backbone() == 'FPN':
        # TODO: For eval self.max_num_proposals needs to be modified.
        # It is currently the same as first_stage_max_proposals, which is
        # wrong for fpn backbone.
        if self._is_training:
            max_num_proposals = proposal_boxes.shape[1].value
        else:
            max_num_proposals = tf.shape(proposal_boxes)[1]
    else:
        raise ValueError("Backbone is not supported...")

    with tf.name_scope('BoxClassifierLoss'):
      paddings_indicator = lib.padded_batched_proposals_indicator(
          num_proposals, max_num_proposals)
      proposal_boxlists = [
          box_list.BoxList(proposal_boxes_single_image)
          for proposal_boxes_single_image in tf.unstack(proposal_boxes)]
      batch_size = len(proposal_boxlists)

      num_proposals_or_one = tf.expand_dims(
          tf.maximum(num_proposals, tf.ones_like(num_proposals)), 1)
      normalizer = tf.cast(tf.tile(num_proposals_or_one,
                                   [1, max_num_proposals]) * batch_size,
                           dtype=tf.float32)

      (batch_cls_targets_with_background, batch_cls_weights, batch_reg_targets,
       batch_reg_weights, _) = target_assigner.batch_assign_targets(
           target_assigner=self._detector_target_assigner,
           anchors_batch=proposal_boxlists,
           gt_box_batch=groundtruth_boxlists,
           gt_class_targets_batch=groundtruth_classes_with_background_list,
           unmatched_class_label=tf.constant(
               [1] + self._num_classes * [0], dtype=tf.float32),
           gt_weights_batch=groundtruth_weights_list)

      class_predictions_with_background = tf.reshape(
          class_predictions_with_background,
          [batch_size, max_num_proposals, self.num_classes + 1])

      flat_cls_targets_with_background = tf.reshape(
          batch_cls_targets_with_background,
          [batch_size * max_num_proposals, -1])
      one_hot_flat_cls_targets_with_background = tf.argmax(
          flat_cls_targets_with_background, axis=1)
      if self._is_training:
          one_hot_flat_cls_targets_with_background = tf.one_hot(
              one_hot_flat_cls_targets_with_background,
              flat_cls_targets_with_background.get_shape()[1])
      else:
          one_hot_flat_cls_targets_with_background = tf.one_hot(
              one_hot_flat_cls_targets_with_background,
              tf.shape(flat_cls_targets_with_background)[1])

      # DEBUG: TO CHECK THE ONE HOT VECTOR FOR EACH PROPOSALS.
      # for idx, i in enumerate(tf.unstack(
      #         one_hot_flat_cls_targets_with_background)):
      #     one_hot_flat_cls_targets_with_background = tf.Print(
      #         one_hot_flat_cls_targets_with_background, [i],
      #         summarize=100, message=str(idx)+' : ')

      # If using a shared box across classes use directly
      if refined_box_encodings.shape[1] == 1:
        reshaped_refined_box_encodings = tf.reshape(
            refined_box_encodings,
            [batch_size, max_num_proposals, self._box_coder.code_size])
      # For anchors with multiple labels, picks refined_location_encodings
      # for just one class to avoid over-counting for regression loss and
      # (optionally) mask loss.
      else:
        # TODO : If code is running we can use the new function.
        # # We only predict refined location encodings for the non background
        # # classes, but we now pad it to make it compatible with the class
        # # predictions
        # refined_box_encodings_with_background = tf.pad(
        #     refined_box_encodings, [[0, 0], [1, 0], [0, 0]])
        # refined_box_encodings_masked_by_class_targets = tf.boolean_mask(
        #     refined_box_encodings_with_background,
        #     tf.greater(one_hot_flat_cls_targets_with_background, 0))
        # reshaped_refined_box_encodings = tf.reshape(
        #     refined_box_encodings_masked_by_class_targets,
        #     [batch_size, max_num_proposals, self._box_coder.code_size])
        reshaped_refined_box_encodings = (
            self._get_refined_encodings_for_postitive_class(
                refined_box_encodings,
                one_hot_flat_cls_targets_with_background, batch_size,
                max_num_proposals))

      losses_mask = None
      if self.groundtruth_has_field(fields.InputDataFields.is_annotated):
        losses_mask = tf.stack(self.groundtruth_lists(
            fields.InputDataFields.is_annotated))
      second_stage_loc_losses = self._second_stage_localization_loss(
          reshaped_refined_box_encodings,
          batch_reg_targets,
          weights=batch_reg_weights,
          losses_mask=losses_mask) / normalizer
      second_stage_cls_losses = ops.reduce_sum_trailing_dimensions(
          self._second_stage_classification_loss(
              class_predictions_with_background,
              batch_cls_targets_with_background,
              weights=batch_cls_weights,
              losses_mask=losses_mask),
          ndims=2) / normalizer

      second_stage_loc_loss = tf.reduce_sum(
          second_stage_loc_losses * tf.to_float(paddings_indicator))
      second_stage_cls_loss = tf.reduce_sum(
          second_stage_cls_losses * tf.to_float(paddings_indicator))

      if self._hard_example_miner:
        (second_stage_loc_loss, second_stage_cls_loss
        ) = self._unpad_proposals_and_apply_hard_mining(
                proposal_boxlists, second_stage_loc_losses,
                second_stage_cls_losses, num_proposals)

      localization_loss = tf.multiply(self._second_stage_loc_loss_weight,
                                      second_stage_loc_loss,
                                      name='localization_loss')

      classification_loss = tf.multiply(self._second_stage_cls_loss_weight,
                                        second_stage_cls_loss,
                                        name='classification_loss')

      loss_dict = {localization_loss.op.name: localization_loss,
                   classification_loss.op.name: classification_loss}

      object_prediction_masks = None
      second_stage_mask_loss = None
      if prediction_masks is not None:
        if groundtruth_masks_list is None:
          raise ValueError('Groundtruth instance masks not provided. '
                           'Please configure input reader.')

        unmatched_mask_label = tf.zeros(image_shape[1:3], dtype=tf.float32)
        (batch_mask_targets, _, _, batch_mask_target_weights,
         _) = target_assigner.batch_assign_targets(
             target_assigner=self._detector_target_assigner,
             anchors_batch=proposal_boxlists,
             gt_box_batch=groundtruth_boxlists,
             gt_class_targets_batch=groundtruth_masks_list,
             unmatched_class_label=unmatched_mask_label,
             gt_weights_batch=groundtruth_weights_list)

        # batch_size = prediction_masks.shape[0].value
        num_classes = prediction_masks.shape[1].value
        mask_height = prediction_masks.shape[2].value
        mask_width = prediction_masks.shape[3].value

        # Pad the prediction_masks with to add zeros for background class to be
        # consistent with class predictions.
        if num_classes == 1:
          # Class agnostic masks or masks for one-class prediction. Logic for
          # both cases is the same since background predictions are ignored
          # through the batch_mask_target_weights.
          prediction_masks_masked_by_class_targets = prediction_masks
        else:
          prediction_masks_with_background = tf.pad(
              prediction_masks, [[0, 0], [1, 0], [0, 0], [0, 0]])
          prediction_masks_masked_by_class_targets = tf.boolean_mask(
              prediction_masks_with_background,
              tf.greater(one_hot_flat_cls_targets_with_background, 0))

        reshaped_prediction_masks = tf.reshape(
            prediction_masks_masked_by_class_targets,
            [batch_size, -1, mask_height * mask_width])
        object_prediction_masks = tf.reshape(
            prediction_masks_masked_by_class_targets,
            [batch_size, -1, mask_height, mask_width])

        batch_mask_targets_shape = tf.shape(batch_mask_targets)
        flat_gt_masks = tf.reshape(batch_mask_targets,
                                   [-1, batch_mask_targets_shape[2],
                                    batch_mask_targets_shape[3]])

        # Use normalized proposals to crop mask targets from image masks.
        flat_normalized_proposals = box_list_ops.to_normalized_coordinates(
            box_list.BoxList(tf.reshape(proposal_boxes, [-1, 4])),
            image_shape[1], image_shape[2]).get()

        if self._is_training:
            total_proposals = flat_normalized_proposals.shape[0].value
        else:
            total_proposals = tf.shape(flat_normalized_proposals)[0]

        # TODO : If the code is running we can remove the following lines.
        # flat_cropped_gt_mask = tf.image.crop_and_resize(
        #     tf.expand_dims(flat_gt_masks, -1),
        #     tf.cast(flat_normalized_proposals, dtype=tf.float32),
        #     tf.range(total_proposals),
        #     [mask_height, mask_width])
        flat_cropped_gt_mask = self._crop_and_resize_fn(
            tf.expand_dims(flat_gt_masks, -1),
            tf.expand_dims(flat_normalized_proposals, axis=1),
            [mask_height, mask_width])

        batch_cropped_gt_mask = tf.reshape(
            flat_cropped_gt_mask,
            [batch_size, -1, mask_height * mask_width])

        second_stage_mask_losses = ops.reduce_sum_trailing_dimensions(
            self._second_stage_mask_loss(
                reshaped_prediction_masks,
                batch_cropped_gt_mask,
                weights=batch_mask_target_weights,
                losses_mask=losses_mask),
            ndims=2) / (
                mask_height * mask_width * tf.maximum(
                    tf.reduce_sum(
                        batch_mask_target_weights, axis=1, keepdims=True
                    ), tf.ones((batch_size, 1), dtype=tf.float32)))
        second_stage_mask_loss = tf.reduce_sum(
            tf.where(paddings_indicator, second_stage_mask_losses,
                     tf.zeros_like(second_stage_mask_losses)))

      if second_stage_mask_loss is not None:
        mask_loss = tf.multiply(self._second_stage_mask_loss_weight,
                                second_stage_mask_loss,
                                name='mask_loss')
        loss_dict[mask_loss.op.name] = mask_loss

    return loss_dict, object_prediction_masks

  def _get_refined_encodings_for_postitive_class(
      self, refined_box_encodings, flat_cls_targets_with_background,
      batch_size, max_num_proposals):
    # We only predict refined location encodings for the non background
    # classes, but we now pad it to make it compatible with the class
    # predictions
    refined_box_encodings_with_background = tf.pad(refined_box_encodings,
                                                   [[0, 0], [1, 0], [0, 0]])
    refined_box_encodings_masked_by_class_targets = (
        box_list_ops.boolean_mask(
            box_list.BoxList(
                tf.reshape(refined_box_encodings_with_background,
                           [-1, self._box_coder.code_size])),
            tf.reshape(tf.greater(flat_cls_targets_with_background, 0), [-1]),
            use_static_shapes=self._use_static_shapes,
            indicator_sum=batch_size * max_num_proposals
            if self._use_static_shapes else None).get())
    return tf.reshape(
        refined_box_encodings_masked_by_class_targets, [
            batch_size, max_num_proposals,
            self._box_coder.code_size
        ])

  def _loss_kl_divergence(self,
                          refined_box_encodings,
                          class_predictions_with_background,
                          proposal_boxes,
                          num_proposals,
                          image_shape,
                          prediction_masks,
                          semantic_logits,
                          semantic_mask=True):
    """ Calculates the kl divergence loss between mask and semantic and
    vice versa.

    Args:
      refined_box_encodings: a 3-D tensor with shape
        [total_num_proposals, num_classes, box_coder.code_size] representing
        predicted (final) refined box encodings. If using a shared box
        across classes this will instead have shape
        [total_num_proposals, 1, box_coder.code_size].
      class_predictions_with_background: a 2-D tensor with shape
        [total_num_proposals, num_classes + 1] containing class
        predictions (logits) for each of the anchors.  Note that this tensor
        *includes* background class predictions (at class index 0).
      proposal_boxes: [batch_size, self.max_num_proposals, 4] representing
        decoded proposal bounding boxes.
      num_proposals: A Tensor of type `int32`. A 1-D tensor of shape [batch]
        representing the number of proposals predicted for each image in
        the batch.
      image_shape: a 1-D tensor of shape [4] representing the image shape.
      prediction_masks: a 4-D tensor with shape
        [total_num_proposals, num_classes, mask_height, mask_width]
        containing the instance masks for each box.
      semantic_logits: a 4-D tensor with shape
        [batch_size, height, width, num_classes]
        semantic segmentation of the image.
      semantic_mask: Boolean flag.
        When True : KL(semantic||mask)
        When False : KL(mask||semantic)

    Returns:
        A tuple of (kl_loss_semantic_mask, kl_loss_mask_semantic) that is
        the reduced sum of kl divergence for each object.
    """
    with tf.name_scope('KLDivergenceLoss'):
        # 0. Stopping the gradient.
        refined_box_encodings = tf.stop_gradient(refined_box_encodings)
        class_predictions_with_background = tf.stop_gradient(
            class_predictions_with_background)
        proposal_boxes = tf.stop_gradient(proposal_boxes)
        num_proposals = tf.stop_gradient(num_proposals)
        if semantic_mask:
            semantic_logits = tf.stop_gradient(semantic_logits)
        else:
            prediction_masks = tf.stop_gradient(prediction_masks)
        # 1. Getting the needed information.
        batch_size = proposal_boxes.get_shape()[0]
        # 2. Getting the number of valid proposals.
        max_num_proposals = proposal_boxes.get_shape()[1]
        # 3. Processing the raw predictions.
        class_predictions_with_background_batch = tf.reshape(
            class_predictions_with_background,
            [batch_size, max_num_proposals, self.num_classes + 1])
        class_predictions_with_background_batch = (
            self._second_stage_score_conversion_fn(
                class_predictions_with_background_batch))
        class_predictions_batch = tf.reshape(
            tf.slice(class_predictions_with_background_batch,
                     [0, 0, 1], [-1, -1, -1]),
            [batch_size, max_num_proposals, self.num_classes])
        refined_box_encodings_batch = tf.reshape(
            refined_box_encodings,
            [batch_size, max_num_proposals, self.num_classes,
             self._box_coder.code_size])
        refined_decoded_boxes_batch = self._batch_decode_boxes(
            refined_box_encodings_batch, proposal_boxes)
        refined_decoded_boxes_batch = tf.reshape(
            refined_decoded_boxes_batch,
            [batch_size, max_num_proposals, self.num_classes, 4])
        tmp_shape = tf.reshape(
            tf.tile(image_shape[1:], [batch_size]),
            [batch_size, tf.shape(image_shape[1:])[0]])
        clip_window_batch = lib.compute_clip_window(tmp_shape)
        tmp_shape = prediction_masks.get_shape()
        prediction_masks_batch = tf.reshape(
            tf.sigmoid(prediction_masks),
            [batch_size, max_num_proposals, tmp_shape[-3], tmp_shape[-2],
             tmp_shape[-1]])
        # 4. Processing per batch (per image)
        # elems = [semantic_logits, refined_decoded_boxes_batch,
        #          class_predictions_batch, prediction_masks_batch,
        #          num_proposals, clip_window_batch]
        # kl_loss_semantic_mask, kl_loss_mask_semantic = \
        #     tf.map_fn(lib.kl_divergence, elems=elems,
        #               dtype=(tf.float32, tf.float32))
        # kl_loss_semantic_mask = tf.reduce_mean(kl_loss_semantic_mask)
        # kl_loss_mask_semantic = tf.reduce_mean(kl_loss_mask_semantic)
        # return kl_loss_semantic_mask, kl_loss_mask_semantic

        kl_loss = []
        for (single_image_semantic_logits,
             single_image_boxes,
             single_image_scores,
             single_image_masks,
             single_image_num_proposals,
             clip_window) in zip(
            tf.unstack(semantic_logits),
            tf.unstack(refined_decoded_boxes_batch),
            tf.unstack(class_predictions_batch),
            tf.unstack(prediction_masks_batch),
            tf.unstack(num_proposals),
            tf.unstack(clip_window_batch)
        ):
            elems = [single_image_semantic_logits,
                     single_image_boxes,
                     single_image_scores,
                     single_image_masks,
                     single_image_num_proposals,
                     clip_window,
                     semantic_mask]
            _kl_loss = lib.kl_divergence(elems)
            kl_loss.append(_kl_loss)
        kl_loss = tf.reduce_mean(tf.stack(kl_loss))
        return kl_loss

  def _unpad_proposals_and_apply_hard_mining(self,
                                             proposal_boxlists,
                                             second_stage_loc_losses,
                                             second_stage_cls_losses,
                                             num_proposals):
    """Unpads proposals and applies hard mining.

    Args:
      proposal_boxlists: A list of `batch_size` BoxLists each representing
        `self.max_num_proposals` representing decoded proposal bounding boxes
        for each image.
      second_stage_loc_losses: A Tensor of type `float32`. A tensor of shape
        `[batch_size, self.max_num_proposals]` representing per-anchor
        second stage localization loss values.
      second_stage_cls_losses: A Tensor of type `float32`. A tensor of shape
        `[batch_size, self.max_num_proposals]` representing per-anchor
        second stage classification loss values.
      num_proposals: A Tensor of type `int32`. A 1-D tensor of shape [batch]
        representing the number of proposals predicted for each image in
        the batch.

    Returns:
      second_stage_loc_loss: A scalar float32 tensor representing the second
        stage localization loss.
      second_stage_cls_loss: A scalar float32 tensor representing the second
        stage classification loss.
    """
    for (proposal_boxlist, single_image_loc_loss, single_image_cls_loss,
         single_image_num_proposals) in zip(
             proposal_boxlists,
             tf.unstack(second_stage_loc_losses),
             tf.unstack(second_stage_cls_losses),
             tf.unstack(num_proposals)):
      proposal_boxlist = box_list.BoxList(
          tf.slice(proposal_boxlist.get(),
                   [0, 0], [single_image_num_proposals, -1]))
      single_image_loc_loss = tf.slice(single_image_loc_loss,
                                       [0], [single_image_num_proposals])
      single_image_cls_loss = tf.slice(single_image_cls_loss,
                                       [0], [single_image_num_proposals])
      return self._hard_example_miner(
          location_losses=tf.expand_dims(single_image_loc_loss, 0),
          cls_losses=tf.expand_dims(single_image_cls_loss, 0),
          decoded_boxlist_list=[proposal_boxlist])

  def restore_map(self,
                  fine_tune_checkpoint_type='detection',
                  load_all_detection_checkpoint_vars=False,
                  **kwargs):
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
      return self._feature_extractor.restore_from_classification_checkpoint_fn(
          scope1=kwargs['mixed_precision_scope'],
          scope2=self.first_stage_feature_extractor_scope,
          scope3=self.second_stage_feature_extractor_scope)

    variables_to_restore = tf.global_variables()
    variables_to_restore.append(slim.get_or_create_global_step())
    # Only load feature extractor variables to be consistent with loading from
    # a classification checkpoint.
    include_patterns = None
    if not load_all_detection_checkpoint_vars:
      include_patterns = [
          self.first_stage_feature_extractor_scope,
          self.second_stage_feature_extractor_scope
      ]
    feature_extractor_variables = tf.contrib.framework.filter_variables(
        variables_to_restore, include_patterns=include_patterns)
    return {var.op.name: var for var in feature_extractor_variables}

  def _add_summary_prediction_image(self, params):
    """ Checking the validity of the training labels. Here we draw bounding
    box predictions only.

    Args:
        params: a tuple of tensors
    """
    (images, image_shape, batch_reg_targets, reshaped_refined_box_encodings,
     batch_cls_targets_with_background, class_predictions_with_background,
     batch_cropped_gt_mask, batch_mask_targets,  # Uncropped mask.
     reshaped_prediction_masks, groundtruth_boxlists, proposal_boxlists,
     num_proposals, flat_gt_masks, prediction_masks_masked_by_class_targets,
     mask_width) = params

    batch_size = images.shape.as_list()[0]

    with tf.name_scope('Masks'):
        if Options.masks_gt:
            _img = tf.expand_dims(
                tf.expand_dims(
                    tf.reduce_sum(flat_gt_masks, axis=0), axis=-1), axis=0)
            tf.summary.image("masks_gt", _img)
        if Options.masks_prediction:
            _img = tf.expand_dims(
                tf.expand_dims(
                    tf.reshape(prediction_masks_masked_by_class_targets,
                               [-1, mask_width]), axis=-1), axis=0)
            tf.summary.image("masks_prediction",
                             tf.cast(_img, dtype=tf.float32))

    with tf.name_scope('BoundingBoxes'):
        mask_shape = batch_cropped_gt_mask.shape.as_list()
        # 1. Ground truth BB.
        if Options.gt_bb:
            _image_gts = []
            for img, gt_boxlist in zip(
                    tf.unstack(images), groundtruth_boxlists):
                gt_boxlist_normalized = \
                    box_list_ops.to_normalized_coordinates(
                        gt_boxlist,
                        tf.shape(images)[1], tf.shape(images)[2],
                        check_range=False)
                gt_boxlist_normalized = gt_boxlist_normalized.get()
                gt_boxlist_normalized = tf.expand_dims(
                    gt_boxlist_normalized, axis=0)
                _image_gts.append(tf.image.draw_bounding_boxes(
                    tf.expand_dims(img, axis=0), gt_boxlist_normalized,
                    'gt'))
            tf.summary.image("gt_bb",
                             tf.concat(_image_gts, axis=0))

        # 2. Ground truth BB with full mask.
        if Options.gt_bb_full_mask:
            _refined_box_encodings = tf.tile(
                tf.expand_dims(
                    tf.unstack(batch_reg_targets, axis=0)[0], axis=1),
                [1, self.num_classes, 1])
            _class_predictions_with_background = tf.unstack(
                batch_cls_targets_with_background)[0]
            _mask_predictions = tf.tile(
                tf.expand_dims(
                    tf.unstack(
                        tf.reshape(batch_cropped_gt_mask,
                                   [mask_shape[0], -1,
                                    int(math.sqrt(mask_shape[2])),
                                    int(math.sqrt(mask_shape[2]))]),
                        axis=0)[0],
                    axis=1),
                [1, self.num_classes, 1, 1])
            _proposal_boxes = tf.expand_dims(
                proposal_boxlists[0].get(), axis=0)
            _num_proposals = tf.expand_dims(tf.unstack(num_proposals)[0], 0)
            _image_shapes = tf.expand_dims(image_shape[1:3], axis=0)
            local_detection_dict = self._postprocess_box_classifier(
                _refined_box_encodings,
                _class_predictions_with_background,
                _proposal_boxes,
                _num_proposals,
                _image_shapes,
                mask_predictions=_mask_predictions)
            local_detection_dict['detection_classes'] = tf.cast(
                local_detection_dict['detection_classes'], dtype=tf.uint8)
            # We fix it as 1 because we are only viewing the first image of
            # the batch.
            batch_size_1 = 1
            local_detection_dict['detection_boxes'].set_shape([
                batch_size_1,
                local_detection_dict['detection_boxes'].shape[1].value,
                local_detection_dict['detection_boxes'].shape[2].value])
            local_detection_dict['detection_scores'].set_shape([
                batch_size_1,
                local_detection_dict['detection_scores'].shape[1].value])
            local_detection_dict['detection_classes'].set_shape([
                batch_size_1,
                local_detection_dict['detection_classes'].shape[1].value])
            local_detection_dict['num_detections'].set_shape([
                batch_size_1])
            paddings_indicator = lib.padded_batched_proposals_indicator(
                tf.cast(local_detection_dict['num_detections'],
                        dtype=tf.int32),
                self._second_stage_nms_fn.keywords['max_total_size'])
            # INFO: we cannot directly use the original mask since it is too
            # big for memory.
            local_detection_dict['detection_masks'] = tf.cast(
                tf.greater(batch_mask_targets, 0.00001),
                dtype=tf.uint8)
            _boxes = lib.apply_boolean_mask(
                local_detection_dict['detection_boxes'],
                paddings_indicator)
            _classes = lib.apply_boolean_mask(
                local_detection_dict['detection_classes'],
                paddings_indicator)
            _scores = lib.apply_boolean_mask(
                local_detection_dict['detection_scores'],
                paddings_indicator)
            lib.draw_results(
                tf.expand_dims(tf.unstack(images)[0], axis=0),
                _boxes,
                self._category_index,
                classes=_classes,
                scores=_scores,
                instance_masks=None,
                min_score_thresh=0.02,
                max_boxes_to_draw=2000,
                use_normalized_coordinates=True,
                name='gt_bb_full_mask')

        # 3. Ground truth BB with reconstructed mask.
        if Options.gt_bb_cropped_mask:
            _refined_box_encodings = tf.tile(
                tf.expand_dims(
                    tf.unstack(batch_reg_targets, axis=0)[0], axis=1),
                [1, self.num_classes, 1])
            _class_predictions_with_background = tf.unstack(
                batch_cls_targets_with_background)[0]
            _mask_predictions = tf.tile(
                tf.expand_dims(
                    tf.unstack(
                        tf.reshape(batch_cropped_gt_mask,
                                   [mask_shape[0], -1,
                                    int(math.sqrt(mask_shape[2])),
                                    int(math.sqrt(mask_shape[2]))]),
                        axis=0)[0],
                    axis=1),
                [1, self.num_classes, 1, 1])
            _proposal_boxes = tf.expand_dims(
                proposal_boxlists[0].get(), axis=0)
            _num_proposals = tf.expand_dims(tf.unstack(num_proposals)[0], 0)
            _image_shapes = tf.expand_dims(image_shape[1:3], axis=0)
            local_detection_dict = self._postprocess_box_classifier(
                _refined_box_encodings,
                _class_predictions_with_background,
                _proposal_boxes,
                _num_proposals,
                _image_shapes,
                mask_predictions=_mask_predictions)
            paddings_indicator = lib.padded_batched_proposals_indicator(
                tf.cast(local_detection_dict['num_detections'],
                        dtype=tf.int32),
                self._second_stage_nms_fn.keywords['max_total_size'])
            local_detection_dict['detection_classes'] = tf.cast(
                local_detection_dict['detection_classes'], dtype=tf.uint8)
            m = local_detection_dict['detection_masks']
            m = ops.reframe_box_masks_to_image_masks(
                tf.unstack(m)[0],
                tf.unstack(local_detection_dict['detection_boxes'])[0],
                image_shape[1], image_shape[2])
            m = tf.cast(tf.greater(m, 0.5), dtype=tf.uint8)
            local_detection_dict['detection_masks'] =\
                tf.expand_dims(m, axis=0)
            lib.draw_results(
                tf.expand_dims(tf.unstack(images)[0], axis=0),
                tf.boolean_mask(
                    local_detection_dict['detection_boxes'],
                    paddings_indicator),
                self._category_index,
                classes=tf.boolean_mask(
                    local_detection_dict['detection_classes'],
                    paddings_indicator),
                scores=tf.boolean_mask(
                    local_detection_dict['detection_scores'],
                    paddings_indicator),
                instance_masks=tf.boolean_mask(
                    local_detection_dict['detection_masks'],
                    paddings_indicator),
                min_score_thresh=0.0,
                max_boxes_to_draw=2000,
                use_normalized_coordinates=True,
                name='Ground_truth_detection_targets_full_mask')

        # 4 Predictions
        if Options.predictions:
            refined_box_encodings = tf.tile(tf.expand_dims(
                reshaped_refined_box_encodings, axis=2),
                [1, 1, self.num_classes, 1])
            mask_predictions = tf.tile(
                tf.expand_dims(
                    tf.reshape(reshaped_prediction_masks,
                               [mask_shape[0], -1,
                                int(math.sqrt(mask_shape[2])),
                                int(math.sqrt(mask_shape[2]))]),
                    axis=2),
                [1, 1, self.num_classes, 1, 1])
            proposal_boxes = []
            for box in proposal_boxlists:
                proposal_boxes.append(tf.expand_dims(box.get(), axis=0))
            proposal_boxes = tf.concat(proposal_boxes, axis=0)
            image_shapes = tf.tile(
                tf.expand_dims(image_shape[1:3], axis=0),
                [batch_size, 1])
            refined_box_encodings = tf.reshape(
                refined_box_encodings,
                [-1,
                 refined_box_encodings.shape[2].value,
                 refined_box_encodings.shape[3].value])
            class_predictions_with_background = tf.reshape(
                class_predictions_with_background,
                [-1, class_predictions_with_background.shape[2].value])
            mask_predictions = tf.reshape(
                mask_predictions,
                [-1,
                 mask_predictions.shape[2].value,
                 mask_predictions.shape[3].value,
                 mask_predictions.shape[4].value])
            local_detection_dict = self._postprocess_box_classifier(
                refined_box_encodings,
                class_predictions_with_background,
                proposal_boxes,
                num_proposals,
                image_shapes,
                mask_predictions=mask_predictions)
            local_detection_dict['detection_classes'] = tf.cast(
                local_detection_dict['detection_classes'], dtype=tf.uint8)
            local_detection_dict['detection_masks'].set_shape(
                [batch_size,
                 local_detection_dict['detection_masks'].shape[1].value,
                 local_detection_dict['detection_masks'].shape[2].value,
                 local_detection_dict['detection_masks'].shape[3].value])
            local_detection_dict['detection_boxes'].set_shape(
                [batch_size,
                 local_detection_dict['detection_boxes'].shape[1].value,
                 local_detection_dict['detection_boxes'].shape[2].value])
            # _masks = []
            # for m, bb in zip(
            #         tf.unstack(local_detection_dict['detection_masks']),
            #         tf.unstack(local_detection_dict['detection_boxes'])):
            #     m = ops.reframe_box_masks_to_image_masks(
            #         m, bb, image_shape[1], image_shape[2])
            #     m = tf.cast(tf.greater(m, 0.5), dtype=tf.uint8)
            #     _masks.append(tf.cast(tf.expand_dims(m, axis=0),
            #                   dtype=tf.uint8))
            # local_detection_dict['detection_masks'] = tf.concat(
            #     _masks, axis=0)
            local_detection_dict['detection_masks'] = \
                ops.batch_reframe_box_masks_to_image_masks(
                    local_detection_dict['detection_masks'],
                    local_detection_dict['detection_boxes'],
                    image_shape[1], image_shape[2])
            local_detection_dict['detection_boxes'].set_shape([
                batch_size,
                local_detection_dict['detection_boxes'].shape[1].value,
                local_detection_dict['detection_boxes'].shape[2].value])
            local_detection_dict['detection_scores'].set_shape([
                batch_size,
                local_detection_dict['detection_scores'].shape[1].value])
            local_detection_dict['detection_classes'].set_shape([
                batch_size,
                local_detection_dict['detection_classes'].shape[1].value])
            local_detection_dict['num_detections'].set_shape([batch_size])
            local_detection_dict['detection_masks'].set_shape([
                batch_size,
                local_detection_dict['detection_masks'].shape[1].value,
                local_detection_dict['detection_masks'].shape[2].value,
                local_detection_dict['detection_masks'].shape[3].value])
            paddings_indicator = lib.padded_batched_proposals_indicator(
                tf.cast(local_detection_dict['num_detections'],
                        dtype=tf.int32),
                self._second_stage_nms_fn.keywords['max_total_size'])
            _images = tf.expand_dims(tf.unstack(images)[0], axis=0)
            _boxes = tf.expand_dims(
                tf.unstack(local_detection_dict['detection_boxes'])[0], axis=0)
            _classes = tf.expand_dims(
                tf.unstack(local_detection_dict['detection_classes'])[0],
                axis=0)
            _scores = tf.expand_dims(
                tf.unstack(local_detection_dict['detection_scores'])[0], axis=0)
            _masks = tf.expand_dims(
                tf.unstack(local_detection_dict['detection_masks'])[0], axis=0)
            paddings_indicator = tf.expand_dims(
                tf.unstack(paddings_indicator)[0], axis=0)
            _boxes = lib.apply_boolean_mask(_boxes, paddings_indicator)
            _classes = lib.apply_boolean_mask(_classes, paddings_indicator)
            _scores = lib.apply_boolean_mask(_scores, paddings_indicator)
            _masks = lib.apply_boolean_mask(_masks, paddings_indicator)

            for i in [0.02, 0.3, 0.5]:
                lib.draw_results(
                    _images,
                    _boxes,
                    self._category_index,
                    classes=_classes,
                    scores=_scores,
                    instance_masks=_masks,
                    min_score_thresh=i,
                    max_boxes_to_draw=2000,
                    use_normalized_coordinates=True,
                    name='predictions_' + str(int(i*100)))

  def _add_summary_semantic_image(self, params):
    """ Checking the validity of the training labels. Here we draw
    semantic segmentation only.

    Args:
        params: a tuple of tensors
    """
    (logit_prediction) = params
    # 1. Groundtruth
    gt_image = self.groundtruth_lists('semantic')[0]
    gt_image = tf.expand_dims(gt_image, axis=0)
    summary_label = vis.visualize_segmentation_labels(
        gt_image,
        lower_half=self.num_classes + 1,
        upper_half=-1)
    tf.summary.image('Things/Original', summary_label)
    summary_label = vis.visualize_segmentation_labels(
        gt_image,
        lower_half=-1,
        upper_half=self.num_classes)
    tf.summary.image('Stuff/Original', summary_label)
    # 2. Prediction
    predictions = tf.image.resize_nearest_neighbor(
        tf.expand_dims(
            tf.argmax(tf.nn.softmax(logit_prediction), axis=3), axis=-1),
        tf.shape(summary_label)[1:3],
        align_corners=True)
    predictions = tf.expand_dims(predictions[0], axis=0)
    summary_label = vis.visualize_segmentation_labels(
        predictions,
        lower_half=self.num_classes + 1,
        upper_half=-1)
    tf.summary.image('Things/Prediction', summary_label)
    summary_label = vis.visualize_segmentation_labels(
        predictions,
        lower_half=-1,
        upper_half=self.num_classes)
    tf.summary.image('Stuff/Prediction', summary_label)
    # 3. Prediction heat map
    prediction_heat_map = tf.image.resize_nearest_neighbor(
        tf.expand_dims(
            tf.reduce_max(tf.nn.softmax(logit_prediction), axis=3), axis=-1),
        tf.shape(summary_label)[1:3],
        align_corners=True)
    prediction_heat_map_0 = tf.cast(
        tf.round(prediction_heat_map[0] * 255.0), dtype=tf.uint8)
    prediction_heat_map_0 = tf.expand_dims(prediction_heat_map_0, axis=0)
    tf.summary.image('zHM/PredictionHeatMap', prediction_heat_map_0)

  @staticmethod
  def _add_summary_panoptic_image(params):
    """ Checking the validity of the training labels. Here we draw
    panoptic segmentation only.

    Args:
        params: a tuple of tensors
    """
    (instances, panoptic) = params
    tf.summary.image('Panoptic/instances', instances)
    tf.summary.image('Panoptic/panoptic', panoptic)
