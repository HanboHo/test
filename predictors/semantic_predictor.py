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

"""Full Image Semantic Predictor."""
import tensorflow as tf

slim = tf.contrib.slim


class SemanticPredictor(object):
  """Full Image Semantic Predictor.

  """

  def __init__(self,
               is_training,
               num_classes,
               depth,
               kernel_size,
               arg_scope_fn,
               backbone=None):
    """Constructor.

    Args:
      is_training: Indicates whether the BoxPredictor is in training mode.
      num_classes: number of classes.  Note that num_classes *does not*
        include the background category, so if groundtruth labels take values
        in {0, 1, .., K-1}, num_classes=K (and not K+1, even though the
        assigned classification targets can range from {0,... K}).
      box_prediction_head: The head that predicts the boxes in second stage.
      class_prediction_head: The head that predicts the classes in second stage.
      third_stage_heads: A dictionary mapping head names to mask rcnn head
        classes.
    """
    self._num_classes = num_classes
    self._depth = depth
    self._kernel_size = kernel_size
    self._arg_scope_fn = arg_scope_fn
    self._backbone = backbone

  @property
  def semantic_predictor_scope(self):
    return 'SemanticPredictor'

  # @property
  # def num_classes(self):
  #   return self._num_classes
  #
  # def get_prediction_head_backbone(self):
  #   return self._backbone
  #
  # def get_second_stage_prediction_heads(self):
  #   return BOX_ENCODINGS, CLASS_PREDICTIONS_WITH_BACKGROUND
  #
  # def get_third_stage_prediction_heads(self):
  #   return sorted(self._third_stage_heads.keys())

  def predict_semantic_labels(self, semantic_predictor_features,
                              aspp_with_batch_norm=None,
                              atrous_rates=None,
                              kernel_size=1):
    """Uses the last layer lowest feature level of the FPN to predict the
    semantic labels.

    Note resulting tensors are not postprocessed.

    Args:
        semantic_predictor_features: A 4-D float32 tensor with shape
            [batch, height, width, depth] to be used for predicting
            semantic labels / scores.
        atrous_rates: A list of atrous convolution rates for last layer.
        aspp_with_batch_norm: Use batch normalization layers for ASPP.
        kernel_size: Kernel size for convolution.

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
    # When using batch normalization with ASPP, ASPP has been applied before
    # in extract_features, and thus we simply apply 1x1 convolution here.
    if aspp_with_batch_norm or atrous_rates is None:
        if kernel_size != 1:
            raise ValueError(
                'Kernel size must be 1 when atrous_rates is None or '
                'using aspp_with_batch_norm. Gets %d.' % kernel_size)
        atrous_rates = [1]

    with slim.arg_scope(self._arg_scope_fn):
        with tf.variable_scope(self.semantic_predictor_scope,
                               self.semantic_predictor_scope,
                               [semantic_predictor_features]):
            num_convs = 2
            decoder_features = slim.repeat(
                semantic_predictor_features,
                num_convs,
                slim.conv2d,
                self._depth,
                self._kernel_size,
                scope='decoder_conv')

            # # with tf.name_scope('decoder_conv'):
            # #     decoder_features = slim.conv2d(
            # #         semantic_predictor_features,
            # #         self._first_stage_semantic_depth,
            # #         kernel_size=self._first_stage_semantic_kernel_size,
            # #         activation_fn=tf.nn.relu,
            # #         scope='decoder_conv_1')
            # #     decoder_features = tf.image.resize_bilinear(
            # #         decoder_features,
            # #         tf.shape(decoder_features)[1:3] * 2,
            # #         align_corners=True)
            # #     decoder_features = slim.conv2d(
            # #         decoder_features,
            # #         self._first_stage_semantic_depth,
            # #         kernel_size=self._first_stage_semantic_kernel_size,
            # #         activation_fn=tf.nn.relu,
            # #         scope='decoder_conv_2')
            # semantic_logits = slim.conv2d(
            #     decoder_features,
            #     self._num_classes,
            #     kernel_size=1,
            #     activation_fn=None,
            #     normalizer_fn=None,
            #     scope='semantic_logits')
            # return semantic_logits

            # Taken straight from Deeplabv3+
            branch_logits = []
            for i, rate in enumerate(atrous_rates):
                scope = 'semantic_logit_branch'
                if i:
                    scope += '_%d' % i
                branch_logits.append(
                    slim.conv2d(
                        decoder_features,
                        self._num_classes,
                        kernel_size=kernel_size,
                        rate=rate,
                        activation_fn=None,
                        normalizer_fn=None,
                        scope=scope))
            return tf.add_n(branch_logits, name='semantic_logits')


