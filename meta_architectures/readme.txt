################################################################################
Description for faster_rcnn_meta_arch.py

"""Faster R-CNN meta-architecture definition.

General tensorflow implementation of Faster R-CNN detection models.

See Faster R-CNN: Ren, Shaoqing, et al.
"Faster R-CNN: Towards real-time object detection with region proposal
networks." Advances in neural information processing systems. 2015.

We allow for three modes: number_of_stages={1, 2, 3}. In case of 1 stage,
all of the user facing methods (e.g., predict, postprocess, loss) can be used as
if the model consisted only of the RPN, returning class agnostic proposals
(these can be thought of as approximate detections with no associated class
information).  In case of 2 stages, proposals are computed, then passed
through a second stage "box classifier" to yield (multi-class) detections.
Finally, in case of 3 stages which is only used during eval, proposals are
computed, then passed through a second stage "box classifier" that will compute
refined boxes and classes, and then features are pooled from the refined and
non-maximum suppressed boxes and are passed through the box classifier again. If
number of stages is 3 during training it will be reduced to two automatically.

Implementations of Faster R-CNN models must define a new
FasterRCNNFeatureExtractor and override three methods: `preprocess`,
`_extract_proposal_features` (the first stage of the model), and
`_extract_box_classifier_features` (the second stage of the model). Optionally,
the `restore_fn` method can be overridden.  See tests for an example.

A few important notes:
+ Batching conventions:  We support batched inference and training where
all images within a batch have the same resolution.  Batch sizes are determined
dynamically via the shape of the input tensors (rather than being specified
directly as, e.g., a model constructor).

A complication is that due to non-max suppression, we are not guaranteed to get
the same number of proposals from the first stage RPN (region proposal network)
for each image (though in practice, we should often get the same number of
proposals).  For this reason we pad to a max number of proposals per image
within a batch. This `self.max_num_proposals` property is set to the
`first_stage_max_proposals` parameter at inference time and the
`second_stage_batch_size` at training time since we subsample the batch to
be sent through the box classifier during training.

For the second stage of the pipeline, we arrange the proposals for all images
within the batch along a single batch dimension.  For example, the input to
_extract_box_classifier_features is a tensor of shape
`[total_num_proposals, crop_height, crop_width, depth]` where
total_num_proposals is batch_size * self.max_num_proposals.  (And note that per
the above comment, a subset of these entries correspond to zero paddings.)

+ Coordinate representations:
Following the API (see model.DetectionModel definition), our outputs after
postprocessing operations are always normalized boxes however, internally, we
sometimes convert to absolute --- e.g. for loss computation.  In particular,
anchors and proposal_boxes are both represented as absolute coordinates.

Images are resized in the `preprocess` method.

The Faster R-CNN meta architecture has two post-processing methods
`_postprocess_rpn` which is applied after first stage and
`_postprocess_box_classifier` which is applied after second stage. There are
three different ways post-processing can happen depending on number_of_stages
configured in the meta architecture:

1. When number_of_stages is 1:
  `_postprocess_rpn` is run as part of the `postprocess` method where
  true_image_shapes is used to clip proposals, perform non-max suppression and
  normalize them.
2. When number of stages is 2:
  `_postprocess_rpn` is run as part of the `_predict_second_stage` method where
  `resized_image_shapes` is used to clip proposals, perform non-max suppression
  and normalize them. In this case `postprocess` method skips `_postprocess_rpn`
  and only runs `_postprocess_box_classifier` using `true_image_shapes` to clip
  detections, perform non-max suppression and normalize them.
3. When number of stages is 3:
  `_postprocess_rpn` is run as part of the `_predict_second_stage` using
  `resized_image_shapes` to clip proposals, perform non-max suppression and
  normalize them. Subsequently, `_postprocess_box_classifier` is run as part of
  `_predict_third_stage` using `true_image_shapes` to clip detections, peform
  non-max suppression and normalize them. In this case, the `postprocess` method
  skips both `_postprocess_rpn` and `_postprocess_box_classifier`.

+ Info about 'prediction_dict' updated, the ones in the code might not be.
prediction_dict: a dictionary holding "raw" prediction tensors:
1) rpn_box_predictor_features: A list of 4-D float32 tensor with shape
  [batch_size, height, width, depth] to be used for predicting proposal
  boxes and corresponding objectness scores.
2) rpn_features_to_crop: A list of 4-D float32 tensor with shape
  [batch_size, height, width, depth] representing image features to crop
  using the proposal boxes predicted by the RPN.
3) image_shape: a 1-D tensor of shape [4] representing the input
  image shape.
4) rpn_box_encodings: A list of 3-D float tensor of shape
  [batch_size, num_anchors, self._box_coder.code_size] containing
  predicted boxes.
5) rpn_objectness_predictions_with_background: A list of 3-D float tensor of
  shape [batch_size, num_anchors, 2] containing class
  predictions (logits) for each of the anchors.  Note that this
  tensor *includes* background class predictions (at class index 0).
6) anchors: A  list of 2-D tensor of shape [num_anchors, 4] representing anchors
  for the first stage RPN (in absolute coordinates).  Note that
  `num_anchors` can differ depending on whether the model is created in
  training or inference mode.
7) anchors_boxlist: Same as (6) but as 'boxlist' object instead of absolute
  coordinates.
(and if number_of_stages > 1):
8) refined_box_encodings: a 3-D tensor with shape
  [total_num_proposals, num_classes, self._box_coder.code_size]
  representing predicted (final) refined box encodings, where
  total_num_proposals=batch_size*self._max_num_proposals. If using
  a shared box across classes the shape will instead be
  [total_num_proposals, 1, self._box_coder.code_size].
9) class_predictions_with_background: a 3-D tensor with shape
  [total_num_proposals, num_classes + 1] containing class
  predictions (logits) for each of the anchors, where
  total_num_proposals=batch_size*self._max_num_proposals.
  Note that this tensor *includes* background class predictions
  (at class index 0).
10) num_proposals: An int32 tensor of shape [batch_size] representing the
  number of proposals generated by the RPN.  `num_proposals` allows us
  to keep track of which entries are to be treated as zero paddings and
  which are not since we always pad the number of proposals to be
  `self.max_num_proposals` for each image.
11) proposal_boxes: A float32 tensor of shape
  [batch_size, self.max_num_proposals, 4] representing
  decoded proposal bounding boxes in absolute coordinates.
12) proposal_boxes_normalized: A float32 tensor of shape
  [batch_size, self.max_num_proposals, 4] representing decoded proposal
  bounding boxes in normalized coordinates. Can be used to override the
  boxes proposed by the RPN, thus enabling one to extract features and
  get box classification and prediction for externally selected areas
  of the image.
13) box_classifier_features: a 4-D float32 tensor representing the
  features for each proposal.
14) mask_predictions: (optional) a 4-D tensor with shape
  [total_num_padded_proposals, num_classes, mask_height, mask_width]
  containing instance mask predictions.
15) semantic_predictions: a 4-D tensor with shape
  [batch_size, height, width, num_semantic_classes] containing semantic mask
  predictions.
16) final_top_down_feature: Final feature layer of the top down module in
  FPN. Used for semantic segmentation.
"""
################################################################################
