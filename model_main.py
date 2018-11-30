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
"""Binary to run train and evaluation on object detection model."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl import flags

import tensorflow as tf

from object_detection import model_hparams
from object_detection import model_lib

# tf.logging.set_verbosity(tf.logging.INFO)
tf.logging.set_verbosity(tf.logging.DEBUG)

flags.DEFINE_string(
    'model_dir', None, 'Path to output model directory '
    'where event and checkpoint files will be written.')
flags.DEFINE_string(
    'restore_dir', None, 'Path to model directory that we want to restore from'
    'where checkpoint files is to be found.')
flags.DEFINE_string('pipeline_config_path', None, 'Path to pipeline config '
                    'file.')
flags.DEFINE_boolean('eval_training_data', False,
                     'If training data should be evaluated for this job. Note '
                     'that one call only use this in eval-only mode, and '
                     '`checkpoint_dir` must be supplied.')
flags.DEFINE_integer('sample_1_of_n_eval_examples', 1, 'Will sample one of '
                     'every n eval input examples, where n is provided.')
flags.DEFINE_integer('sample_1_of_n_eval_on_train_examples', 5, 'Will sample '
                     'one of every n train input examples for evaluation, '
                     'where n is provided. This is only used if '
                     '`eval_training_data` is True.')
flags.DEFINE_string(
    'hparams_overrides', None, 'Hyperparameter overrides, '
    'represented as a string containing comma-separated '
    'hparam_name=value pairs.')
flags.DEFINE_string(
    'checkpoint_dir', None, 'Path to directory holding a checkpoint.  If '
    '`checkpoint_dir` is provided, this binary operates in eval-only mode, '
    'writing resulting metrics to `model_dir`.')
flags.DEFINE_boolean(
    'run_once', False, 'If running in eval-only mode, whether to run just '
    'one round of eval vs running continuously (default).'
)
flags.DEFINE_string('dataset_name', 'coco',
                    'dataset name.')
flags.DEFINE_string('images_json_file', '',
                    'JSON file containing the image set information.')
flags.DEFINE_string('panoptic_gt_folder', '',
                    'Groundtruth folder for panoptic segmentation.')
flags.DEFINE_string('panoptic_export_folder', None,
                    'Export folder for panoptic segmentation.')
flags.DEFINE_boolean('distribute', False,
                     'Distributed training or not.')

flags.DEFINE_integer('num_train_steps', None, 'Number of train steps.')
flags.DEFINE_integer('num_eval_steps', None, 'Number of eval steps.')
flags.DEFINE_integer('log_step_count_steps', 100, 'Number of log steps.')
flags.DEFINE_integer('save_summary_steps', 1000, 'Number of summary steps.')
flags.DEFINE_integer('keep_checkpoint_max', None, 'Number of eval steps.')
flags.DEFINE_integer('throttle_secs', 7200, 'Evaluation interval.')
flags.DEFINE_integer('start_delay_secs', 14400,
                     'Delay in evaluation. (default 4h)')
flags.DEFINE_integer('save_checkpoints_secs', 3600, 'Checkpoint save interval.')
flags.DEFINE_integer('keep_checkpoint_every_n_hours', 3, 'Master save of ckpt.')

FLAGS = flags.FLAGS


def distribution_strategy(num_gpus):
    if num_gpus == 1:
        return tf.contrib.distribute.OneDeviceStrategy(device='/gpu:0')
    elif num_gpus > 1:
        return tf.contrib.distribute.MirroredStrategy(num_gpus=num_gpus)
    else:
        return None


def main(unused_argv):
  del unused_argv

  flags.mark_flag_as_required('model_dir')
  flags.mark_flag_as_required('pipeline_config_path')
  if FLAGS.panoptic_export_folder == '':
      raise ValueError("It is not recommended to give an empty path, please "
                       "use 'None' instead...")

  # [CONFIGS] ******************************************************************
  # Soft placement allows placing on CPU ops without GPU implementation.
  # gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.9,
  #                             allow_growth=True)
  session_config = tf.ConfigProto(allow_soft_placement=True,
                                  log_device_placement=False)
  session_config.gpu_options.allow_growth = True

  # It is better to use secs because:
  # - ckpts are about faulty recovery, step is irrelevant.
  # - it is easier to use with throttle_secs for train_and_evaluate(...)
  run_config = tf.estimator.RunConfig(
      model_dir=FLAGS.model_dir,
      session_config=session_config,
      keep_checkpoint_max=FLAGS.keep_checkpoint_max,
      save_checkpoints_secs=FLAGS.save_checkpoints_secs,
      keep_checkpoint_every_n_hours=FLAGS.keep_checkpoint_every_n_hours,
      log_step_count_steps=FLAGS.log_step_count_steps,
      save_summary_steps=FLAGS.save_summary_steps,
      train_distribute=tf.contrib.distribute.MirroredStrategy()
      if FLAGS.distribute else None)

  # issue 21405 : we need to remove tf.contrib.layers and use tf.layers.
  # train_distribute=tf.contrib.distribute.MirroredStrategy(
  #     devices=["/device:CPU:0",
  #              "/device:GPU:0"]) if FLAGS.distribute else None)

  # ****************************************************************** [CONFIGS]

  # [Setup Estimator and Inputs] ***********************************************
  train_and_eval_dict = model_lib.create_estimator_and_inputs(
      run_config=run_config,
      hparams=model_hparams.create_hparams(FLAGS.hparams_overrides),
      pipeline_config_path=FLAGS.pipeline_config_path,
      train_steps=FLAGS.num_train_steps,
      eval_steps=FLAGS.num_eval_steps,
      dataset_name=FLAGS.dataset_name,
      images_json_file=FLAGS.images_json_file,
      panoptic_gt_folder=FLAGS.panoptic_gt_folder,
      panoptic_export_folder=FLAGS.panoptic_export_folder,
      sample_1_of_n_eval_examples=FLAGS.sample_1_of_n_eval_examples,
      sample_1_of_n_eval_on_train_examples=(
          FLAGS.sample_1_of_n_eval_on_train_examples),
      restore_dir=FLAGS.restore_dir)
  estimator = train_and_eval_dict['estimator']
  train_input_fn = train_and_eval_dict['train_input_fn']
  eval_input_fns = train_and_eval_dict['eval_input_fns']
  eval_on_train_input_fn = train_and_eval_dict['eval_on_train_input_fn']
  predict_input_fn = train_and_eval_dict['predict_input_fn']
  train_steps = train_and_eval_dict['train_steps']
  eval_steps = train_and_eval_dict['eval_steps']
  # *********************************************** [Setup Estimator and Inputs]

  if FLAGS.checkpoint_dir:
    # [Evaluation Only] ********************************************************
    if FLAGS.eval_training_data:
      name = 'training_data'
      input_fn = eval_on_train_input_fn
    else:
      name = 'validation_data'
      # The first eval input will be evaluated.
      input_fn = eval_input_fns[0]
    if FLAGS.run_once:
      estimator.evaluate(input_fn,
                         eval_steps,
                         checkpoint_path=tf.train.latest_checkpoint(
                             FLAGS.checkpoint_dir))
    else:
      model_lib.continuous_eval(estimator, FLAGS.checkpoint_dir, input_fn,
                                train_steps, name)
  else:
    # [Training and Evaluation] ************************************************
    train_spec, eval_specs = model_lib.create_train_and_eval_specs(
        train_input_fn,
        eval_input_fns,
        eval_on_train_input_fn,
        predict_input_fn,
        train_steps,
        eval_steps,
        eval_on_train_data=False,
        start_delay_secs=FLAGS.start_delay_secs,
        throttle_secs=FLAGS.throttle_secs)
    # Currently only a single Eval Spec is allowed.
    tf.estimator.train_and_evaluate(estimator, train_spec, eval_specs[0])
    # estimator.train(input_fn=train_input_fn, max_steps=FLAGS.num_train_steps)
    # estimator.evaluate(input_fn=eval_input_fns[0], steps=FLAGS.num_eval_steps)
    # ************************************************ [Training and Evaluation]

    # [PROFILING] **************************************************************
    # Option 1:
    # with tf.contrib.tfprof.ProfileContext(
    #         '/tmp/train_dir', trace_steps=range(0, 10), dump_steps=[9]):
    #     estimator.train(input_fn=train_input_fn, max_steps=10)
    #
    # # Option 2:
    # profiler_hook = tf.train.ProfilerHook(
    #     save_steps=1, output_dir='/tmp/train_dir', show_memory=True)
    # estimator.train(input_fn=train_input_fn, max_steps=15,
    #                 hooks=[profiler_hook])
    # ************************************************************** [PROFILING]


if __name__ == '__main__':
  tf.app.run()
