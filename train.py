# Copyright 2020 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Trainer binary."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import functools
import os
import time

from absl import app
from absl import flags
from absl import logging
from flax import jax_utils
from flax import optim
from flax.metrics import tensorboard
from flax.training import checkpoints
import jax
from jax import lax
import jax.numpy as jnp
from fitvid import data
from fitvid import robomimic_data
from fitvid import models
from fitvid import utils
from fitvid.metrics import fvd
from fitvid.metrics import lpips
from fitvid.metrics import psnr
from fitvid.metrics import psnr_per_frame
from fitvid.metrics import ssim
import numpy as np
import tensorflow.compat.v2 as tf


FLAGS = flags.FLAGS

flags.DEFINE_string('output_dir', None, 'Path to model checkpoints/summaries.')
flags.DEFINE_boolean('depth_objective', None, 'Use depth image decoding as a auxiliary objective.')
flags.DEFINE_boolean('depth_weight', 1, 'Weighting on depth loss component')
flags.DEFINE_boolean('film', None, 'Use film action conditioning layers.')
flags.DEFINE_integer('batch_size', 32, 'Batch size.')
flags.DEFINE_integer('g_dim', 128, 'g dim.')
flags.DEFINE_integer('rnn_size', 256, 'rnn size.')
flags.DEFINE_integer('n_past', 2, 'Number of past frames.')
flags.DEFINE_integer('n_future', 10, 'Number of future frames.')
flags.DEFINE_integer('training_steps', 10000000, 'Number of training steps.')
flags.DEFINE_integer('log_every', 1000, 'How frequently log.')
flags.DEFINE_spaceseplist('dataset_file', [], 'Dataset to load.')


MODEL_CLS = models.FitVid


def additional_metrics(metrics, gt, out_video):
  metrics['metrics/psnr'] = psnr(gt, out_video)
  metrics['metrics/ssim'] = ssim(gt, out_video)
  metrics['metrics/fvd'] = fvd(gt, out_video)
  metrics['metrics/lpips'] = lpips(gt, out_video)
  return metrics


def write_summaries(summary_writer, metrics, step, vid_out, gt):
  """"Writes TensorBoard summaries."""
  # Scalar summaries
  for key, val in metrics.items():
    tag = key
    if key == 'graphs/psnr':
      image = utils.plot_1d_signals([np.mean(val, axis=0)], [''])
      summary_writer.image(tag=tag, image=image, step=step)
    elif key.startswith('hist'):
      summary_writer.histogram(tag, val, step)
    else:
      summary_writer.scalar(tag, val, step)
  # GIFs
  video_summary = np.concatenate([gt, vid_out], axis=3)
  utils.write_video_summaries(summary_writer, video_summary, 1, step)
  summary_writer.flush()


@functools.partial(jax.pmap, axis_name='batch', static_broadcasted_argnums=0)
def eval_step(model, batch, state, rng):
  """A single evaluation step."""
  variables = {'params': state.optimizer.target, **state.model_state}
  (_, out_video, metrics) = model.apply(
      variables,
      video=batch['video'],
      actions=batch['actions'],
      depth_video=batch.get('depth_video', None),
      rngs=utils.generate_rng_dict(rng),
      step=state.step,)
      #mutable=['batch_stats'])
  n_past = FLAGS.n_past
  out_video = jax.lax.all_gather(out_video[:, n_past-1:], axis_name='batch')
  gt = jax.lax.all_gather(batch['video'][:, n_past:], axis_name='batch')
  metrics = jax.lax.all_gather(metrics, axis_name='batch')
  return gt, out_video, metrics


@functools.partial(
    jax.pmap, axis_name='batch',
    static_broadcasted_argnums=0, donate_argnums=(2,))
def train_step(model, batch, state, rng):
  """A single training step."""
  def loss(params):
    variables = {'params': params, **state.model_state}
    (loss, out_video, metrics), new_model_state = model.apply(
        variables,
        video=batch['video'],
        actions=batch['actions'],
        depth_video=batch.get('depth_video', None),
        rngs=utils.generate_rng_dict(rng),
        step=state.step,
        mutable=['batch_stats'])
    return loss, (new_model_state, out_video, metrics)

  optimizer = state.optimizer
  grad_fn = jax.value_and_grad(loss, has_aux=True)
  aux, grads = grad_fn(optimizer.target)
  new_model_state, out_video, metrics = aux[1]
  grads = lax.pmean(grads, axis_name='batch')
  # metrics = jax.lax.pmean(metrics, axis_name='batch')
  grads_clipped = utils.clip_grads(grads, 100.0)
  new_optimizer = optimizer.apply_gradient(grads_clipped)
  # Apply update if the new optimizer state is all finite
  ok = jnp.all(jnp.asarray([
      jnp.all(jnp.isfinite(p)) for p in jax.tree_leaves(new_optimizer)]))
  new_state_with_update = state.replace(
      step=state.step + 1,
      optimizer=new_optimizer,
      model_state=new_model_state)
  new_state_no_update = state.replace(
      step=state.step + 1)
  new_state = jax.tree_multimap(
      lambda a, b: jnp.where(ok, a, b),
      new_state_with_update, new_state_no_update)
  rng = jax.random.split(rng)[1]
  new_state = state.replace(
      step=state.step + 1, optimizer=new_optimizer, model_state=new_model_state)
  return new_state, rng, metrics, out_video


def get_log_directories():
  output_dir = FLAGS.output_dir
  model_dir = os.path.join(output_dir, 'model')
  log_dir = os.path.join(output_dir, 'train')
  eval_log_dir = os.path.join(output_dir, 'evaluate')
  summary_writer = tensorboard.SummaryWriter(log_dir)
  eval_summary_writer = tensorboard.SummaryWriter(eval_log_dir)
  return model_dir, summary_writer, eval_summary_writer


def get_data(training, depth=False):
  video_len = FLAGS.n_past + FLAGS.n_future
  local_batch_size = FLAGS.batch_size // jax.process_count()
    
  if FLAGS.dataset_file is None:
    assert False, "dataset file not specified!"
    print('Loading RoboNet dataset...')
    return data.load_dataset_robonet(local_batch_size, video_len, training)
  else:
    print(f'Loading RoboMimic dataset from files {FLAGS.dataset_file}...')
    return robomimic_data.load_dataset_robomimic(FLAGS.dataset_file, local_batch_size, video_len, training, depth)


def init_model_state(rng_key, model, sample):
  """Initialize the model state."""
  variables = model.init(
      rngs=utils.generate_rng_dict(rng_key),
      video=sample['video'],
      actions=sample['actions'],
      depth_video=sample.get('depth_video', None),
      step=0)
  model_state, params = variables.pop('params')

  optimizer_def = optim.Adam(learning_rate=1e-3)
  optimizer = optimizer_def.create(params)
  utils.print_model_size(params, 'Model Size')
  return utils.TrainState(step=0, optimizer=optimizer, model_state=model_state)


def evaluate(rng_key, state, model, data_itr, eval_steps):
  """Evaluates the model on the entire dataset."""
  all_metrics = []
  gts = []
  out_videos = []
  for _ in range(eval_steps):
    batch = next(data_itr)
    gt, out_video, metrics = eval_step(model, batch, state, rng_key)

    def get_all(x):
      return utils.get_all_devices(jax_utils.unreplicate(x))
    out_video = get_all(out_video)
    gt = get_all(gt)
    metrics = jax.tree_map(get_all, metrics)
    metrics = additional_metrics(metrics, gt, out_video)
    all_metrics.append(metrics)
    out_videos.append(out_video)
    gts.append(gt)

  if jax.host_id() == 0:
    metrics = {
        k: np.mean([dic[k] for dic in all_metrics]) for k in all_metrics[0]}
    metrics['graphs/psnr'] = psnr_per_frame(gt, out_video)
  return metrics, gts, out_videos


def train():
  """Main training loop."""
  rng_key = jax.random.PRNGKey(0)
  training_steps = FLAGS.training_steps
  log_every = FLAGS.log_every

  model_dir, summary_writer, eval_summary_writer = get_log_directories()
  data_itr = get_data(True, depth=FLAGS.depth_objective)
  eval_data_itr = get_data(True, depth=FLAGS.depth_objective)

  batch = next(data_itr)
  sample = utils.get_first_device(batch)

  model = MODEL_CLS(n_past=FLAGS.n_past,
                    g_dim=FLAGS.g_dim,
                    rnn_size=FLAGS.rnn_size,
                    training=True,
                    depth_head=FLAGS.depth_objective,
                    depth_weight=FLAGS.depth_weight,
                    use_film=FLAGS.film)
  state = init_model_state(rng_key, model, sample)
  state = checkpoints.restore_checkpoint(model_dir, state)
  start_step = int(state.step)
  state = jax_utils.replicate(state)

  rng_key = jax.random.split(rng_key, jax.local_device_count())
  t_loop_start = time.time()
  for step in range(start_step, training_steps):
    output = train_step(model, batch, state, rng_key)
    state, rng_key, metrics, out_video = output

    if step % log_every == 0:
      state = utils.sync_batch_stats(state)
      steps_per_sec = log_every / (time.time() - t_loop_start)
      t_loop_start = time.time()

      if jax.host_id() == 0:
        train_metrics = utils.get_average_across_devices(metrics)
        state_ = jax_utils.unreplicate(state)
        checkpoints.save_checkpoint(model_dir, state_, step, keep=3)
        summary_writer.scalar('info/steps-per-second', steps_per_sec, step)
        out_video = utils.get_all_devices(out_video)
        gt = utils.get_all_devices(batch['video'])[:, 1:]
        train_metrics = additional_metrics(train_metrics, gt, out_video)
        train_metrics['graphs/psnr'] = psnr_per_frame(gt, out_video)
        write_summaries(summary_writer, train_metrics, step, out_video, gt)
        logging.info('>>> Step: %d Loss: %.4f', step, train_metrics['loss/all'])

      eval_batch = next(eval_data_itr)
      model.training = False
      eval_output = eval_step(model, eval_batch, state, rng_key)
      eval_gt, eval_out, eval_metrics = eval_output
      model.training = True

      if jax.host_id() == 0:
        # Log eval
        eval_metrics = utils.get_average_across_devices(eval_metrics)
        eval_out = utils.get_all_devices(jax_utils.unreplicate(eval_out))
        eval_gt = utils.get_all_devices(eval_batch['video'])[:, FLAGS.n_past:]
        eval_metrics = additional_metrics(eval_metrics, eval_gt, eval_out)
        eval_metrics['graphs/psnr'] = psnr_per_frame(eval_gt, eval_out)
        write_summaries(eval_summary_writer, eval_metrics, step, eval_out, eval_gt)

    batch = next(data_itr)


def main(argv):
  del argv  # Unused
  tf.enable_v2_behavior()
  # make sure tf does not allocate gpu memory
  tf.config.experimental.set_visible_devices([], 'GPU')
  train()
  # Wait until computations are done before exiting
  jax.random.normal(jax.random.PRNGKey(0), ()).block_until_ready()


if __name__ == '__main__':
  flags.mark_flags_as_required(['output_dir'])
  app.run(main)
