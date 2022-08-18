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

"""Eval binary."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import time

from absl import app
from absl import flags
from fitvid.scripts.train_fitvid import load_data, dict_to_cuda, depth_to_rgb_im
from fitvid.model.fitvid import FitVid
import torch
import numpy as np
import matplotlib.pyplot as plt

FLAGS = flags.FLAGS


def log_eval_gifs():
    """Evaluates the latest model checkpoint."""
    import random

    random.seed(0)
    torch.manual_seed(0)
    np.random.seed(0)
    os.environ["PYTHONHASHSEED"] = str(0)

    model = FitVid(
        stage_sizes=[int(i) for i in FLAGS.stage_sizes],
        z_dim=FLAGS.z_dim,
        g_dim=FLAGS.g_dim,
        rnn_size=FLAGS.rnn_size,
        num_base_filters=FLAGS.num_base_filters,
        first_block_shape=[int(i) for i in FLAGS.first_block_shape],
        skip_type=FLAGS.skip_type,
        n_past=FLAGS.n_past,
        action_conditioned=eval(FLAGS.action_conditioned),
        action_size=4,  # hardcode for now
        is_inference=False,
        has_depth_predictor=FLAGS.depth_objective,
        expand_decoder=FLAGS.expand,
        beta=FLAGS.beta,
        depth_weight=FLAGS.depth_weight,
    )
    NGPU = torch.cuda.device_count()
    print("CUDA available devices: ", NGPU)

    data_itr, prep_data = load_data(FLAGS.dataset_file, "valid", depth=True)
    data_itr = iter(data_itr)
    batch = next(data_itr)

    checkpoint = FLAGS.output_dir
    print(
        f'Attempting to load from checkpoint {checkpoint if checkpoint else "None, no model found"}'
    )
    if checkpoint:
        model.load_parameters(checkpoint)
    model.eval()
    model.cuda()

    preds_all, depth_preds_all = [], []
    gt, gt_depths = [], []
    for i in range(10):
        batch = next(data_itr)
        batch = prep_data(batch)
        with torch.no_grad():
            _, preds = model.evaluate(dict_to_cuda(batch))
        if FLAGS.depth_objective:
            preds, depth_preds = preds
            depth_preds_all.append(depth_preds)
            gt_depths.append(batch["depth_video"][:, 1:])
        preds_all.append(preds)
        gt.append(batch["video"][:, 1:])

    def numpify(x):
        x = x.permute(0, 1, 3, 4, 2)
        return x.cpu().numpy()

    cmap = plt.get_cmap("jet_r")
    preds = numpify(torch.cat(preds_all, dim=0))
    gt = numpify(torch.cat(gt, dim=0))
    write_vis_gifs(preds, gt, "eval_preds")

    if FLAGS.depth_objective:
        gt_depths = depth_to_rgb_im(numpify(torch.cat(gt_depths, dim=0)), cmap) / 255.0
        depth_preds = (
            depth_to_rgb_im(numpify(torch.cat(depth_preds_all, dim=0)), cmap) / 255.0
        )
        write_vis_gifs(depth_preds, gt_depths, "eval_depth_preds")


def write_vis_gifs(out_vid, gt, name):
    # write visualization GIF
    n_vis = 32
    # out_vid = np.concatenate(out_vid, axis=0)
    # gt = np.concatenate(gt, axis=0)
    side_by_side_all = np.concatenate((out_vid, gt), axis=2)

    def add_border(video):
        # video dimensions are [B, T, H, W, C]
        border_shape = list(video.shape)
        border_shape[3] = 3
        video = np.concatenate(
            (np.zeros(border_shape), video, np.zeros(border_shape)), axis=3
        )
        border_shape = list(video.shape)
        border_shape[2] = 3
        video = np.concatenate(
            (np.zeros(border_shape), video, np.zeros(border_shape)), axis=2
        )
        return video

    for i in range(256 // n_vis):
        side_by_side = add_border(side_by_side_all[i * n_vis : (i + 1) * n_vis])
        shape = side_by_side.shape
        # [B, T, W, H, C]
        new_shape = (4, n_vis // 4) + shape[1:]
        # [4, B/4, T, W, H, C]
        side_by_side = np.reshape(side_by_side, new_shape)
        side_by_side = np.concatenate(side_by_side, axis=2)
        # [B/4, T, W, H, C]
        side_by_side = np.concatenate(side_by_side, axis=2)
        side_by_side = side_by_side * 255
        from moviepy.editor import ImageSequenceClip

        obs_list = list(np.array(side_by_side))
        fps = 5
        clip = ImageSequenceClip(obs_list, fps=fps)
        log_dir = os.path.join(os.path.dirname(FLAGS.output_dir), "evaluate")
        if not os.path.exists(log_dir):
            os.mkdir(log_dir)
        clip.write_gif(os.path.join(log_dir, f"{name}_{i}.gif"), fps=fps)


def main(argv):
    del argv  # Unused
    log_eval_gifs()


if __name__ == "__main__":
    # We assume that checkpoints are in the output_dir
    flags.mark_flags_as_required(["output_dir"])
    app.run(main)
