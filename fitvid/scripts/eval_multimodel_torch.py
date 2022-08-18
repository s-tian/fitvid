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

""" Usage example: 
python eval_torch.py --eval_folders torch-outputs/5Hz-agentview-shift2-expert-bl/model_epoch40 --dataset_file  "/viscam/u/stian/perceptual-metrics/robosuite/robosuite/models/assets/policy_rollouts/pushcenter_osc_position_eval/image_and_depth.hdf5" --rnn_size 128 --g_dim 64 --batch_size 24  --save_freq 10 --camera_view agentview_shift_2 
"""

import os
import time

from absl import app
from absl import flags
from fitvid.scripts.train_fitvid import load_data, dict_to_cuda, depth_to_rgb_im
from fitvid.model.fitvid import FitVid
from fitvid.utils import generate_text_square, save_moviepy_gif
import torch
import numpy as np
import matplotlib.pyplot as plt

FLAGS = flags.FLAGS
flags.DEFINE_spaceseplist("eval_folders", [], "Dataset to load.")
flags.DEFINE_spaceseplist("friendly_model_names", [], "Friendly model names")
flags.DEFINE_boolean("valid", True, "use validation data")


def get_data_batches(n):
    data_itr, prep_data = load_data(
        FLAGS.dataset_file, "valid" if FLAGS.valid else "train", depth=True
    )
    data_itr = iter(data_itr)
    batches = []
    for i in range(5):
        _ = next(data_itr)
    for i in range(n):
        batch = prep_data(next(data_itr))
        batches.append(batch)
    return batches


def get_model_predictions(model, ckp, batches):
    model.load_parameters(ckp)
    preds_all, depth_preds_all = [], []
    gt, gt_depths = [], []
    for batch in batches:
        with torch.no_grad():
            _, preds = model.evaluate(dict_to_cuda(batch))
        if FLAGS.depth_objective:
            preds, depth_preds = preds
            depth_preds_all.append(depth_preds)
            gt_depths.append(batch["depth_video"][:, 1:])
        preds_all.append(preds * 255)
        gt.append(batch["video"][:, 1:] * 255)
    return (
        torch.cat(gt),
        torch.cat(gt_depths),
        torch.cat(preds_all),
        torch.cat(depth_preds_all),
    )


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
        tv_weight=FLAGS.tv_weight,
        depth_weight=FLAGS.depth_weight,
        pretrained_depth_path=FLAGS.depth_model_path,
        depth_model_size=FLAGS.depth_model_size,
        freeze_depth_model=FLAGS.freeze_pretrained,
        segmentation_loss_weight=0,
        segmentation_depth_loss_weight=FLAGS.segmentation_depth_loss_weight,
        policy_feature_path=FLAGS.policy_feature_path,
        policy_feature_weight=FLAGS.policy_feature_weight,
    )
    NGPU = torch.cuda.device_count()
    print("CUDA available devices: ", NGPU)

    # checkpoint = FLAGS.eval_folder
    # print(f'Attempting to load from checkpoint {checkpoint if checkpoint else "None, no model found"}')
    # if checkpoint:
    #     model.load_parameters(checkpoint)
    model.eval()
    model.cuda()

    batches = get_data_batches(3)

    eval_outputs = []
    for ckpt in FLAGS.eval_folders:
        eval_outputs.append(get_model_predictions(model, ckpt, batches))

    def numpify(x):
        return x.detach().cpu().numpy()

    cmap = plt.get_cmap("jet_r")
    gt, depth_gt, _, _ = eval_outputs[0]
    tlen = gt.shape[1]
    header_items = (
        ["GT"] + FLAGS.friendly_model_names + ["GT Depth"] + FLAGS.friendly_model_names
    )
    # generate header
    text_headers = [generate_text_square(h) for h in header_items]
    text_headers = np.concatenate(text_headers, axis=-1)
    # duplicate text headers through time
    text_headers = np.tile(text_headers[None], (tlen, 1, 1, 1))

    image_row = [numpify(gt)]

    for eval_outs in eval_outputs:
        _, _, rgb_pred, _ = eval_outs
        image_row.append(numpify(rgb_pred))
    image_row.append(np.moveaxis(depth_to_rgb_im(numpify(depth_gt), cmap), 4, 2))
    for eval_outs in eval_outputs:
        _, _, _, depth_pred = eval_outs
        image_row.append(np.moveaxis(depth_to_rgb_im(numpify(depth_pred), cmap), 4, 2))
    image_row = np.concatenate(image_row, axis=-1)
    image_row = np.concatenate(image_row, axis=-2)
    image_row = np.concatenate((text_headers, image_row), axis=-2)
    image_row = np.moveaxis(image_row, 1, 3)
    tag = "valid" if FLAGS.valid else "train"
    save_moviepy_gif(list(image_row), f"multimodel_sobel_eval_{tag}")


def main(argv):
    del argv  # Unused
    log_eval_gifs()


if __name__ == "__main__":
    # We assume that checkpoints are in the output_dir
    flags.mark_flags_as_required(["output_dir"])
    app.run(main)
