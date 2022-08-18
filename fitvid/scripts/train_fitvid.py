import torch
from torch import nn as nn
import torch.nn.functional as F
from absl import flags
from absl import app
import numpy as np
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
import os
import glob
import json
from copy import deepcopy
from tqdm import tqdm
import moviepy
import wandb

from fitvid.data.robomimic_data import load_dataset_robomimic_torch
from fitvid.data.hdf5_data_loader import load_hdf5_data
from fitvid.model.fitvid import FitVid
from fitvid.utils.utils import dict_to_cuda, count_parameters
from fitvid.utils.vis_utils import save_moviepy_gif, build_visualization

FLAGS = flags.FLAGS

# Model training
flags.DEFINE_integer('seed', 0, 'Random seed.')  # changed
flags.DEFINE_boolean('debug', False, 'Debug mode.')  # changed
flags.DEFINE_integer('batch_size', 32, 'Batch size.')  # changed
flags.DEFINE_integer('n_past', 2, 'Number of past frames.')
flags.DEFINE_integer('n_future', 10, 'Number of future frames.')  # not used, inferred directly from data
flags.DEFINE_integer('num_epochs', 1000, 'Number of steps to train for.')
flags.DEFINE_integer('train_epoch_max_length', 45000, 'Max number of training batches per epoch. Actual epoch length will be min(dataset length, this value).')
flags.DEFINE_float('beta', 1e-4, 'Weight on KL.')
flags.DEFINE_string('data_in_gpu', 'True', 'whether to put data in GPU, or RAM')
flags.DEFINE_string('rgb_loss_type', 'l2', 'whether to use l2 or l1 loss')
flags.DEFINE_float('lr', 1e-3, 'learning rate')
flags.DEFINE_boolean('adamw_1cycle', False, 'use AdamW and 1cycle learning rate scheduler')
flags.DEFINE_boolean('lecun_initialization', False, 'use LeCun weight initialization as in original implementation')
flags.DEFINE_float('weight_decay', 0.0, 'weight decay value')
flags.DEFINE_float('adam_eps', 1e-8, 'epsilon parameter for Adam optimizer')
flags.DEFINE_boolean('stochastic', True, 'Use a stochastic model.')
flags.DEFINE_boolean('multistep', False, 'Multi-step training.')  # changed
flags.DEFINE_boolean('fp16', False, 'Use lower precision training for perf improvement.')  # changed
flags.DEFINE_float('rgb_weight', 1, 'Weight on rgb objective (default 1).')
flags.DEFINE_boolean('fitvid_augment', False, 'Use fitvid-style data augmentation.')  # changed

# Model architecture
flags.DEFINE_integer('z_dim', 10, 'LSTM output size.')  #
flags.DEFINE_integer('rnn_size', 256, 'LSTM hidden size.')  #
flags.DEFINE_integer('g_dim', 128, 'CNN feature size / LSTM input size.')  #
flags.DEFINE_list('stage_sizes', [1, 1, 1, 1], 'Number of layers for encoder/decoder.')  #
flags.DEFINE_list('first_block_shape', [8, 8, 512], 'Decoder first conv size')  #
flags.DEFINE_string('action_conditioned', 'True', 'Action conditioning.')
flags.DEFINE_integer('num_base_filters', 32, 'num_filters = num_base_filters * 2^layer_index.')  # 64
flags.DEFINE_integer('expand', 1, 'multiplier on decoder\'s num_filters.')  # 4
flags.DEFINE_integer('action_size', 4, 'number of actions')  # 4
flags.DEFINE_string('skip_type', 'residual', 'skip type: residual, concat, no_skip, pixel_residual')  # residual

# Model saving
flags.DEFINE_string('output_dir', None, 'Path to model checkpoints/summaries.')
flags.DEFINE_integer('save_freq', 10, 'number of steps between checkpoints')
flags.DEFINE_boolean('wandb_online', None, 'Use wandb online mode (probably should disable on cluster)')
flags.DEFINE_string('project', 'perceptual-metrics', 'wandb project name')

# Data
flags.DEFINE_spaceseplist('dataset_file', [], 'Dataset to load.')
flags.DEFINE_boolean('hdf5_data', None, 'using hdf5 data')
flags.DEFINE_string('cache_mode', 'low_dim', 'Dataset cache mode')
flags.DEFINE_string('camera_view', 'agentview', 'Camera view of data to load. Default is "agentview".')
flags.DEFINE_list('image_size', [], 'H, W of images')
flags.DEFINE_boolean('has_segmentation', True, 'Does dataset have segmentation masks')

# depth objective
flags.DEFINE_boolean('only_depth', False, 'Depth to depth prediction model.')
flags.DEFINE_float('depth_weight', 0, 'Weight on depth objective.')
flags.DEFINE_string('depth_model_path', None, 'Path to load pretrained depth model from.')
flags.DEFINE_integer('depth_model_size', 256, 'Depth model size.')
flags.DEFINE_integer('depth_start_epoch', 2, 'When to start training with depth objective.')

# normal model
flags.DEFINE_float('normal_weight', 0, 'Weight on normal objective.')
flags.DEFINE_string('normal_model_path', None, 'Path to load pretrained depth model from.')

# define weights for other types of losses
flags.DEFINE_float('lpips_weight', 0, 'Weight on lpips objective.')
flags.DEFINE_float('tv_weight', 0, 'Weight on tv objective.')
flags.DEFINE_float('segmented_object_weight', 0, 'Extra weighting on pixels with segmented objects.')
flags.DEFINE_boolean('corr_wise', False, 'Correlation-wise loss.')

# Policy networks
flags.DEFINE_float('policy_weight', 0, 'Weight on policy loss.')
flags.DEFINE_spaceseplist('policy_network_paths', [], 'Policy feature network locations.')
flags.DEFINE_string('policy_network_layer', 'fc0', 'Layer to use for policy feature network.')

# post hoc analysis
flags.DEFINE_string('re_eval', 'False', 'Re evaluate all available checkpoints saved.')
flags.DEFINE_integer('re_eval_bs', 20, 'bs override')


def load_data(dataset_files, data_type='train', depth=False, normal=False, seg=True, augmentation=None):
    video_len = FLAGS.n_past + FLAGS.n_future
    if FLAGS.image_size:
        assert len(FLAGS.image_size) == 2, "Image size should be (H, W)"
        image_size = [int(i) for i in FLAGS.image_size]
    else:
        image_size = None
    return load_dataset_robomimic_torch(dataset_files, FLAGS.batch_size, video_len, image_size,
                                        data_type, depth, normal, view=FLAGS.camera_view, cache_mode=FLAGS.cache_mode,
                                        seg=seg, only_depth=FLAGS.only_depth, augmentation=augmentation)


def get_most_recent_checkpoint(dir):
    if os.path.isdir(dir):
        pass
    else:
        os.mkdir(dir)
    existing_checkpoints = glob.glob(os.path.join(dir, 'model_epoch*'))
    if existing_checkpoints:
        checkpoint_nums = [int(s.split('/')[-1][len('model_epoch'):]) for s in existing_checkpoints]
        best_ind = checkpoint_nums.index(max(checkpoint_nums))
        return existing_checkpoints[best_ind], max(checkpoint_nums)
    else:
        return None, -1


def main(argv):
    import random

    if FLAGS.seed: # If seed is not set, use a random one
        random.seed(FLAGS.seed)
        torch.manual_seed(FLAGS.seed)
        np.random.seed(FLAGS.seed)
        os.environ['PYTHONHASHSEED'] = str(FLAGS.seed)

    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.enabled = True

    if FLAGS.depth_model_path:
        depth_predictor_kwargs = {
            'depth_model_type': 'mns',
            'pretrained_weight_path': os.path.abspath(FLAGS.depth_model_path),
            'input_size': FLAGS.depth_model_size,
        }
    else:
        depth_predictor_kwargs = None

    if FLAGS.normal_model_path:
        normal_predictor_kwargs = {
            'pretrained_weight_path': os.path.abspath(FLAGS.normal_model_path),
        }
    else:
        normal_predictor_kwargs = None

    if FLAGS.policy_network_paths:
        policy_network_kwargs = {
            'pretrained_weight_paths': [os.path.abspath(p) for p in FLAGS.policy_network_paths],
            'layer': FLAGS.policy_network_layer
        }
    else:
        policy_network_kwargs = None

    total_weights = FLAGS.rgb_weight + FLAGS.depth_weight + FLAGS.normal_weight
    if total_weights > 0:
        rgb_weight = FLAGS.rgb_weight / total_weights
        depth_weight = FLAGS.depth_weight / total_weights
        normal_weight = FLAGS.normal_weight / total_weights
    else:
        rgb_weight = 0
        depth_weight = 0
        normal_weight = 0

    loss_weights = {
        'kld': FLAGS.beta,
        'rgb': rgb_weight,
        'depth': depth_weight,
        'normal': normal_weight,
        'policy': FLAGS.policy_weight,
        'lpips': FLAGS.lpips_weight,
        'tv': FLAGS.tv_weight,
        'segmented_object': FLAGS.segmented_object_weight,
    }

    model_kwargs = dict(
        stage_sizes=[int(i) for i in FLAGS.stage_sizes],
        z_dim=FLAGS.z_dim,
        g_dim=FLAGS.g_dim,
        rnn_size=FLAGS.rnn_size,
        num_base_filters=FLAGS.num_base_filters,
        first_block_shape=[int(i) for i in FLAGS.first_block_shape],
        skip_type=FLAGS.skip_type,
        n_past=FLAGS.n_past,
        action_conditioned=eval(FLAGS.action_conditioned),
        action_size=FLAGS.action_size,
        expand_decoder=FLAGS.expand,
        stochastic=FLAGS.stochastic,
        video_channels=1 if FLAGS.only_depth else 3,
        lecun_initialization=FLAGS.lecun_initialization,
    )

    model = FitVid(model_kwargs=model_kwargs,
                   beta=FLAGS.beta,
                   multistep=FLAGS.multistep,
                   is_inference=False,
                   depth_predictor=depth_predictor_kwargs,
                   normal_predictor=normal_predictor_kwargs,
                   policy_networks=policy_network_kwargs,
                   loss_weights=loss_weights,
                   rgb_loss_type=FLAGS.rgb_loss_type,
                   corr_wise=FLAGS.corr_wise,
                   )
    print(f"Built FitVid model with {count_parameters(model)} parameters!")

    NGPU = torch.cuda.device_count()
    print('CUDA available devices: ', NGPU)

    checkpoint, resume_epoch = get_most_recent_checkpoint(FLAGS.output_dir)
    print(f'Attempting to load from checkpoint {checkpoint if checkpoint else "None, no model found"}')
    if checkpoint:
        model.load_parameters(checkpoint)

    model.setup_train_losses()

    # dump model config
    with open(os.path.join(FLAGS.output_dir, 'config.json'), 'w') as config_file:
        print('Writing config file...')
        json.dump(model.config, config_file)

    model = torch.nn.DataParallel(model)
    model.to('cuda:0')

    image_size = [int(i) for i in FLAGS.image_size]

    from fitvid.model.augmentations import FitVidAugment
    if FLAGS.fitvid_augment:
        image_augmentation = FitVidAugment(image_size)
    else:
        image_augmentation = None

    if FLAGS.hdf5_data:
        data_loader = load_hdf5_data(FLAGS.dataset_file, FLAGS.batch_size, sel_len=FLAGS.n_past + FLAGS.n_future,
                                     image_size=image_size, data_type='train')
        test_data_loader = load_hdf5_data(FLAGS.dataset_file, FLAGS.batch_size, sel_len=FLAGS.n_past + FLAGS.n_future,
                                          image_size=image_size, data_type='val')
        prep_data = prep_data_test = lambda x: x
    else:
        if FLAGS.debug:
            data_loader, prep_data = load_data(FLAGS.dataset_file, data_type='valid',
                                               depth=FLAGS.depth_model_path, normal=FLAGS.normal_model_path,
                                               seg=FLAGS.has_segmentation, augmentation=image_augmentation)
        else:
            data_loader, prep_data = load_data(FLAGS.dataset_file, data_type='train',
                                               depth=FLAGS.depth_model_path, normal=FLAGS.normal_model_path,
                                               seg=FLAGS.has_segmentation, augmentation=image_augmentation)
        test_data_loader, prep_data_test = load_data(FLAGS.dataset_file, data_type='valid',
                                                     depth=FLAGS.depth_model_path,
                                                     normal=FLAGS.normal_model_path,
                                                     seg=FLAGS.has_segmentation)

    if FLAGS.adamw_1cycle:
        optimizer = torch.optim.AdamW(model.parameters(), lr=FLAGS.lr, eps=FLAGS.adam_eps, weight_decay=FLAGS.weight_decay)
        scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=1e-3, epochs=FLAGS.num_epochs, steps_per_epoch=min(FLAGS.train_epoch_max_length, len(data_loader)))
    else:
        optimizer = torch.optim.Adam(model.parameters(), lr=FLAGS.lr, eps=FLAGS.adam_eps, weight_decay=FLAGS.weight_decay)

    wandb.init(
        project=FLAGS.project,
        reinit=True,
        mode='online' if FLAGS.wandb_online else 'offline',
        settings=wandb.Settings(start_method='fork'),
    )

    if FLAGS.output_dir is not None:
        wandb.run.name = f"{FLAGS.output_dir.split('/')[-1]}"
        wandb.run.save()

    wandb.config.update(FLAGS)
    wandb.config.slurm_job_id = os.getenv('SLURM_JOB_ID', 0)

    train_losses = []
    train_mse = []
    test_mse = []
    num_epochs = FLAGS.num_epochs
    train_steps = 0
    num_batch_to_save = 1

    from torch.cuda.amp import autocast, GradScaler
    from contextlib import ExitStack
    scaler = GradScaler()

    for epoch in range(resume_epoch + 1, num_epochs):
        print(f'\nEpoch {epoch} / {num_epochs}')
        print('Evaluating')
        model.eval()
        with torch.no_grad():
            epoch_mse = []
            eval_metrics = dict()
            wandb_log = dict()
            total_test_batches = len(test_data_loader)
            for iter_item in enumerate(tqdm(test_data_loader)):
                test_batch_idx, batch = iter_item
                batch = dict_to_cuda(prep_data_test(batch))
                with autocast() if FLAGS.fp16 else ExitStack() as ac:
                    metrics, eval_preds = model.module.evaluate(batch, compute_metrics=test_batch_idx % 1 == 0)
                    if test_batch_idx < num_batch_to_save:
                        for ag_type, eval_pred in eval_preds.items():
                            if depth_predictor_kwargs:
                                with torch.no_grad():
                                    gt_depth_preds = model.module.depth_predictor(batch['video'][:, 1:])
                                test_videos_log = {
                                    'gt': batch['video'][:, 1:],
                                    'pred': eval_pred['rgb'],
                                    'gt_depth': batch['depth_video'][:, 1:],
                                    'gt_depth_pred': gt_depth_preds,
                                    'pred_depth': eval_pred['depth'],
                                    'rgb_loss': metrics[f'{ag_type}/loss/mse_per_sample'],
                                    'depth_loss_weight': model.module.loss_weights['depth'],
                                    'depth_loss': metrics[f'{ag_type}/loss/depth_loss_per_sample'],
                                    'name': 'Depth',
                                }
                                test_depth_visualization = build_visualization(**test_videos_log)
                                wandb_log.update(
                                    {f'{ag_type}/test_depth_vis': wandb.Video(test_depth_visualization, fps=4, format='gif')})
                            else:  # only rgb or only depth case
                                test_videos_log = {
                                    'gt': batch['video'][:, 1:],
                                    'pred': eval_pred['rgb'],
                                    'rgb_loss': metrics[f'{ag_type}/loss/mse_per_sample'],
                                }
                                test_visualization = build_visualization(**test_videos_log)
                                wandb_log.update({f'{ag_type}/test_vis': wandb.Video(test_visualization, fps=4, format='gif')})
                            if normal_predictor_kwargs:
                                with torch.no_grad():
                                    gt_normal_preds = model.module.normal_predictor(batch['video'][:, 1:])
                                test_videos_log = {
                                    'gt': batch['video'][:, 1:],
                                    'pred': eval_pred['rgb'],
                                    'gt_depth': batch['normal'][:, 1:],
                                    'gt_depth_pred': gt_normal_preds,
                                    'pred_depth': eval_pred['normal'],
                                    'rgb_loss': metrics['loss/mse_per_sample'],
                                    'depth_loss_weight': model.module.loss_weights['normal'],
                                    'depth_loss': metrics['loss/normal_loss_per_sample'],
                                    'name': 'Normal',
                                }
                                test_normal_visualization = build_visualization(**test_videos_log)
                                wandb_log.update({f'{ag_type}/test_normal_vis': wandb.Video(test_normal_visualization, fps=4, format='gif')})
                epoch_mse.append(metrics['ag/loss/mse'].item())
                for k, v in metrics.items():
                    if k in eval_metrics:
                        eval_metrics[k].append(v)
                    else:
                        eval_metrics[k] = [v]
                if (FLAGS.debug and test_batch_idx > 25) or test_batch_idx == 50 or test_batch_idx == total_test_batches - 1:
                    wandb_log.update({f'eval/{k}': torch.stack(v).mean() for k, v in eval_metrics.items()})
                    wandb.log(wandb_log)
                    break
            test_mse.append(np.mean(epoch_mse))
        print(f'Test MSE: {np.mean(epoch_mse)}')

        if test_mse[-1] == np.min(test_mse):
            if not os.path.isdir(FLAGS.output_dir):
                os.mkdir(FLAGS.output_dir)
            save_path = os.path.join(FLAGS.output_dir, f'model_best')
            torch.save(model.module.state_dict(), save_path)
            print(f'Saved new best model to {save_path}')

        print('Training')
        model.train()

        if epoch < FLAGS.depth_start_epoch:
            model.module.loss_weights['depth'] = 0
            model.module.loss_weights['normal'] = 0
        else:
            model.module.loss_weights['depth'] = depth_weight
            model.module.loss_weights['normal'] = normal_weight

        for iter_item in enumerate(tqdm(data_loader)):
            wandb_log = dict()

            batch_idx, batch = iter_item
            batch = dict_to_cuda(prep_data(batch))

            # get segmentations from data if they exist
            batch_segmentations = batch.get('segmentation', None)
            inputs = batch['video'], batch['actions'], batch_segmentations

            if depth_predictor_kwargs:
                inputs = inputs + (batch['depth_video'],)
            else:
                inputs = inputs + (None, )

            if normal_predictor_kwargs:
                inputs = inputs + (batch['normal'],)
            else:
                inputs = inputs + (None,)

            optimizer.zero_grad()
            with autocast() if FLAGS.fp16 else ExitStack() as ac:
                loss, preds, metrics = model(*inputs, compute_metrics=(batch_idx % 200 == 0))

            if NGPU > 1:
                loss = loss.mean()
                for k, v in metrics.items():
                    if len(metrics[k].shape) == 1:
                        if metrics[k].shape[0] == NGPU:
                            metrics[k] = v.mean()

            if FLAGS.fp16:
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 100)
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 100)
                optimizer.step()

            if FLAGS.adamw_1cycle:
                scheduler.step()

            train_steps += 1
            train_losses.append(loss.item())
            train_mse.append(metrics['loss/mse'].item())
            if batch_idx < num_batch_to_save:
                if depth_predictor_kwargs:
                    with torch.no_grad():
                        gt_depth_preds = model.module.depth_predictor(batch['video'][:, 1:])
                if depth_predictor_kwargs:
                    train_videos_log = {
                        'gt': batch['video'][:, 1:],
                        'pred': preds['rgb'],
                        'gt_depth': batch['depth_video'][:, 1:],
                        'gt_depth_pred': gt_depth_preds,
                        'pred_depth': preds['depth'],
                        'rgb_loss': metrics['loss/mse_per_sample'],
                        'depth_loss_weight': model.module.loss_weights['depth'],
                        'depth_loss': metrics['loss/depth_loss_per_sample'],
                    }
                    train_depth_visualization = build_visualization(**train_videos_log)
                    wandb_log.update({'train_depth_vis': wandb.Video(train_depth_visualization, fps=4, format='gif')})
                else:
                    train_videos_log = {
                        'gt': batch['video'][:, 1:],
                        'pred': preds['rgb'],
                        'rgb_loss': metrics['loss/mse_per_sample'],
                    }
                    train_visualization = build_visualization(**train_videos_log)
                    wandb_log.update({'train_vis': wandb.Video(train_visualization, fps=4, format='gif')})
                if normal_predictor_kwargs:
                    with torch.no_grad():
                        gt_normal_preds = model.module.normal_predictor(batch['video'][:, 1:])
                    train_videos_log = {
                        'gt': batch['video'][:, 1:],
                        'pred': preds['rgb'],
                        'gt_depth': batch['normal'][:, 1:],
                        'gt_depth_pred': gt_normal_preds,
                        'pred_depth': preds['normal'],
                        'rgb_loss': metrics['loss/mse_per_sample'],
                        'depth_loss_weight': model.module.loss_weights['normal'],
                        'depth_loss': metrics['loss/normal_loss_per_sample'],
                        'name': 'Normal',
                    }
                    train_normal_visualization = build_visualization(**train_videos_log)
                    wandb_log.update({'train_normal_vis': wandb.Video(train_normal_visualization, fps=4, format='gif')})

            wandb_log.update({f'train/{k}': v for k, v in metrics.items()})
            wandb.log(wandb_log)
            if (FLAGS.debug and batch_idx > 25) or batch_idx > FLAGS.train_epoch_max_length:
                break

        if epoch % FLAGS.save_freq == 0:
            if os.path.isdir(FLAGS.output_dir):
                pass
            else:
                os.mkdir(FLAGS.output_dir)
            save_path = os.path.join(FLAGS.output_dir, f'model_epoch{epoch}')
            torch.save(model.module.state_dict(), save_path)
            print(f'Saved model to {save_path}')
        else:
            print('Skip saving models')


if __name__ == "__main__":
    flags.mark_flags_as_required(['output_dir'])
    app.run(main)
