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
from copy import deepcopy
from tqdm import tqdm
import moviepy
import wandb

from fitvid.data import robomimic_data
from fitvid.model.fitvid import FitVid


FLAGS = flags.FLAGS

# Model training
flags.DEFINE_boolean('debug', False, 'Debug mode.') # changed
flags.DEFINE_integer('batch_size', 32, 'Batch size.') # changed
flags.DEFINE_integer('n_past', 2, 'Number of past frames.')
flags.DEFINE_integer('n_future', 10, 'Number of future frames.') # not used, inferred directly from data
flags.DEFINE_integer('num_epochs', 1000, 'Number of steps to train for.')
flags.DEFINE_float('beta', 1e-4, 'Weight on KL.')
flags.DEFINE_string('data_in_gpu', 'True', 'whether to put data in GPU, or RAM')
flags.DEFINE_string('loss', 'l2', 'whether to use l2 or l1 loss')
flags.DEFINE_string('lr', '1e-3', 'learning rate')
flags.DEFINE_float('weight_decay', 0.0, 'weight decay value')
flags.DEFINE_boolean('stochastic', True, 'Use a stochastic model.')

# Model architecture
flags.DEFINE_integer('z_dim', 10, 'LSTM output size.') #
flags.DEFINE_integer('rnn_size', 256, 'LSTM hidden size.') #
flags.DEFINE_integer('g_dim', 128, 'CNN feature size / LSTM input size.') #
flags.DEFINE_list('stage_sizes', [1, 1, 1, 1], 'Number of layers for encoder/decoder.') #
flags.DEFINE_list('first_block_shape', [8, 8, 512], 'Decoder first conv size') #
flags.DEFINE_string('action_conditioned', 'True', 'Action conditioning.')
flags.DEFINE_integer('num_base_filters', 32, 'num_filters = num_base_filters * 2^layer_index.') # 64
flags.DEFINE_integer('expand', 1, 'multiplier on decoder\'s num_filters.') # 4
flags.DEFINE_integer('action_size', 4, 'number of actions') # 4
flags.DEFINE_string('skip_type', 'residual', 'skip type: residual, concat, no_skip, pixel_residual') # residual

# Model saving
flags.DEFINE_string('output_dir', None, 'Path to model checkpoints/summaries.')
flags.DEFINE_integer('save_freq', 10, 'number of steps between checkpoints')
flags.DEFINE_boolean('wandb_online', None, 'Use wandb online mode (probably should disable on cluster)')

# Data
flags.DEFINE_spaceseplist('dataset_file', [], 'Dataset to load.')
flags.DEFINE_boolean('hdf5_data', None, 'using hdf5 data')

# post hoc analysis
flags.DEFINE_string('re_eval', 'False', 'Re evaluate all available checkpoints saved.')
flags.DEFINE_integer('re_eval_bs', 20, 'bs override')


def load_data(dataset_files, data_type='train', depth=False):
    video_len = FLAGS.n_past + FLAGS.n_future
    return robomimic_data.load_dataset_robomimic_torch(dataset_files, FLAGS.batch_size, video_len, data_type, depth)


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
    random.seed(0)
    torch.manual_seed(0)
    np.random.seed(0)
    os.environ['PYTHONHASHSEED'] = str(0)

    if FLAGS.loss=='l2':
        loss_fn = F.mse_loss
    elif FLAGS.loss == 'l1':
        loss_fn = F.l1_loss
    else:
        raise NotImplementedError

    model = FitVid(stage_sizes=[int(i) for i in FLAGS.stage_sizes],
                   z_dim=FLAGS.z_dim,
                   g_dim=FLAGS.g_dim,
                   rnn_size=FLAGS.rnn_size,
                   num_base_filters=FLAGS.num_base_filters,
                   first_block_shape=[int(i) for i in FLAGS.first_block_shape],
                   skip_type=FLAGS.skip_type,
                   n_past=FLAGS.n_past,
                   action_conditioned=eval(FLAGS.action_conditioned),
                   action_size=FLAGS.action_size,
                   is_inference=False,
                   expand_decoder=FLAGS.expand,
                   beta=FLAGS.beta,
                   loss_fn=loss_fn,
                   stochastic=FLAGS.stochastic
                   )

    NGPU = torch.cuda.device_count()
    print('CUDA available devices: ', NGPU)

    checkpoint, resume_epoch = get_most_recent_checkpoint(FLAGS.output_dir)
    print(f'Attempting to load from checkpoint {checkpoint if checkpoint else "None, no model found"}')
    if checkpoint:
        model.load_parameters(checkpoint)

    # dump model config
    with open(os.path.join(FLAGS.output_dir, 'config.json'), 'w') as config_file:
        print('Writing config file...')
        json.dump(model.config, config_file)

    model = torch.nn.DataParallel(model)
    model.to('cuda:0')

    optimizer = torch.optim.Adam(model.parameters(), lr=FLAGS.lr, weight_decay=FLAGS.weight_decay)

    if FLAGS.hdf5_data:
        from fitvid.data import load_hdf5_data
        data_loader = load_hdf5_data(FLAGS.dataset_file, FLAGS.batch_size, data_type='train')
        test_data_loader = load_hdf5_data(FLAGS.dataset_file, FLAGS.batch_size, data_type='val')
        prep_data = prep_data_test = lambda x: x
    else:
        data_loader, prep_data = load_data(FLAGS.dataset_file, data_type='train', depth=FLAGS.depth_objective)
        test_data_loader, prep_data_test = load_data(FLAGS.dataset_file, data_type='valid', depth=FLAGS.depth_objective)

    wandb.init(
        project='fitvid-torch',
        reinit=True,
        mode='online' if FLAGS.wandb_online else 'offline'
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

    for epoch in range(resume_epoch+1, num_epochs):

        print(f'\nEpoch {epoch} / {num_epochs}')
        train_save_videos = []
        test_save_videos = []

        print('Evaluating')
        model.eval()
        with torch.no_grad():
            wandb_log = {}
            epoch_mse = []
            epoch_depth_mse = []
            for iter_item in enumerate(tqdm(test_data_loader)):
                test_batch_idx, batch = iter_item
                batch = dict_to_cuda(prep_data_test(batch))
                test_videos = batch['video']

                if NGPU > 1:
                    mse, eval_preds = model.module.evaluate(batch)
                else:
                    mse, eval_preds = model.evaluate(batch)
                if test_batch_idx < num_batch_to_save:
                    save_vids = torch.cat([test_videos[:4, 1:], eval_preds[:4]], dim=-1).detach().cpu().numpy() # save only first 4 of a batch
                    [test_save_videos.append(vid) for vid in save_vids]
                epoch_mse.append(mse.item())
                if FLAGS.debug and test_batch_idx > 5:
                    break
            test_mse.append(np.mean(epoch_mse))
            wandb_log.update({'eval/mse': mse})
            wandb.log(wandb_log)
        print(f'Test MSE: {test_mse[-1]}')

        if test_mse[-1] == np.min(test_mse):
            if not os.path.isdir(FLAGS.output_dir):
                os.mkdir(FLAGS.output_dir)
            save_path = os.path.join(FLAGS.output_dir, f'model_best')
            if NGPU > 1:
                torch.save(model.module.state_dict(), save_path)
            else:
                torch.save(model.state_dict(), save_path)
            print(f'Saved new best model to {save_path}')

        print('Training')
        model.train()
        optimizer.zero_grad()

        for iter_item in enumerate(tqdm(data_loader)):
            wandb_log = {}

            batch_idx, batch = iter_item
            batch = dict_to_cuda(prep_data(batch))
            if predicting_depth:
                inputs = batch['video'], batch['actions'], batch['depth_video']
            else:
                inputs = batch['video'], batch['actions']
            loss, preds, metrics = model(*inputs)

            if NGPU > 1:
                loss = loss.mean()
                metrics = {k: v.mean() for k, v in metrics.items()}
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            train_steps += 1

            train_losses.append(loss.item())
            train_mse.append(metrics['loss/mse'].item())
            videos = batch['video']
            if batch_idx < num_batch_to_save:
                save_vids = torch.cat([videos[:4, 1:], preds[:4]], dim=-1).detach().cpu().numpy()
                [train_save_videos.append(vid) for vid in save_vids]

            wandb_log.update({f'train/{k}': v for k, v in metrics.items()})
            wandb.log(wandb_log)
            if FLAGS.debug and batch_idx > 5:
                break

        if epoch % FLAGS.save_freq == 0:
            if os.path.isdir(FLAGS.output_dir):
                pass
            else:
                os.mkdir(FLAGS.output_dir)
            save_path = os.path.join(FLAGS.output_dir, f'model_epoch{epoch}')
            if NGPU > 1:
                torch.save(model.module.state_dict(), save_path)
            else:
                torch.save(model.state_dict(), save_path)
            print(f'Saved model to {save_path}')
        else:
            print('Skip saving models')

        if epoch % 1 == 0:
            wandb_log = dict()
            for i, vid in enumerate(test_save_videos):
                vid = np.moveaxis(vid, 1, 3)
                vid = np.array(vid * 255)
                wandb_log.update({f'video_test_{i}': wandb.Video(np.moveaxis(vid, 3, 1), fps=4, format='gif')})
                save_moviepy_gif(list(vid), os.path.join(FLAGS.output_dir, f'video_test_epoch{epoch}_{i}'), fps=5)
                print('Saved test vid for the epoch')

            for i, vid in enumerate(train_save_videos):
                vid = np.moveaxis(vid, 1, 3)
                vid = np.array(vid * 255)
                wandb_log.update({f'video_train_{i}': wandb.Video(np.moveaxis(vid, 3, 1), fps=4, format='gif')})
                save_moviepy_gif(list(vid), os.path.join(FLAGS.output_dir, f'video_train_epoch{epoch}_{i}'), fps=5)
                print('Saved train vid for each epoch')

            wandb.log(wandb_log)


if __name__ == "__main__":
    flags.mark_flags_as_required(['output_dir'])
    app.run(main)
