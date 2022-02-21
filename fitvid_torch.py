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

from fitvid import robomimic_data

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

# depth objective
flags.DEFINE_boolean('depth_objective', False, 'Use depth image decoding as a auxiliary objective.')
flags.DEFINE_float('depth_weight', 100, 'Weight on depth objective.')
flags.DEFINE_string('depth_loss', 'norm_mse', 'Depth objective loss type. Choose from "norm_mse" and "eigen".')
flags.DEFINE_float('depth_start_epoch', 80, 'Weight on depth objective.')
flags.DEFINE_boolean('pretrained_depth_objective', True, 'Instead of using a learned depth model, use a pretrained one.')
flags.DEFINE_boolean('freeze_pretrained', False, 'Whether to freeze the weights of the pretrained depth model.')

# post hoc analysis
flags.DEFINE_string('re_eval', 'False', 'Re evaluate all available checkpoints saved.')
flags.DEFINE_integer('re_eval_bs', 20, 'bs override')


class MultiGaussianLSTM(nn.Module):
    """Multi layer lstm with Gaussian output."""

    def __init__(self, input_size, output_size, hidden_size, num_layers):
        super(MultiGaussianLSTM, self).__init__()

        # assert num_layers == 1
        self.embed = nn.Linear(input_size, hidden_size)
        self.mean = nn.Linear(hidden_size, output_size)
        self.logvar = nn.Linear(hidden_size, output_size)
        self.layers_0 = nn.LSTM(input_size=hidden_size, hidden_size=hidden_size, num_layers=num_layers)
        # init hidden state is init to zero

    def forward(self, x, states):
        # assume x to only contain one timestep i.e. (bs, feature_dim)
        x = self.embed(x)
        x = x.view((1,) + x.shape)
        x, new_states = self.layers_0(x, states)
        mean = self.mean(x)[0]
        logvar = self.logvar(x)[0]

        epsilon = torch.normal(mean=0, std=1, size=mean.shape).cuda()
        var = torch.exp(0.5 * logvar)
        z_t = mean + var * epsilon
        return (z_t, mean, logvar), new_states


class SEBlock(nn.Module):
    """Applies Squeeze-and-Excitation."""
    def __init__(self, channel):
        super(SEBlock, self).__init__()
        hidden_size = max(channel // 16, 4)

        self.reduce = nn.Linear(in_features=channel, out_features=hidden_size)
        self.act = nn.ReLU()
        self.expand = nn.Linear(in_features=hidden_size, out_features=channel)

    def forward(self, x): # (10, 3, 64, 64)
        y = x.mean(axis=(2, 3)) # (10, 3)
        y = self.reduce(y) # (10, 4)
        y = self.act(y)
        y = self.expand(y)
        y = y.view(y.shape + (1, 1))
        return torch.sigmoid(y) * x


class EncoderBlock(nn.Module):
    """NVAE ResNet block."""

    def __init__(self, in_channels, out_channels, downsample):
        super(EncoderBlock, self).__init__()

        stride = (2, 2) if downsample else (1, 1)

        self.Conv_0 = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=(3, 3), stride=stride, bias=False, padding=1) # pad = (kernel-1)//2
        self.Conv_1 = nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=(3, 3), bias=False, padding=1)
        self.BatchNorm_0 = nn.BatchNorm2d(num_features=in_channels, eps=1e-5, momentum=0.9, affine=True, track_running_stats=True)
        self.BatchNorm_1 = nn.BatchNorm2d(num_features=out_channels, eps=1e-5, momentum=0.9, affine=True, track_running_stats=True)
        self.act1 = nn.SiLU()
        self.act2 = nn.SiLU()
        self.act3 = nn.SiLU()
        self.SEBlock_0 = SEBlock(out_channels)

        if (in_channels != out_channels) or downsample:
            self.reshape_residual = True
            self.conv_proj = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=(1, 1), stride=stride, bias=False, padding=0)
            self.norm_proj = nn.BatchNorm2d(num_features=out_channels, eps=1e-5, momentum=0.9, affine=True, track_running_stats=True)
        else:
            self.reshape_residual = False
        self.act4 = nn.SiLU()

    def forward(self, x):
        residual = x
        y = self.BatchNorm_0(x)
        y = self.act1(y)
        y = self.Conv_0(y)
        y = self.BatchNorm_1(y)
        y = self.act2(y)
        y = self.Conv_1(y)
        y = self.SEBlock_0(y)

        if self.reshape_residual:
            residual = self.conv_proj(residual)
            residual = self.norm_proj(residual)

        return self.act4(residual + y)


class DecoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels, expand, upsample):
        super(DecoderBlock, self).__init__()
        self.upsample = upsample
        if self.upsample:
            self.upsample_layer = nn.Upsample(scale_factor=2)
        self.BatchNorm_0 = nn.BatchNorm2d(num_features=in_channels, eps=1e-5, momentum=0.9, affine=True, track_running_stats=True)
        self.Conv_0 = nn.Conv2d(in_channels=in_channels, out_channels=out_channels*expand, kernel_size=(1, 1), bias=False, padding=0) # pad = (kernel-1)//2
        self.BatchNorm_1 = nn.BatchNorm2d(num_features=out_channels*expand, eps=1e-5, momentum=0.9, affine=True, track_running_stats=True)
        self.act1 = nn.SiLU()
        self.Conv_1 = nn.Conv2d(in_channels=out_channels*expand, out_channels=out_channels*expand, kernel_size=(5, 5), bias=False, padding=2) # pad = (kernel-1)//2
        self.BatchNorm_2 = nn.BatchNorm2d(num_features=out_channels*expand, eps=1e-5, momentum=0.9, affine=True, track_running_stats=True)
        self.act2 = nn.SiLU()
        self.Conv_2 = nn.Conv2d(in_channels=out_channels*expand, out_channels=out_channels, kernel_size=(1, 1), bias=False, padding=0) # pad = (kernel-1)//2
        self.BatchNorm_3 = nn.BatchNorm2d(num_features=out_channels, eps=1e-5, momentum=0.9, affine=True, track_running_stats=True)
        param = self.BatchNorm_3.state_dict()['weight']
        param.copy_(torch.zeros_like(param))
        # for the 3 lines above, original code is: y = self.norm(scale_init=nn.initializers.zeros)(y)
        self.SEBlock_0 = SEBlock(out_channels)

        if in_channels != out_channels:
            self.reshape_residual = True
            self.conv_proj = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=(1, 1), bias=False, padding=0)
            self.norm_proj = nn.BatchNorm2d(num_features=out_channels, eps=1e-5, momentum=0.9, affine=True, track_running_stats=True)
        else:
            self.reshape_residual = False
        self.act3 = nn.SiLU()

    def forward(self, x):
        if self.upsample:
            x = self.upsample_layer(x)

        residual = x
        y = self.BatchNorm_0(x)
        y = self.Conv_0(y)
        y = self.BatchNorm_1(y)
        y = self.act1(y)
        y = self.Conv_1(y)
        y = self.BatchNorm_2(y)
        y = self.act2(y)
        y = self.Conv_2(y)
        y = self.BatchNorm_3(y)
        y = self.SEBlock_0(y)

        if self.reshape_residual:
            residual = self.conv_proj(residual)
            residual = self.norm_proj(residual)

        return self.act3(residual + y)


class ModularEncoder(nn.Module):
    """Modular Encoder."""
    def __init__(self, stage_sizes, output_size, num_base_filters):
        super(ModularEncoder, self).__init__()
        self.stages = nn.ModuleList()
        prev_num_filters = 3
        num_filters = None
        count = 0
        for i, block_size in enumerate(stage_sizes):
            blocks = nn.ModuleList()
            for j in range(block_size):
                num_filters = num_base_filters * 2 ** i
                downsample = True if i > 0 and j == 0 else False
                block = EncoderBlock(in_channels=prev_num_filters, out_channels=num_filters, downsample=downsample)
                setattr(self, f'EncoderBlock_{count}', block)
                count += 1
                prev_num_filters = num_filters
                blocks.append(block)
            self.stages.append(blocks)

        self.Dense_0 = nn.Linear(in_features=num_filters, out_features=output_size)

    def forward(self, x):
        skips = {}
        for i, stage in enumerate(self.stages):
            for j, block in enumerate(stage):
                x = block(x)
                skips[(i, j)] = x
        x = x.mean(axis=(2, 3))
        x = self.Dense_0(x)

        return x, skips


class ModularDecoder(nn.Module):
    def __init__(self, first_block_shape, input_size, stage_sizes, num_base_filters, skip_type, expand):
        super(ModularDecoder, self).__init__()
        self.skip_type = skip_type
        self.stage_sizes = stage_sizes
        self.first_block_shape = first_block_shape

        first_block_size = np.prod(np.array(first_block_shape))
        self.Dense_0 = nn.Linear(in_features=input_size, out_features=first_block_size)

        self.stages = nn.ModuleList()
        prev_num_filters = self.first_block_shape[-3]
        count = 0
        for i, block_size in enumerate(reversed(stage_sizes)):
            blocks = nn.ModuleList()
            for j in range(block_size):
                num_filters = num_base_filters * 2 ** (len(stage_sizes) - i - 1)
                upsample = True if i > 0 and j == 0 else False
                block = DecoderBlock(in_channels=prev_num_filters, out_channels=num_filters, expand=expand, upsample=upsample)
                setattr(self, f'DecoderBlock_{count}', block)
                blocks.append(block)
                count += 1
                prev_num_filters = num_filters
            self.stages.append(blocks)

        self.Conv_0 = nn.Conv2d(in_channels=prev_num_filters, out_channels=3, kernel_size=(3, 3), bias=False, padding=1) # pad = (kernel-1)//2

    def forward(self, x, skips, has_time_dim=True):
        if has_time_dim:
            shape = x.shape
            x = x.view((shape[0]*shape[1],) + shape[2:]) # collapse batch*time dims [b0t0, b0t1, b0t2... b1t0, b1t1, b1t2...]
        x = self.Dense_0(x)
        x = x.view((x.shape[0],) + tuple(self.first_block_shape))

        for i, stage in enumerate(self.stages):
            for j, block in enumerate(stage):
                x = block(x)
                if self.skip_type == 'residual':
                    skip = skips[(len(self.stage_sizes) - i - 1, len(stage) - j - 1)]
                    repeat = int(x.shape[0] / skip.shape[0])
                    skip = torch.repeat_interleave(skip, repeat, dim=0)
                    x = x + skip
                elif self.skip_type == 'concat':
                    skip = skips[(len(self.stage_sizes) - i - 1, len(stage) - j - 1)]
                    repeat = int(x.shape[0] / skip.shape[0])
                    skip = torch.repeat_interleave(skip, repeat, dim=0)
                    x = torch.concatenate([x, skip], axis=-1)
                elif self.skip_type in ['no_skip', 'pixel_residual_ff', 'pixel_residual_pf']:
                    pass
                elif self.skip_type is not None:
                    raise Exception('Unknown Skip Type.')
        x = self.Conv_0(x)
        x = torch.sigmoid(x)
        if has_time_dim: # add time dim back
            return x.view((shape[0], shape[1],) + tuple(x.shape[1:]))
        else:
            return x # for VAE


class FitVid(nn.Module):
    """FitVid video predictor."""

    def __init__(self, stage_sizes, z_dim, g_dim, rnn_size, num_base_filters, first_block_shape, expand_decoder,
                 skip_type, n_past, action_conditioned, action_size, is_inference, has_depth_predictor, beta, depth_weight,
                 loss_fn, stochastic):
        super(FitVid, self).__init__()
        self.n_past = n_past
        self.action_conditioned = action_conditioned
        self.beta = beta
        self.stochastic = stochastic
        self.is_inference = is_inference
        self.skip_type = skip_type
        self.has_depth_predictor = has_depth_predictor
        self.depth_weight = depth_weight
        self.loss_fn = loss_fn
        self.z_dim = z_dim

        if not is_inference:
            if FLAGS.depth_loss == 'norm_mse':
                self.depth_loss = self.depth_mse_loss
            elif FLAGS.depth_loss == 'eigen':
                self.depth_loss = self.eigen_depth_loss
            else:
                raise NotImplementedError

        first_block_shape = [first_block_shape[-1]] + first_block_shape[:2]
        self.encoder = ModularEncoder(stage_sizes=stage_sizes, output_size=g_dim, num_base_filters=num_base_filters)
        if self.stochastic:
            self.prior = MultiGaussianLSTM(input_size=g_dim, output_size=z_dim, hidden_size=rnn_size, num_layers=1)
            self.posterior = MultiGaussianLSTM(input_size=g_dim, output_size=z_dim, hidden_size=rnn_size, num_layers=1)
        else:
            self.prior, self.posterior = None, None

        self.decoder = ModularDecoder(first_block_shape=first_block_shape, input_size=g_dim, stage_sizes=stage_sizes,
                                      num_base_filters=num_base_filters, skip_type=skip_type, expand=expand_decoder)

        input_size = self.get_input_size(g_dim, action_size, z_dim)
        self.frame_predictor = MultiGaussianLSTM(input_size=input_size, output_size=g_dim, hidden_size=rnn_size, num_layers=2)
    
    def predict_depth(self, pred_frame, time_axis=True):
        if time_axis:
            shape = pred_frame.shape
            pred_frame = pred_frame.view(
                (shape[0] * shape[1],) + shape[2:])  # collapse batch*time dims [b0t0, b0t1, b0t2... b1t0, b1t1, b1t2...]
        pred_frame = (pred_frame - self.depth_head_mean) / self.depth_head_std # normalize as done for pretrained MiDaS
        #pred_frame = resize(pred_frame, scale_factors=6) # resize to 384x384

        pred_frame = pred_frame
        pred_frame = torch.nn.Upsample(scale_factor=4)(pred_frame)
        #pred_frame = pred_frame.to(memory_format=torch.channels_last).half()
        #pred_frame = pred_frame.half()
        depth_pred = self.depth_head(pred_frame)[:, None].float()
        # normalize to [0, 1]
        depth_pred = torch.nn.functional.interpolate(depth_pred, size=(64, 64))
        depth_pred = 1 - self.normalize(depth_pred, across_dims=1)
        if time_axis:
            return depth_pred.view((shape[0], shape[1],) + tuple(depth_pred.shape[1:]))
        else:
            return depth_pred

    def get_input(self, hidden, action, z):
        inp = [hidden]
        if self.action_conditioned:
            inp += [action]
        if self.stochastic:
            inp += [z]
        return torch.cat(inp, dim=1)

    def get_input_size(self, hidden_size, action_size, z_size):
        inp = hidden_size
        if self.action_conditioned:
            inp += action_size
        if self.stochastic:
            inp += z_size
        return inp

    def kl_divergence(self, mean1, logvar1, mean2, logvar2, batch_size):
        kld = 0.5 * (-1.0 + logvar2 - logvar1 + torch.exp(logvar1 - logvar2)
                     + torch.square(mean1 - mean2) * torch.exp(-logvar2))
        return torch.sum(kld) / batch_size

    def forward(self, video, actions, depth=None):
        batch_size, video_len = video.shape[0], video.shape[1]
        video = video.view((batch_size*video_len,) + video.shape[2:]) # collapse first two dims
        hidden, skips = self.encoder(video)
        hidden = hidden.view((batch_size, video_len) + hidden.shape[1:])
        video = video.view((batch_size, video_len,) + video.shape[1:])  # reconstruct first two dims
        skips = {k: skips[k].view((batch_size, video_len,) + tuple(skips[k].shape[1:]))[:, self.n_past - 1] for k in skips.keys()}
        loss, preds, metrics = self.calc_loss_helper(video, actions, hidden, skips, self.posterior, self.prior, self.frame_predictor, depth=depth)
        return loss, preds, metrics

    def calc_loss_helper(self, video, actions, hidden, skips, posterior, prior, frame_predictor, depth=None):
        if self.is_inference:
            assert False
        #video, actions = batch['video'], batch['actions']
        batch_size, video_len = video.shape[0], video.shape[1]
        pred_state = post_state = prior_state = None

        kld, means, logvars = torch.tensor(0).to(video), [], []
        # training
        h_preds = []
        for i in range(1, video_len):
            h, h_target = hidden[:, i - 1], hidden[:, i]
            if self.stochastic:
                (z_t, mu, logvar), post_state = posterior(h_target, post_state)
                (_, prior_mu, prior_logvar), prior_state = prior(h, prior_state)
            else:
                z_t = torch.zeros((h.shape[0], self.z_dim)).to(h)
            inp = self.get_input(h, actions[:, i - 1], z_t)
            (_, h_pred, _), pred_state = frame_predictor(inp, pred_state)
            h_pred = torch.sigmoid(h_pred) # TODO notice
            h_preds.append(h_pred)
            if self.stochastic:
                means.append(mu)
                logvars.append(logvar)
                kld += self.kl_divergence(mu, logvar, prior_mu, prior_logvar, batch_size)
        h_preds = torch.stack(h_preds, axis=1)
        preds = self.decoder(h_preds, skips)
        if self.stochastic:
            means = torch.stack(means, axis=1)
            logvars = torch.stack(logvars, axis=1)
        else:
            means, logvars = torch.zeros(h.shape[0], video_len-1, 1).to(h), torch.zeros(h.shape[0], video_len-1, 1).to(h)
        mse = self.loss_fn(preds, video[:, 1:])
        loss = mse + kld * self.beta
        # Metrics
        metrics = {
            'hist/mean': means,
            'hist/logvars': logvars,
            'loss/mse': mse,
            'loss/kld': kld,
            'loss/all': loss,
        }
        return loss, preds, metrics

    def depth_mse_loss(self, pred, gt, video=True):
        dims = 2 if video else 1
        return F.mse_loss(self.normalize(pred, across_dims=dims), self.normalize(gt, across_dims=dims))


    def evaluate(self, batch):
        """Predict the full video conditioned on the first self.n_past frames. """
        if self.is_inference:
            assert False
        video, actions = batch['video'], batch['actions']
        batch_size, video_len = video.shape[0], video.shape[1]
        pred_state = prior_state = None
        video = video.view((batch_size * video_len,) + video.shape[2:])  # collapse first two dims
        hidden, skips = self.encoder(video)
        skips = {k: skips[k].view((batch_size, video_len,) + tuple(skips[k].shape[1:]))[:, self.n_past - 1] for k in skips.keys()}
        # evaluating
        preds = []
        depth_preds = []
        masks = []
        hidden = hidden.view((batch_size, video_len) + hidden.shape[1:])
        for i in range(1, video_len):
            h, _ = hidden[:, i - 1], hidden[:, i]
            if i > self.n_past:
                h, _ = self.encoder(pred)
            if self.stochastic:
                (z_t, prior_mu, prior_logvar), prior_state = self.prior(h, prior_state)
            else:
                z_t = torch.zeros((h.shape[0], self.z_dim)).to(h)
            inp = self.get_input(h, actions[:, i - 1], z_t)
            (_, h_pred, _), pred_state = self.frame_predictor(inp, pred_state)
            h_pred = torch.sigmoid(h_pred) # TODO notice

            pred = self.decoder(h_pred[None, :], skips)[0]

            preds.append(pred)

        preds = torch.stack(preds, axis=1)

        video = video.view((batch_size, video_len,) + video.shape[1:])  # reconstuct first two dims
        mse = self.loss_fn(preds, video[:, 1:])

        if self.has_depth_predictor and 'depth_video' in batch:
            depth_preds = torch.stack(depth_preds, axis=1)
            depth_video = batch['depth_video']
            depth_loss = self.depth_loss(depth_preds, depth_video[:, 1:])
            mse = (mse, depth_loss)
            preds = (preds, depth_preds)

        return mse, preds

    def test(self, batch):
        """Predict the full video conditioned on the first self.n_past frames. """
        video, actions = batch['video'], batch['actions']
        batch_size, video_len = video.shape[0], video.shape[1]
        action_len = actions.shape[1]
        pred_state = prior_state = None
        video = video.view((batch_size * video_len,) + video.shape[2:])  # collapse first two dims
        hidden, skips = self.encoder(video)
        skips = {k: skips[k].view((batch_size, video_len,) + tuple(skips[k].shape[1:]))[:, self.n_past - 1] for k in skips.keys()}
        # evaluating
        preds = []
        depth_preds = []
        masks = []
        hidden = hidden.view((batch_size, video_len) + hidden.shape[1:])
        for i in range(1, action_len+1):
            if i <= self.n_past:
                h = hidden[:, i-1]
            if i > self.n_past:
                h, _ = self.encoder(pred)
            if self.stochastic:
                (z_t, prior_mu, prior_logvar), prior_state = self.prior(h, prior_state)
            else:
                z_t = torch.zeros((h.shape[0], self.z_dim)).to(h)
            inp = self.get_input(h, actions[:, i - 1], z_t)
            (_, h_pred, _), pred_state = self.frame_predictor(inp, pred_state)
            h_pred = torch.sigmoid(h_pred) # TODO notice
            pred = self.decoder(h_pred[None, :], skips)[0]
            preds.append(pred)
        preds = torch.stack(preds, axis=1)
        video = video.view((batch_size, video_len,) + video.shape[1:])  # reconstuct first two dims
        return preds

    def normalize(self, t, across_dims=1):
        dims = tuple(range(len(t.shape))[across_dims:])
        t = t - t.amin(dim=dims, keepdim=True)
        t = t / (t.amax(dim=dims, keepdim=True) + 1e-10)
        return t

    def load_parameters(self, path):
        # load everything
        state_dict = torch.load(path)
        self.load_state_dict(state_dict, strict=True)
        print(f'Loaded checkpoint {path}')


def load_data(dataset_files, data_type='train', depth=False):
    video_len = FLAGS.n_past + FLAGS.n_future
    return robomimic_data.load_dataset_robomimic_torch(dataset_files, FLAGS.batch_size, video_len, data_type, depth)


def dict_to_cuda(d):
    return {k: v.cuda() for k, v in d.items()}


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


def depth_to_rgb_im(im, cmap):
    # shape = [(T), 1, W, H]
    # normalize
    axes = tuple(range(len(im.shape)))[-3:]
    im = im - np.amin(im, axis=axes, keepdims=True)
    im = im / np.amax(im, axis=axes, keepdims=True)
    im = np.squeeze(im)
    # convert to rgb using given cmap
    im = cmap(im)[..., :3]
    return im * 255


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
                   action_size=FLAGS.action_size, # hardcode for now
                   is_inference=False,
                   has_depth_predictor=FLAGS.depth_objective,
                   expand_decoder=FLAGS.expand,
                   beta=FLAGS.beta,
                   depth_weight=FLAGS.depth_weight,
                   loss_fn=loss_fn,
                   stochastic=FLAGS.stochastic
                   )
    NGPU = torch.cuda.device_count()
    print('CUDA available devices: ', NGPU)

    checkpoint, resume_epoch = get_most_recent_checkpoint(FLAGS.output_dir)
    print(f'Attempting to load from checkpoint {checkpoint if checkpoint else "None, no model found"}')
    if checkpoint:
        model.load_parameters(checkpoint)

    if NGPU > 1:
        model = torch.nn.DataParallel(model)
    model.to('cuda:0')

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=FLAGS.weight_decay)

    if FLAGS.hdf5_data:
        from fitvid.hdf5_data_loader import load_hdf5_data
        data_loader = load_hdf5_data(FLAGS.dataset_file, FLAGS.batch_size, data_type='train')
        test_data_loader = load_hdf5_data(FLAGS.dataset_file, FLAGS.batch_size, data_type='val')
        prep_data = prep_data_test = lambda x: x
    else:
        data_loader, prep_data = load_data(FLAGS.dataset_file, data_type='train', depth=FLAGS.depth_objective)
        test_data_loader, prep_data_test = load_data(FLAGS.dataset_file, data_type='valid', depth=FLAGS.depth_objective)

    wandb.init(
        project='perceptual-metrics',
        reinit=True,
        #resume=True,
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
    test_depth_mse = []
    num_epochs = FLAGS.num_epochs
    train_steps = 0
    num_batch_to_save = 1

    for epoch in range(resume_epoch+1, num_epochs):
        predicting_depth = FLAGS.depth_objective and epoch > FLAGS.depth_start_epoch

        print(f'\nEpoch {epoch} / {num_epochs}')
        train_save_videos = []
        test_save_videos = []
        train_save_depth_videos = []
        test_save_depth_videos = []

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
                if not predicting_depth and 'depth_video' in batch:
                    batch.pop('depth_video')
                if NGPU > 1:
                    mse, eval_preds = model.module.evaluate(batch)
                else:
                    mse, eval_preds = model.evaluate(batch)
                if predicting_depth:
                    mse, depth_mse = mse
                    eval_preds, eval_depth_preds = eval_preds
                    if test_batch_idx < num_batch_to_save:
                        save_depth_vids = torch.cat([batch['depth_video'][:4, 1:], eval_depth_preds[:4]], dim=-1).detach().cpu().numpy() # save only first 4 of a batch
                        [test_save_depth_videos.append(vid) for vid in save_depth_vids]
                    epoch_depth_mse.append(depth_mse.item())
                if test_batch_idx < num_batch_to_save:
                    save_vids = torch.cat([test_videos[:4, 1:], eval_preds[:4]], dim=-1).detach().cpu().numpy() # save only first 4 of a batch
                    [test_save_videos.append(vid) for vid in save_vids]
                epoch_mse.append(mse.item())
                if FLAGS.debug and test_batch_idx > 5:
                    break
            test_mse.append(np.mean(epoch_mse))
            wandb_log.update({'eval/mse': mse})
            if predicting_depth:
                test_depth_mse.append(np.mean(epoch_depth_mse))
                wandb_log.update({'eval/depth_mse': np.mean(epoch_depth_mse)})
                print(f'Test Depth MSE: {test_depth_mse[-1]}')
            wandb.log(wandb_log)
        print(f'Test MSE: {test_mse[-1]}')

        if test_mse[-1] == np.min(test_mse):
            if os.path.isdir(FLAGS.output_dir):
                pass
            else:
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
            #batch = prep_data(batch)
            if predicting_depth:
                inputs = batch['video'], batch['actions'], batch['depth_video']
            else:
                inputs = batch['video'], batch['actions']
            loss, preds, metrics = model(*inputs)

            if predicting_depth:
                preds, depth_preds = preds

            if NGPU > 1:
                loss = loss.mean()
                metrics = {k: v.mean() for k, v in metrics.items()}
            #print(loss)
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
                if predicting_depth:
                    save_depth_vids = torch.cat([batch['depth_video'][:4, 1:], depth_preds[:4]], dim=-1).detach().cpu().numpy()
                    [train_save_depth_videos.append(vid) for vid in save_depth_vids]

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

        # plot the train/test curve so far
        # plt.figure()
        # x_train = np.linspace(0, epoch+1, len(train_mse))
        # plt.plot(x_train, train_mse, 'b', marker='x', label='train_mse')
        # plt.plot(np.arange(epoch+1), test_mse, 'r', marker='x', label='test_mse')
        # plt.xlabel('Epoch')
        # plt.ylabel('MSE')
        # plt.legend()
        # plt.savefig(os.path.join(FLAGS.output_dir, 'losses.png'))
        # plt.close()
        # print('Saved loss curve')

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

            cmap_name = 'jet_r' #or 'inferno'?
            colormap = plt.get_cmap(cmap_name)

            for i, vid in enumerate(test_save_depth_videos):
                vids = np.split(vid, 2, axis=-1)
                vid = np.concatenate([depth_to_rgb_im(v, colormap) for v in vids], axis=-2)
                wandb_log.update({f'video_depth_test_{i}': wandb.Video(np.moveaxis(vid, 3, 1), fps=4, format='gif')})
                save_moviepy_gif(list(vid), os.path.join(FLAGS.output_dir, f'video_depth_test_epoch{epoch}_{i}'), fps=5)
                print('Saved test depth vid for the epoch')

            for i, vid in enumerate(train_save_depth_videos):
                vids = np.split(vid, 2, axis=-1)
                vid = np.concatenate([depth_to_rgb_im(v, colormap) for v in vids], axis=-2)
                wandb_log.update({f'video_depth_train_{i}': wandb.Video(np.moveaxis(vid, 3, 1), fps=4, format='gif')})
                save_moviepy_gif(list(vid), os.path.join(FLAGS.output_dir, f'video_depth_train_epoch{epoch}_{i}'), fps=5)
                print('Saved train depth vid for the epoch')
            wandb.log(wandb_log)


def save_moviepy_gif(obs_list, name, fps=5):
    from moviepy.editor import ImageSequenceClip
    clip = ImageSequenceClip(obs_list, fps=fps)
    clip.write_gif(f'{name}.gif', fps=fps)

if __name__ == "__main__":
    flags.mark_flags_as_required(['output_dir'])
    app.run(main)
