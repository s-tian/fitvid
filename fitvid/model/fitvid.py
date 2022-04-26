import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from fitvid.model.nvae import ModularEncoder, ModularDecoder
from fitvid.model.depth_predictor import DepthPredictor
from fitvid.utils.depth_utils import mse_loss
from fitvid.utils.pytorch_metrics import psnr, lpips, ssim, tv, PolicyFeatureL2Metric

import piq

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


class FitVid(nn.Module):
    """FitVid video predictor."""

    def __init__(self, **kwargs):
        super(FitVid, self).__init__()
        self.config = kwargs
        model_kwargs = kwargs['model_kwargs']
        self.n_past = model_kwargs['n_past']
        self.action_conditioned = model_kwargs['action_conditioned']
        self.beta = kwargs['beta']
        self.stochastic = model_kwargs['stochastic']
        self.is_inference = kwargs['is_inference']
        self.skip_type = model_kwargs['skip_type']
        self.loss_weights = kwargs['loss_weights']
        self.loss_weights['kld'] = self.beta
        self.multistep = kwargs['multistep']
        self.z_dim = model_kwargs['z_dim']

        first_block_shape = [model_kwargs['first_block_shape'][-1]] + model_kwargs['first_block_shape'][:2]
        self.encoder = ModularEncoder(stage_sizes=model_kwargs['stage_sizes'], output_size=model_kwargs['g_dim'], num_base_filters=model_kwargs['num_base_filters'])
        if self.stochastic:
            self.prior = MultiGaussianLSTM(input_size=model_kwargs['g_dim'], output_size=model_kwargs['z_dim'], hidden_size=model_kwargs['rnn_size'], num_layers=1)
            self.posterior = MultiGaussianLSTM(input_size=model_kwargs['g_dim'], output_size=model_kwargs['z_dim'], hidden_size=model_kwargs['rnn_size'], num_layers=1)
        else:
            self.prior, self.posterior = None, None

        self.decoder = ModularDecoder(first_block_shape=first_block_shape, input_size=model_kwargs['g_dim'], stage_sizes=model_kwargs['stage_sizes'],
                                      num_base_filters=model_kwargs['num_base_filters'], skip_type=model_kwargs['skip_type'], expand=model_kwargs['expand_decoder'])

        input_size = self.get_input_size(model_kwargs['g_dim'], model_kwargs['action_size'], model_kwargs['z_dim'])
        self.frame_predictor = MultiGaussianLSTM(input_size=input_size, output_size=model_kwargs['g_dim'], hidden_size=model_kwargs['rnn_size'], num_layers=2)

        self.lpips = piq.LPIPS()

        if kwargs.get('depth_predictor', None):
            self.has_depth_predictor = True
            self.depth_predictor_cfg = kwargs['depth_predictor']
            self.load_depth_predictor()
        else:
            self.has_depth_predictor = False

        if kwargs.get('normal_predictor', None):
            self.has_normal_predictor = True
            self.normal_predictor_cfg = kwargs['normal_predictor']
            self.load_normal_predictor()
        else:
            self.has_normal_predictor = False

        if kwargs.get('policy_networks', None):
            self.policy_feature_metric = True
            self.policy_networks_cfg = kwargs['policy_networks']
            self.load_policy_networks()
        else:
            self.policy_feature_metric = False

    def load_policy_networks(self):
        layer = self.policy_networks_cfg['layer']
        paths = self.policy_networks_cfg['pretrained_weight_paths']
        self.policy_network_losses = nn.ModuleList([PolicyFeatureL2Metric(path, layer) for path in paths])

    def load_normal_predictor(self):
        from fitvid.scripts.train_surface_normal_model import ConvPredictor
        self.normal_predictor = ConvPredictor()
        self.normal_predictor.load_state_dict(torch.load(self.normal_predictor_cfg['pretrained_weight_path']))
        for param in self.normal_predictor.parameters():
            param.requires_grad = False

    def load_depth_predictor(self):
        self.depth_predictor = DepthPredictor(**self.depth_predictor_cfg)

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

    def compute_metrics(self, vid1, vid2):
        with torch.no_grad():
            metrics = {
                'metrics/psnr': psnr(vid1, vid2),
                'metrics/lpips': lpips(self.lpips, vid1, vid2),
                'metrics/tv': tv(vid1),
                'metrics/ssim': ssim(vid1, vid2),
            }
            if self.policy_feature_metric:
                for i, policy_feature_metric in enumerate(self.policy_network_losses):
                    action_mse, feature_mse = policy_feature_metric(vid1, vid2)
                    metrics.update({
                        f'metrics/action_{i}_mse': action_mse,
                        f'metrics/policy_{i}_feature_mse': feature_mse,
                    })
            return metrics

    def forward(self, video, actions, segmentation=None, depth=None, normal=None, compute_metrics=False):
        batch_size, video_len = video.shape[0], video.shape[1]
        video = video.view((batch_size*video_len,) + video.shape[2:]) # collapse first two dims
        hidden, skips = self.encoder(video)
        hidden = hidden.view((batch_size, video_len) + hidden.shape[1:])
        video = video.view((batch_size, video_len,) + video.shape[1:])  # reconstruct first two dims
        skips = {k: skips[k].view((batch_size, video_len,) + tuple(skips[k].shape[1:]))[:, self.n_past - 1] for k in skips.keys()}
        preds, kld, means, logvars = self.predict_rgb(video, actions, hidden, skips, self.posterior, self.prior, self.frame_predictor)
        loss, preds, metrics = self.compute_loss(preds, video, kld,
                                                 segmentation=segmentation, depth=depth, normal=normal, compute_metrics=compute_metrics)
        metrics.update({
            'hist/mean': means,
            'hist/logvars': logvars,
        })
        return loss, preds, metrics

    def compute_loss(self, preds, video, kld, segmentation=None, depth=None, normal=None, compute_metrics=False):
        total_loss = 0
        metrics = dict()
        preds = dict(rgb=preds)
        for loss, weight in self.loss_weights.items():
            if loss == 'kld':
                total_loss += weight * kld
                metrics['loss/kld'] = kld
            elif loss == 'rgb':
                mse_per_sample = mse_loss(preds['rgb'], video[:, 1:], reduce_batch=False)
                total_loss += mse_per_sample.mean()
                metrics['loss/mse'] = mse_per_sample.mean()
                metrics['loss/mse_per_sample'] = mse_per_sample.detach()
            elif loss == 'tv':
                if weight != 0:
                    tv_loss = tv(preds)
                    total_loss += weight * tv_loss
                metrics['loss/tv'] = tv_loss
            elif loss == 'policy':
                if weight != 0 and self.policy_feature_metric:
                    feature_losses = []
                    for policy_feature_metric in self.policy_network_losses:
                        action_mse, feature_mse = policy_feature_metric(preds['rgb'], video[:, 1:])
                        feature_losses.append(feature_mse)
                    feature_mse = torch.stack(feature_losses).mean()
                    metrics['loss/policy_feature_loss'] = feature_mse
                    total_loss = total_loss + weight * feature_mse
            elif loss == 'depth':
                if self.has_depth_predictor:
                    if weight != 0:
                        depth_preds = self.depth_predictor(preds['rgb'])
                        depth_loss_per_sample = self.depth_predictor.depth_loss(depth_preds, depth[:, 1:], reduce_batch=False)
                        depth_loss = depth_loss_per_sample.mean()
                        total_loss = total_loss + weight * depth_loss
                    else:
                        with torch.no_grad():
                            depth_preds = self.depth_predictor(preds['rgb'])
                            depth_loss_per_sample = self.depth_predictor.depth_loss(depth_preds, depth[:, 1:], reduce_batch=False)
                            depth_loss = depth_loss_per_sample.mean()
                    preds['depth'] = depth_preds
                    metrics['loss/depth_loss'] = depth_loss
                    metrics['loss/depth_loss_per_sample'] = depth_loss_per_sample.detach()
                elif weight != 0:
                    raise ValueError('Trying to use positive depth weight but no depth predictor!')
            elif loss == 'normal':
                if self.has_normal_predictor:
                    if weight != 0:
                        normal_preds = self.normal_predictor(preds['rgb'])
                        normal_loss_per_sample = mse_loss(normal_preds, normal[:, 1:], reduce_batch=False)
                        normal_loss = normal_loss_per_sample.mean()
                        total_loss = total_loss + weight * normal_loss
                    else:
                        with torch.no_grad():
                            normal_preds = self.normal_predictor(preds['rgb'])
                            normal_loss_per_sample = mse_loss(normal_preds, normal[:, 1:], reduce_batch=False)
                            normal_loss = normal_loss_per_sample.mean()
                    preds['normal'] = normal_preds
                    metrics['loss/normal_loss'] = normal_loss
                    metrics['loss/normal_loss_per_sample'] = normal_loss_per_sample.detach()
            else:
                raise NotImplementedError(f'Loss {loss} not implemented!')

        # Metrics
        metrics.update({
            'loss/all': total_loss,
        })
        if compute_metrics:
            metrics.update(
                self.compute_metrics(preds['rgb'], video[:, 1:])
            )

        return total_loss, preds, metrics

    def predict_rgb(self, video, actions, hidden, skips, posterior, prior, frame_predictor):
        if self.is_inference:
            assert False

        batch_size, video_len = video.shape[0], video.shape[1]
        pred_state = post_state = prior_state = None

        kld, means, logvars = torch.tensor(0).to(video), [], []
        # training
        h_preds = []
        if self.training and self.multistep:
            preds = []
            for i in range(1, video_len):
                h, h_target = hidden[:, i - 1], hidden[:, i]
                if i > self.n_past:
                    h, _ = self.encoder(pred)
                (z_t, mu, logvar), post_state = posterior(h_target, post_state)
                (_, prior_mu, prior_logvar), prior_state = prior(h, prior_state)
                inp = self.get_input(h, actions[:, i - 1], z_t)
                (_, h_pred, _), pred_state = frame_predictor(inp, pred_state)
                h_pred = torch.sigmoid(h_pred)  # TODO notice
                h_preds.append(h_pred)
                means.append(mu)
                logvars.append(logvar)
                kld += self.kl_divergence(mu, logvar, prior_mu, prior_logvar, batch_size)
                pred = self.decoder(h_pred[None, :], skips)[0]
                preds.append(pred)
            preds = torch.stack(preds, axis=1)
        else:
            for i in range(1, video_len):
                h, h_target = hidden[:, i - 1], hidden[:, i]
                (z_t, mu, logvar), post_state = posterior(h_target, post_state)
                (_, prior_mu, prior_logvar), prior_state = prior(h, prior_state)
                inp = self.get_input(h, actions[:, i - 1], z_t)
                (_, h_pred, _), pred_state = frame_predictor(inp, pred_state)
                h_pred = torch.sigmoid(h_pred)  # TODO notice
                h_preds.append(h_pred)
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
        return preds, kld, means, logvars

    def evaluate(self, batch, compute_metrics=False):
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
        mse_per_sample = mse_loss(preds, video[:, 1:], reduce_batch=False)
        mse = mse_per_sample.mean()
        metrics = {
            'loss/mse': mse,
            'loss/mse_per_sample': mse_per_sample,
        }

        if compute_metrics:
            metrics.update(self.compute_metrics(preds, video[:, 1:]))

        preds = dict(rgb=preds)

        if self.has_depth_predictor:
            with torch.no_grad():
                depth_preds = self.depth_predictor(preds['rgb'], time_axis=True)
                depth_video = batch['depth_video']
                depth_loss_per_sample = self.depth_predictor.depth_loss(depth_preds, depth_video[:, 1:], reduce_batch=False)
                depth_loss = depth_loss_per_sample.mean()
                metrics.update({
                    'loss/depth_loss': depth_loss,
                    'loss/depth_loss_per_sample': depth_loss_per_sample,
                })

            preds['depth'] = depth_preds

        if self.has_normal_predictor:
            with torch.no_grad():
                normal_preds = self.normal_predictor(preds['rgb'], time_axis=True)
                normal_video = batch['normal']
                normal_loss_per_sample = mse_loss(normal_preds, normal_video[:, 1:], reduce_batch=False)
                normal_loss = normal_loss_per_sample.mean()
                metrics.update({
                    'loss/normal_loss': normal_loss,
                    'loss/normal_loss_per_sample': normal_loss_per_sample,
                })
            preds['normal'] = normal_preds
        return metrics, preds

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

    def load_parameters(self, path):
        # load everything
        state_dict = torch.load(path)
        if 'module.encoder.stages.0.0.Conv_0.weight' in state_dict:
            new_state_dict = {k[7:]: v for k, v in state_dict.items() if k[:7] == 'module.'}
            state_dict = new_state_dict
        self.load_state_dict(state_dict, strict=True)
        print(f'Loaded checkpoint {path}')
        if self.has_depth_predictor:
            self.load_depth_predictor()  # reload pretrained depth model
            print('Reloaded depth model')
        if self.has_normal_predictor:
            self.load_normal_predictor()
            print('Reloaded normal model')
        if self.policy_feature_metric:
            self.load_policy_networks()
            print('Reloaded policy networks')
