import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from fitvid.model.nvae import ModularEncoder, ModularDecoder
from fitvid.model.depth_predictor import DepthPredictor
from fitvid.utils.depth_utils import pixel_wise_loss, pixel_wise_loss_segmented
from fitvid.utils.pytorch_metrics import psnr, lpips, ssim, tv, fvd, PolicyFeatureL2Metric

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

def init_weights_lecun(m):
    """
    Perform LeCun normal initialization for parameters of module m
    Since the default Flax initialization uses LeCun uniform, unlike pytorch default, we use this to try to match the
    official implementation.
    """
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
        # Initialize weights to LeCun normal initialization
        fan_in, fan_out = nn.init._calculate_fan_in_and_fan_out(m.weight)
        std = np.sqrt(1 / fan_in)
        nn.init.trunc_normal_(m.weight, mean=0, std=std, a=-2*std, b=2*std)
        # Initialize biases to zero
        if m.bias is not None:
            nn.init.zeros_(m.bias)
    elif isinstance(m, nn.BatchNorm2d):
        nn.init.ones_(m.weight)
        nn.init.zeros_(m.bias)
    elif isinstance(m, nn.LSTM):
        # Handle LSTMs. In jax, the kernels which transform the input are initialized with LeCun normal, and the
        # ones which transform the hidden state are initialized with orthogonal.
        for name, param in m.named_parameters():
            if 'weight_hh' in name:
                for i in range(0, param.shape[0], param.shape[0]//4):
                    nn.init.orthogonal_(param[i:i+param.shape[0]//4])
            elif 'weight_ih' in name:
                fan_in, fan_out = nn.init._calculate_fan_in_and_fan_out(param)
                std = np.sqrt(1 / fan_in)
                nn.init.trunc_normal_(param, mean=0, std=std, a=-2 * std, b=2 * std)
            elif 'bias' in name:
                nn.init.zeros_(param)

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
        self.num_video_channels = model_kwargs.get('video_channels', 3)

        first_block_shape = [model_kwargs['first_block_shape'][-1]] + model_kwargs['first_block_shape'][:2]
        self.encoder = ModularEncoder(stage_sizes=model_kwargs['stage_sizes'], output_size=model_kwargs['g_dim'],
                                      num_base_filters=model_kwargs['num_base_filters'],
                                      num_input_channels=self.num_video_channels)

        if self.stochastic:
            self.prior = MultiGaussianLSTM(input_size=model_kwargs['g_dim'], output_size=model_kwargs['z_dim'],
                                           hidden_size=model_kwargs['rnn_size'], num_layers=1)
            self.posterior = MultiGaussianLSTM(input_size=model_kwargs['g_dim'], output_size=model_kwargs['z_dim'],
                                               hidden_size=model_kwargs['rnn_size'], num_layers=1)
        else:
            self.prior, self.posterior = None, None

        self.decoder = ModularDecoder(first_block_shape=first_block_shape, input_size=model_kwargs['g_dim'],
                                      stage_sizes=model_kwargs['stage_sizes'],
                                      num_base_filters=model_kwargs['num_base_filters'],
                                      skip_type=model_kwargs['skip_type'], expand=model_kwargs['expand_decoder'],
                                      num_output_channels=self.num_video_channels)

        input_size = self.get_input_size(model_kwargs['g_dim'], model_kwargs['action_size'], model_kwargs['z_dim'])
        self.frame_predictor = MultiGaussianLSTM(input_size=input_size, output_size=model_kwargs['g_dim'],
                                                 hidden_size=model_kwargs['rnn_size'], num_layers=2)

        if model_kwargs.get("lecun_initialization", None):
            self.apply(init_weights_lecun)

        self.lpips = piq.LPIPS()

        self.rgb_loss_type = kwargs.get('rgb_loss_type', 'l2')

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

    def setup_train_losses(self):
        if self.rgb_loss_type == 'l2':
            self.rgb_loss = nn.MSELoss()
        elif self.rgb_loss_type == 'l1':
            self.rgb_loss = nn.L1Loss()

        if self.config.get('corr_wise', None):
            from fitvid.utils.corrwise_loss import CorrWiseLoss
            self.rgb_loss = CorrWiseLoss(self.rgb_loss,
                                     backward_warp=True,
                                     return_warped=False,
                                     padding_mode='reflection',
                                     scale_clip=0.1)

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

    def compute_metrics(self, vid1, vid2, segmentation=None):
        with torch.no_grad():
            metrics = {
                'metrics/psnr': psnr(vid1, vid2),
                'metrics/lpips': lpips(self.lpips, vid1, vid2),
                'metrics/tv': tv(vid1),
                'metrics/ssim': ssim(vid1, vid2),
                'metrics/fvd': fvd(vid1, vid2),
            }

            if segmentation is not None:
                per_sample_segmented_mse = pixel_wise_loss_segmented(vid1, vid2, segmentation, loss=self.rgb_loss_type, reduce_batch=False)
                metrics['metrics/segmented_mse'] = per_sample_segmented_mse.mean()
                metrics['metrics/segmented_mse_per_sample'] = per_sample_segmented_mse.detach()

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
        preds, kld, means, logvars = self.predict_rgb(video, actions, hidden, skips)
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
                # initialize mask to be a torch tensor of all ones with same shape as video
                with torch.no_grad():
                    mse_per_sample = pixel_wise_loss(preds['rgb'], video[:, 1:], loss='l2', reduce_batch=False, mask=None)
                    l1_per_sample = pixel_wise_loss(preds['rgb'], video[:, 1:], loss='l1', reduce_batch=False, mask=None)
                total_loss += self.rgb_loss(preds['rgb'], video[:, 1:]) * weight
                metrics['loss/mse'] = mse_per_sample.mean().detach()
                metrics['loss/mse_per_sample'] = mse_per_sample.detach()
                metrics['loss/l1_loss'] = l1_per_sample.mean().detach()
                metrics['loss/l1_loss_per_sample'] = l1_per_sample.detach()
            elif loss == 'segmented_object':
                if weight > 0:
                    segmented_mse_per_sample = pixel_wise_loss_segmented(preds['rgb'], video[:, 1:], segmentation[:, 1:],
                                                                         loss=self.rgb_loss_type, reduce_batch=False)
                    total_loss += segmented_mse_per_sample.mean() * weight
                    metrics['loss/segmented_mse'] = segmented_mse_per_sample.mean()
                    metrics['loss/segmented_mse_per_sample'] = segmented_mse_per_sample.detach()
            elif loss == 'tv':
                if weight != 0:
                    tv_loss = tv(preds)
                    total_loss += weight * tv_loss
                    metrics['loss/tv'] = tv_loss
            elif loss == 'lpips':
                if weight != 0:
                    lpips_loss = lpips(self.lpips, preds['rgb'], video[:, 1:])
                    total_loss += weight * lpips_loss
                    metrics['loss/lpips'] = lpips_loss
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
            if segmentation is not None:
                metrics.update(
                    self.compute_metrics(preds['rgb'], video[:, 1:], segmentation[:, 1:])
                )
            else:
                metrics.update(
                    self.compute_metrics(preds['rgb'], video[:, 1:])
                )

        return total_loss, preds, metrics

    def predict_rgb(self, video, actions, hidden, skips):
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
                (z_t, mu, logvar), post_state = self.posterior(h_target, post_state)
                (_, prior_mu, prior_logvar), prior_state = self.prior(h, prior_state)
                inp = self.get_input(h, actions[:, i - 1], z_t)
                (_, h_pred, _), pred_state = self.frame_predictor(inp, pred_state)
                h_pred = torch.sigmoid(h_pred)  # TODO notice
                h_preds.append(h_pred)
                means.append(mu)
                logvars.append(logvar)
                kld += self.kl_divergence(mu, logvar, prior_mu, prior_logvar, batch_size)
                pred = self.decoder(h_pred, skips, has_time_dim=False)
                preds.append(pred)
            preds = torch.stack(preds, axis=1)
        else:
            for i in range(1, video_len):
                h, h_target = hidden[:, i - 1], hidden[:, i]
                (z_t, mu, logvar), post_state = self.posterior(h_target, post_state)
                (_, prior_mu, prior_logvar), prior_state = self.prior(h, prior_state)
                inp = self.get_input(h, actions[:, i - 1], z_t)
                (_, h_pred, _), pred_state = self.frame_predictor(inp, pred_state)
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
        ag_metrics, ag_preds = self._evaluate(batch, compute_metrics, autoregressive=True)
        non_ag_metrics, non_ag_preds = self._evaluate(batch, compute_metrics, autoregressive=False)
        ag_metrics = {f'ag/{k}': v for k, v in ag_metrics.items()}
        non_ag_metrics = {f'non_ag/{k}': v for k, v in non_ag_metrics.items()}
        metrics = {**ag_metrics, **non_ag_metrics}
        return metrics, dict(ag=ag_preds, non_ag=non_ag_preds)

    def _evaluate(self, batch, compute_metrics=False, autoregressive=True):
        """Predict the full video conditioned on the first self.n_past frames. """
        video, actions, segmentation = batch['video'], batch['actions'], batch.get('segmentation', None)
        batch_size, video_len = video.shape[0], video.shape[1]
        pred_state = prior_state = post_state = None
        video = video.view((batch_size * video_len,) + video.shape[2:])  # collapse first two dims
        hidden, skips = self.encoder(video)
        skips = {k: skips[k].view((batch_size, video_len,) + tuple(skips[k].shape[1:]))[:, self.n_past - 1] for k in skips.keys()}
        # evaluating
        preds = []
        hidden = hidden.view((batch_size, video_len) + hidden.shape[1:])
        if autoregressive:
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
        else:
            h_preds = []
            kld = torch.tensor(0).to(video)
            for i in range(1, video_len):
                h, h_target = hidden[:, i - 1], hidden[:, i]
                (z_t, mu, logvar), post_state = self.posterior(h_target, post_state)
                (_, prior_mu, prior_logvar), prior_state = self.prior(h, prior_state)
                inp = self.get_input(h, actions[:, i - 1], z_t)
                (_, h_pred, _), pred_state = self.frame_predictor(inp, pred_state)
                h_pred = torch.sigmoid(h_pred)  # TODO notice
                h_preds.append(h_pred)
                kld += self.kl_divergence(mu, logvar, prior_mu, prior_logvar, batch_size)
            h_preds = torch.stack(h_preds, axis=1)
            preds = self.decoder(h_preds, skips)

        video = video.view((batch_size, video_len,) + video.shape[1:])  # reconstuct first two dims
        mse_per_sample = pixel_wise_loss(preds, video[:, 1:], loss='l2', reduce_batch=False)
        mse = mse_per_sample.mean()
        l1_loss_per_sample = pixel_wise_loss(preds, video[:, 1:], loss='l1', reduce_batch=False)
        l1_loss = l1_loss_per_sample.mean()
        metrics = {
            'loss/mse': mse,
            'loss/mse_per_sample': mse_per_sample,
            'loss/l1_loss': l1_loss,
            'loss/l1_loss_per_sample': l1_loss_per_sample,
        }

        if compute_metrics:
            if segmentation is not None:
                metrics.update(self.compute_metrics(preds, video[:, 1:], segmentation[:, 1:]))
            else:
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

    def state_dict(self, *args, **kwargs):
        prefixes_to_remove = ['rgb_loss']
        # get superclass state_dict
        state_dict = super().state_dict(*args, **kwargs)
        # create another state dict without items having prefixes in prefixes_to_remove
        new_state_dict = dict()
        for k, v in state_dict.items():
            if not any(prefix in k for prefix in prefixes_to_remove):
                new_state_dict[k] = v
        return new_state_dict


