import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from fitvid.model.nvae import ModularEncoder, ModularDecoder


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
        self.n_past = kwargs['n_past']
        self.action_conditioned = kwargs['action_conditioned']
        self.beta = kwargs['beta']
        self.stochastic = True
        self.is_inference = kwargs['is_inference']
        self.skip_type = kwargs['skip_type']

        if kwargs['loss_fn'] == 'l2':
            self.loss_fn = F.mse_loss
        elif kwargs['loss_fn'] == 'l1':
            self.loss_fn = F.l1_loss
        else:
            raise NotImplementedError

        first_block_shape = [kwargs['first_block_shape'][-1]] + kwargs['first_block_shape'][:2]
        self.encoder = ModularEncoder(stage_sizes=kwargs['stage_sizes'], output_size=kwargs['g_dim'], num_base_filters=kwargs['num_base_filters'])
        if self.stochastic:
            self.prior = MultiGaussianLSTM(input_size=kwargs['g_dim'], output_size=kwargs['z_dim'], hidden_size=kwargs['rnn_size'], num_layers=1)
            self.posterior = MultiGaussianLSTM(input_size=kwargs['g_dim'], output_size=kwargs['z_dim'], hidden_size=kwargs['rnn_size'], num_layers=1)
        else:
            self.prior, self.posterior = None, None

        self.decoder = ModularDecoder(first_block_shape=first_block_shape, input_size=kwargs['g_dim'], stage_sizes=kwargs['stage_sizes'],
                                      num_base_filters=kwargs['num_base_filters'], skip_type=kwargs['skip_type'], expand=kwargs['expand_decoder'])

        input_size = self.get_input_size(kwargs['g_dim'], kwargs['action_size'], kwargs['z_dim'])
        self.frame_predictor = MultiGaussianLSTM(input_size=input_size, output_size=kwargs['g_dim'], hidden_size=kwargs['rnn_size'], num_layers=2)

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

    def forward(self, video, actions):
        batch_size, video_len = video.shape[0], video.shape[1]
        video = video.view((batch_size*video_len,) + video.shape[2:]) # collapse first two dims
        hidden, skips = self.encoder(video)
        hidden = hidden.view((batch_size, video_len) + hidden.shape[1:])
        video = video.view((batch_size, video_len,) + video.shape[1:])  # reconstruct first two dims
        skips = {k: skips[k].view((batch_size, video_len,) + tuple(skips[k].shape[1:]))[:, self.n_past - 1] for k in skips.keys()}
        loss, preds, metrics = self.calc_loss_helper(video, actions, hidden, skips, self.posterior, self.prior, self.frame_predictor)
        return loss, preds, metrics

    def calc_loss_helper(self, video, actions, hidden, skips, posterior, prior, frame_predictor):
        if self.is_inference:
            assert False

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
        self.load_state_dict(state_dict, strict=True)
        print(f'Loaded checkpoint {path}')