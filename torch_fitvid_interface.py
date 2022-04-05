import torch
import numpy as np
from fitvid.fitvid_torch import FitVid


class FitVidTorchModel:

    DEFAULT_HP = {
        'stage_sizes': [1, 1, 1, 1],
        'z_dim': 10,
        'g_dim': 64,
        'rnn_size': 128,
        'num_base_filters': 32,
        'first_block_shape': [8, 8, 512],
        'expand_decoder': 1,
        'skip_type': 'residual',
        'n_past': 2,
        'action_conditioned': True,
        'action_size': 8,
        'is_inference': True,
        'has_depth_predictor': False,  # TODO don't hardcode this
        'beta': 0,  # for training, irrelevant here
        'depth_weight': 0,  # for training
        'stochastic': False
    }

    def __init__(self, checkpoint_dir, hp_override):
        # TODO import hyperparameters nicely
        self.hp = self.DEFAULT_HP
        self.hp.update(hp_override)
        self.model = FitVid(**hp)
        self.model.load_parameters(checkpoint_dir)
        self.model.cuda()
        self.model.eval()

    def prepare_batch(self, xs):
        keys = ['video', 'actions']
        batch = {k: torch.from_numpy(x).cuda().float() for k, x in xs.items() if k in keys}
        batch['video'] = torch.permute(batch['video'], (0, 1, 4, 2, 3))
        return batch

    def __call__(self, batch):
        with torch.no_grad():
            batch = self.prepare_batch(batch)
            preds = self.model.test(batch)
        preds = torch.permute(preds, (0, 1, 3, 4, 2))
        return preds.cpu().numpy()


if __name__ == '__main__':

    checkpoint_dir = 'soem path here'
    hp_override = {
        'stochastic': True,
        'rnn_size': 64
    }
    model = FitVidTorchModel(checkpoint_dir, hp_override)
    batch_size = 200
    batch = {
        'video': np.zeros((batch_size, 2, 64, 64, 3)),
        'actions':  np.zeros((batch_size, 11, 8)),
    }
    predictions = model(batch)
    # shape of predictions: (batch_size, 11, 64, 64, 3)

