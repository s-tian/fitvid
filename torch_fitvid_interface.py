import torch
import numpy as np
from fitvid.fitvid_torch import FitVid
import datetime

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
        'stochastic': False,
        'loss_fn': 'l2'
    }

    def __init__(self, checkpoint_dir, hp_override):
        # TODO import hyperparameters nicely
        self.hp = self.DEFAULT_HP
        self.hp.update(hp_override)
        self.model = FitVid(**self.hp)
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

    checkpoint_dir = '/home/rjahmed/Fitvid_Ferro/fitvid-ferrofluid/checkpoints/PyTorch model checkpoints/3Hz_larger_stochastic/model_epoch230.pt'
    hp_override = {
        'stochastic': True,
        'rnn_size': 128,
        'g_dim':64,
        'has_depth_predictor': False
        #'num_base_filters':4
    }
    model = FitVidTorchModel(checkpoint_dir, hp_override)
    #model is loaded after this line

    startime = datetime.datetime.now()
    for i in range(0,100):

        batch_size = 200
        batch = {
            'video': np.zeros((batch_size, 2, 64, 64, 3)),
            'actions':  np.zeros((batch_size, 11, 8)),
        }
        predictions = model(batch)
        # shape of predictions: (batch_size, 11, 64, 64, 3)
    endtime = datetime.datetime.now()
    runtime = endtime-startime
    ave_run_time = runtime/100
    print(ave_run_time)

    print(predictions.shape)


