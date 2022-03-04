import numpy as np
import matplotlib.pyplot as plt
import torch.nn.functional as F

DEFAULT_WEIGHT_LOCATIONS = {
    'dpt': '/viscam/u/stian/perceptual-metrics/MiDaS/weights/dpt_hybrid-midas-501f0c75.pt',
    'mn': '/viscam/u/stian/perceptual-metrics/MiDaS/weights/midas_v21-f6b98070.pt',
    'mns': '/viscam/u/stian/perceptual-metrics/MiDaS/weights/midas_v21_small-70d6b9c8.pt'
}


def normalize_depth(t, across_dims=1):
    # Normalize depth values across all but the first dimension.
    # A 4-D input will be normalized per frame, but a 5-D input will be normalized per sequence.
    dims = tuple(range(len(t.shape))[across_dims:])
    t = t - t.amin(dim=dims, keepdim=True)
    t = t / (t.amax(dim=dims, keepdim=True) + 1e-10)
    return t


def normalize_depth_npy(t, across_dims=1):
    # Normalize depth values across all but the first dimension.
    # A 4-D input will be normalized per frame, but a 5-D input will be normalized per sequence.
    dims = tuple(range(len(t.shape))[across_dims:])
    t = t - np.amin(t, axis=dims, keepdims=True)
    t = t / (np.amax(t, axis=dims, keepdims=True) + 1e-10)
    return t


def depth_to_rgb_im(im, cmap=plt.get_cmap('jet_r')):
    # shape = [(T), 1, W, H]
    im = np.squeeze(im)
    # convert to rgb using given cmap
    im = cmap(im)[..., :3]
    return (im * 255).astype(np.uint8)


def depth_mse_loss(pred, gt, reduction='mean'):
    return F.mse_loss(normalize_depth(pred, across_dims=1), normalize_depth(gt, across_dims=1), reduction=reduction)


def save_moviepy_gif(obs_list, name, fps=5):
    from moviepy.editor import ImageSequenceClip
    clip = ImageSequenceClip(obs_list, fps=fps)
    clip.write_gif(f'{name}.gif', fps=fps)


def dict_to_cuda(d):
    return {k: v.cuda() for k, v in d.items()}
