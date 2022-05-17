import numpy as np
import matplotlib.pyplot as plt
import torch.nn.functional as F
import torch

from kornia.filters import sobel

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
    if len(im.shape) >= 3 and im.shape[-3] == 1:
        im = np.squeeze(im, axis=-3)
    if im.shape[-1] == 1:
        im = np.squeeze(im, axis=-1)
    # convert to rgb using given cmap
    im = cmap(im)[..., :3]
    return (im * 255).astype(np.uint8)


def sobel_loss(t1, t2, reduce_batch=True):
    original_shape = t1.shape
    compress_shape = (-1,) + tuple(t1.shape[-3:])  # change to (B, C, H, W)
    t1, t2 = torch.reshape(t1, compress_shape), torch.reshape(t2, compress_shape)
    t1_sobel, t2_sobel = sobel(t1), sobel(t2)
    t1_sobel, t2_sobel = torch.reshape(t1_sobel, original_shape), torch.reshape(t2_sobel, original_shape)
    return mse_loss(t1_sobel, t2_sobel, reduce_batch=reduce_batch)


def mse_loss(t1, t2, reduce_batch=True, mask=None):
    mse = ((t1 - t2) ** 2)
    if mask is not None:
        mse = mask * mse
    if reduce_batch:
        return mse.mean()
    else:
        reduce_dims = tuple(range(len(t1.shape)))
        return mse.mean(dim=reduce_dims[1:])


def test_mse_loss():
    t1, t2 = torch.randn((30, 20, 10)), torch.randn((30, 20, 10))
    my_mse = mse_loss(t1, t2, reduce_batch=True)
    torch_mse = torch.nn.functional.mse_loss(t1, t2)
    print(f'Mine: {my_mse}, nn.Functional: {torch_mse}')
    assert torch.allclose(my_mse, torch_mse), "Reducing all dimensions doesn't match"
    per_sample_mse = mse_loss(t1, t2, reduce_batch=False)
    assert torch.allclose(torch_mse, per_sample_mse.mean()), 'Not reducing yields different results'
    print(f'Mine: {per_sample_mse.mean()}, nn.Functional: {torch_mse}')


if __name__ == '__main__':
    test_mse_loss()