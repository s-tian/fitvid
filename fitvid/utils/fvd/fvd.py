import sys
import os
import math
import torch.nn.functional as F
import numpy as np
import torch

from fitvid.utils.fvd.download_i3d import load_i3d_pretrained

i3d = None


def get_fvd_logits(videos):
    global i3d
    if i3d is None:
        num_devices = torch.cuda.device_count()
        devices = [f"cuda:{i}" for i in range(num_devices)]
        i3d = {device: load_i3d_pretrained(device) for device in devices}

    videos = preprocess(videos)
    # B, C, T, H, W
    chunk_size = min(videos.shape[0], 16)
    with torch.no_grad():
        logits = []
        i3d_model = i3d[f"{videos.device.type}:{videos.device.index}"]
        for i in range(0, videos.shape[0], chunk_size):
            batch = videos[i : i + chunk_size]
            logits.append(i3d_model(batch))
        logits = torch.cat(logits, dim=0)
    return logits


def preprocess_single(video, resolution, sequence_length=None):
    # video: THWC, {0, ..., 255}
    video = video.permute(0, 3, 1, 2).float() / 255.0  # TCHW
    t, c, h, w = video.shape

    # temporal crop
    if sequence_length is not None:
        assert sequence_length <= t
        video = video[:sequence_length]

    # scale shorter side to resolution
    scale = resolution / min(h, w)
    if h < w:
        target_size = (resolution, math.ceil(w * scale))
    else:
        target_size = (math.ceil(h * scale), resolution)
    video = F.interpolate(video, size=target_size, mode="bilinear", align_corners=False)

    # center crop
    t, c, h, w = video.shape
    w_start = (w - resolution) // 2
    h_start = (h - resolution) // 2
    video = video[:, :, h_start : h_start + resolution, w_start : w_start + resolution]
    video = video.permute(1, 0, 2, 3).contiguous()  # CTHW

    video -= 0.5

    return video


def preprocess(videos, target_resolution=224):
    # videos in {0, ..., 255} as np.uint8 array
    b, t, c, h, w = videos.shape
    if isinstance(videos, np.ndarray):
        videos = torch.from_numpy(videos)
    assert isinstance(
        videos, torch.Tensor
    ), "Videos should either be np.ndarray or torch.Tensor!"
    videos = torch.permute(videos, (0, 1, 3, 4, 2))
    videos = torch.stack(
        [preprocess_single(video, target_resolution) for video in videos]
    )
    return videos * 2  # [-0.5, 0.5] -> [-1, 1]


# https://github.com/tensorflow/gan/blob/de4b8da3853058ea380a6152bd3bd454013bf619/tensorflow_gan/python/eval/classifier_metrics.py#L161
def _symmetric_matrix_square_root(mat, eps=1e-10):
    u, s, v = torch.svd(mat)
    si = torch.where(s < eps, s, torch.sqrt(s))
    return torch.matmul(torch.matmul(u, torch.diag(si)), v.t())


# https://github.com/tensorflow/gan/blob/de4b8da3853058ea380a6152bd3bd454013bf619/tensorflow_gan/python/eval/classifier_metrics.py#L400
def trace_sqrt_product(sigma, sigma_v):
    sqrt_sigma = _symmetric_matrix_square_root(sigma)
    sqrt_a_sigmav_a = torch.matmul(sqrt_sigma, torch.matmul(sigma_v, sqrt_sigma))
    return torch.trace(_symmetric_matrix_square_root(sqrt_a_sigmav_a))


# https://discuss.pytorch.org/t/covariance-and-gradient-support/16217/2
def cov(m, rowvar=False):
    """Estimate a covariance matrix given data.

    Covariance indicates the level to which two variables vary together.
    If we examine N-dimensional samples, `X = [x_1, x_2, ... x_N]^T`,
    then the covariance matrix element `C_{ij}` is the covariance of
    `x_i` and `x_j`. The element `C_{ii}` is the variance of `x_i`.

    Args:
        m: A 1-D or 2-D array containing multiple variables and observations.
            Each row of `m` represents a variable, and each column a single
            observation of all those variables.
        rowvar: If `rowvar` is True, then each row represents a
            variable, with observations in the columns. Otherwise, the
            relationship is transposed: each column represents a variable,
            while the rows contain observations.

    Returns:
        The covariance matrix of the variables.
    """
    if m.dim() > 2:
        raise ValueError("m has more than 2 dimensions")
    if m.dim() < 2:
        m = m.view(1, -1)
    if not rowvar and m.size(0) != 1:
        m = m.t()

    fact = 1.0 / (m.size(1) - 1)  # unbiased estimate
    m -= torch.mean(m, dim=1, keepdim=True)
    mt = m.t()  # if complex: mt = m.t().conj()
    return fact * m.matmul(mt).squeeze()


def frechet_distance(x1, x2):
    x1 = x1.flatten(start_dim=1)
    x2 = x2.flatten(start_dim=1)
    m, m_w = x1.mean(dim=0), x2.mean(dim=0)
    sigma, sigma_w = cov(x1, rowvar=False), cov(x2, rowvar=False)

    sqrt_trace_component = trace_sqrt_product(sigma, sigma_w)
    trace = torch.trace(sigma + sigma_w) - 2.0 * sqrt_trace_component

    mean = torch.sum((m - m_w) ** 2)
    fd = trace + mean
    return fd


def main():
    # Number of videos must be divisible by 16.
    NUMBER_OF_VIDEOS = 16
    VIDEO_LENGTH = 15

    # https://github.com/google-research/google-research/blob/master/frechet_video_distance/example.py
    # The FVD for this setup should be around 131.

    # b, t, h, w, c
    first_set_of_videos = np.zeros(
        [NUMBER_OF_VIDEOS, VIDEO_LENGTH, 3, 64, 64], dtype=np.uint8
    )
    second_set_of_videos = (
        np.ones([NUMBER_OF_VIDEOS, VIDEO_LENGTH, 3, 64, 64], dtype=np.uint8) * 255
    )
    emb1 = get_fvd_logits(first_set_of_videos)
    emb2 = get_fvd_logits(second_set_of_videos)

    result = frechet_distance(emb1, emb2)  # 134
    print("computed:", result, "reference: 131")


if __name__ == "__main__":
    main()
