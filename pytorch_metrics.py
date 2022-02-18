import torch
import piq


def flatten_image(x):
    return x.reshape(-1, *x.shape[-3:])


def apply_function_metric(fn, vid1, vid2):
    vid1, vid2 = flatten_image(vid1), flatten_image(vid2)
    return torch.mean(fn(vid1, vid2))


def ssim(vid1, vid2):
    return apply_function_metric(piq.ssim, vid1, vid2)


def psnr(vid1, vid2):
    return apply_function_metric(piq.psnr, vid1, vid2)


def tv(vid1):
    vid1 = flatten_image(vid1)
    return torch.mean(piq.total_variation(vid1))


def lpips(lpips, vid1, vid2):
    return apply_function_metric(lambda x, y: lpips(x, y), vid1, vid2)
