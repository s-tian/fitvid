import numpy as np
import torch
import cv2
import matplotlib.pyplot as plt
from fitvid.utils.depth_utils import depth_to_rgb_im


def build_visualization(gt, pred, gt_depth=None, gt_depth_pred=None, pred_depth=None, rgb_loss=None, depth_loss_weight=None, depth_loss=None):
    tlen = gt.shape[1]

    if gt_depth is None: # for depth only or no depth prediction case
        cmap = plt.get_cmap('jet_r')
        gt = np.moveaxis(depth_to_rgb_im(gt.detach().cpu().numpy(), cmap), 4, 2)
        pred = np.moveaxis(depth_to_rgb_im(pred.detach().cpu().numpy(), cmap), 4, 2)

        rgb_loss_im = generate_sample_metric_imgs(rgb_loss.detach().cpu().numpy(), tlen)
        image_rows = [gt, pred, rgb_loss_im]
        image_rows = np.concatenate(image_rows, axis=-1)  # create a horizontal row
        image_rows = np.concatenate(image_rows, axis=-2)  # create B rows
        headers = ['GT', 'Pred', 'RGB Loss']
        text_headers = [generate_text_square(h) for h in headers]
        text_headers = np.concatenate(text_headers, axis=-1)
        text_headers = np.tile(text_headers[None], (tlen, 1, 1, 1))
        image_rows = np.concatenate((text_headers, image_rows), axis=-2)
        return image_rows

    else:
        # convert depth images to RGB visualizations
        cmap = plt.get_cmap('jet_r')
        gt_depth = np.moveaxis(depth_to_rgb_im(gt_depth.detach().cpu().numpy(), cmap), 4, 2)
        gt_depth_pred = np.moveaxis(depth_to_rgb_im(gt_depth_pred.detach().cpu().numpy(), cmap), 4, 2)
        pred_depth = np.moveaxis(depth_to_rgb_im(pred_depth.detach().cpu().numpy(), cmap), 4, 2)

        # create images for numerical values
        total_loss = depth_loss_weight * depth_loss + (1 - depth_loss_weight) * rgb_loss
        rgb_loss_im = generate_sample_metric_imgs(rgb_loss.detach().cpu().numpy(), tlen)
        depth_loss_im = generate_sample_metric_imgs(depth_loss.detach().cpu().numpy(), tlen)
        total_loss_im = generate_sample_metric_imgs(total_loss.detach().cpu().numpy(), tlen)

        # each shape is [B, T, 3, H, W]
        image_rows = [gt.detach().cpu().numpy() * 255, pred.detach().cpu().numpy() * 255, pred_depth, gt_depth, gt_depth_pred,
                      rgb_loss_im, depth_loss_im, total_loss_im]
        image_rows = np.concatenate(image_rows, axis=-1) # create a horizontal row
        image_rows = np.concatenate(image_rows, axis=-2) # create B rows

        headers = ['GT', 'Pred', 'Pred Depth', 'GT Depth', 'GT D Pred', 'RGB Loss', 'D Loss', 'Total L']
        text_headers = [generate_text_square(h) for h in headers]
        text_headers = np.concatenate(text_headers, axis=-1)
        # duplicate text headers through time
        text_headers = np.tile(text_headers[None], (tlen, 1, 1, 1))
        image_rows = np.concatenate((text_headers, image_rows), axis=-2)
        return image_rows


def generate_sample_metric_imgs(l, tlen, size=(64, 64)):
    # output should be in shape [B, T, 3, *size]
    squares = []
    for num in l:
        text = '{:.3e}'.format(num)  # 3 decimal points in scientific notation
        squares.append(generate_text_square(text, size))
    squares = np.stack(squares)
    squares = np.tile(squares[:, None], (1, tlen, 1, 1, 1))
    return squares


def generate_text_square(text, size=(64, 64), fontscale=2.5):
    img = np.ones(shape=(512, 512, 3), dtype=np.int16)
    cv2.putText(img=img, text=text, org=(50, 250), fontFace=cv2.FONT_HERSHEY_TRIPLEX, fontScale=fontscale,
                color=(255, 255, 255), thickness=3)
    img = cv2.resize(img, size, interpolation=cv2.INTER_AREA)
    img = np.moveaxis(img, 2, 0)
    return img


def save_moviepy_gif(obs_list, name, fps=5):
    from moviepy.editor import ImageSequenceClip
    clip = ImageSequenceClip(obs_list, fps=fps)
    clip.write_gif(f'{name}.gif', fps=fps)