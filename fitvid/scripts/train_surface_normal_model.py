import argparse
import os
import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

import midas
from fitvid.utils.utils import dict_to_cuda
from fitvid.utils.vis_utils import save_moviepy_gif
from fitvid.utils.pytorch_metrics import tv
from fitvid.data.robomimic_data import load_dataset_robomimic_torch
from fitvid.utils.depth_utils import normalize_depth, depth_to_rgb_im, DEFAULT_WEIGHT_LOCATIONS
from perceptual_metrics.mpc.utils import save_torch_img

class ResidualBlock(nn.Module):

    def __init__(self, in_channels):
        super().__init__()
        self.cnv = nn.Sequential(
            nn.Conv2d(
                in_channels=32,
                out_channels=32,
                kernel_size=3,
                stride=1,
                padding=1,
            ),
            nn.ReLU(),
            nn.Conv2d(
                in_channels=32,
                out_channels=32,
                kernel_size=3,
                stride=1,
                padding=1,
            ),
        )

    def forward(self, x):
        return x + self.cnv(x)


class ConvPredictor(nn.Module):

    def __init__(self, nonlinearity):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(
                in_channels=3,
                out_channels=32,
                kernel_size=3,
                stride=1,
                padding=1,
            ),
            ResidualBlock(in_channels=32),
            nn.ReLU(),
            ResidualBlock(in_channels=32),
            nn.ReLU(),
            ResidualBlock(in_channels=32),
            nn.ReLU(),
            nn.Conv2d(
                in_channels=32,
                out_channels=3,
                kernel_size=3,
                stride=1,
                padding=1,
            ),
            nn.Tanh()
        )

    def forward(self, x):
        return self.net(x)


def load_model(model_name, path):
    # assert model_name in DEFAULT_WEIGHT_LOCATIONS.keys(), f"Model name was {model_name} but must be in {list(DEFAULT_LOCATIONS.keys())}"
    if model_name == 'two_conv_layers_relu':
        depth_model = ConvPredictor(nonlinearity='relu')
    elif model_name == 'two_conv_layers_gelu':
        depth_model = ConvPredictor(nonlinearity='gelu')
    else:
        raise NotImplementedError
    return depth_model


def get_dataloaders(dataset_files, bs, dims, view):
    train_load = load_dataset_robomimic_torch(dataset_files, batch_size=bs, video_len=10, video_dims=dims, phase='train',
                                              depth=False, normal=True, view=view, cache_mode='all_nogetitem')
    val_load = load_dataset_robomimic_torch(dataset_files, batch_size=bs, video_len=10, video_dims=dims, phase='valid',
                                            depth=False, normal=True, view=view, cache_mode='all_nogetitem')
    return train_load, val_load


def flatten_dims(img):
    shape = img.shape
    final_shape = [-1] + list(shape[-3:])
    img = torch.reshape(img, final_shape)
    return img


def loss_fn(pred, actual):
    # pred = normalize_depth(pred, 1)
    # actual = normalize_depth(actual, 1)
    return F.mse_loss(pred, actual)


def prep_batch(batch, upsample_factor):
    normals = batch['normal']
    images = flatten_dims(batch['video'])
    return images, normals


def log_preds(folder, rgb_images, true_images, preds, epoch, phase):
    preds = preds.detach().cpu().numpy()
    preds = np.transpose(preds, (0, 1, 3, 4, 2)) * 255
    true_images = true_images.detach().cpu().numpy()
    true_images = np.transpose(true_images, (0, 1, 3, 4, 2)) * 255
    rgb_images = rgb_images.detach().cpu().numpy()
    rgb_images = np.transpose(rgb_images, (0, 1, 3, 4, 2)) * 255
    for i, (rgb_image, pred, timg) in enumerate(zip(rgb_images, preds, true_images)):
        if i > 10:
            continue
        # depth_video = depth_to_rgb_im(pred)
        # true_image = depth_to_rgb_im(timg)
        video = np.concatenate([pred, timg, rgb_image], axis=-2)
        save_moviepy_gif(list(video), os.path.join(folder, f'{phase}_epoch_{epoch}_pred_{i}'))


def main(args):
    model = load_model(args.model_type, args.checkpoint)
    model = model.cuda()
    train_loader, val_loader = get_dataloaders(args.dataset_files, args.batch_size, (args.image_size, args.image_size), args.view)
    train_loader, train_prep = train_loader
    val_loader, val_prep = val_loader
    # loss_fn = scale_invariant_loss
    print(f'Train loader has length {len(train_loader)}')

    train_steps_per_epoch = 300
    val_steps_per_epoch = 24

    optimizer = torch.optim.Adam(model.parameters())

    output_folder = os.path.dirname(args.output_file)
    if not os.path.exists(output_folder):
        print(f'Creating output folder {output_folder}')
        os.makedirs(output_folder)

    for epoch in range(args.epochs):
        model.eval()
        print('Running validation...')
        for i, batch in enumerate(val_loader):
            # save_torch_img(batch['obs']['agentview_shift_2_image'][0][0], 'test_image')
            batch = dict_to_cuda(val_prep(batch))
            traj_length = batch['video'].shape[1]
            images, normal_images = prep_batch(batch, args.upsample_factor)
            with torch.no_grad():
                preds = model(images)
                if args.upsample_factor != 1:
                    # preds = torch.nn.functional.interpolate(preds[:, None], size=(64, 64))
                    preds = torch.nn.functional.interpolate(preds[:, None], size=(64, 64), mode='bilinear')
                preds = preds.reshape(-1, traj_length, *preds.shape[1:])
                # preds = 1.0 / (preds + 1e-10) + 0.5
                # preds = 1 - preds
            val_loss = loss_fn(preds, normal_images)
            if i > val_steps_per_epoch:
                break
        print(f'Epoch {epoch} validation loss: {val_loss}')
        log_preds(output_folder, batch['video'], normal_images, preds, epoch, 'val')

        model.train()
        for i, batch in tqdm.tqdm(enumerate(train_loader)):
            # save_torch_img(batch['obs']['agentview_shift_2_image'][0][0], 'test_image')
            batch = dict_to_cuda(train_prep(batch))
            shape = batch['video'].shape
            traj_length = shape[1]
            images, normal_images = prep_batch(batch, args.upsample_factor)
            preds = model(images)
            if args.upsample_factor != 1:
                # preds = torch.nn.functional.interpolate(preds[:, None], size=(64, 64))
                preds = torch.nn.functional.interpolate(preds[:, None], size=(64, 64), mode='bilinear')
            preds = preds.reshape(-1, traj_length, *preds.shape[1:])
            # preds = 1 - preds
            # preds = 1.0 / (preds + 1e-10) + 0.5
            optimizer.zero_grad()
            loss = loss_fn(preds, normal_images)
            loss.backward()
            torch.nn.utils.clip_grad_norm(model.parameters(), 1)
            optimizer.step()
            if i % 100 == 0:
                print(f'Train loss: {loss}')
            if i > train_steps_per_epoch:
                break
        log_preds(output_folder, batch['video'], normal_images, preds, epoch, 'train')
        print(f'Epoch {epoch} training loss: {loss}')
        torch.save(model.state_dict(), args.output_file)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Finetune MiDaS depth model.")
    parser.add_argument(
        '--checkpoint', default='', help='Model checkpoint to load')
    parser.add_argument(
        '--output_file', default='', required=True, help='Where to save final model params')
    parser.add_argument(
        '--view', default='agentview', required=True, help='Camera view to use for training')
    parser.add_argument(
        '--upsample_factor', default=1, type=int, help='factor to upsample images')
    parser.add_argument(
        '--image_size', default=64, help='image dimension')
    parser.add_argument(
        '--dataset_files', nargs='+', required=True, help='number of trajectories to run for complete eval')
    parser.add_argument(
        '--model_type', default='', required=True, help='which MiDaS model to use')
    parser.add_argument(
        '--epochs', type=int, default=10, help='number of finetuning epochs')
    parser.add_argument(
        '--batch_size', type=int, default=32, help='batchsize')
    args = parser.parse_args()
    main(args)