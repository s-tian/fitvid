import argparse
import os
import tqdm
import torch
import torch.nn.functional as F
import numpy as np

import midas
from fitvid.fitvid_torch import dict_to_cuda, normalize_depth
from fitvid.robomimic_data import load_dataset_robomimic_torch
from fitvid.depth_utils import normalize_depth, depth_to_rgb_im, save_moviepy_gif, DEFAULT_WEIGHT_LOCATIONS
from perceptual_metrics.utils import save_torch_img


def load_model(model_name, path):
    assert model_name in DEFAULT_WEIGHT_LOCATIONS.keys(), f"Model name was {model_name} but must be in {list(default_locations.keys())}"
    checkpoint = path if path else DEFAULT_WEIGHT_LOCATIONS[model_name]
    if model_name == 'dpt':
        from midas.dpt_depth import DPTDepthModel
        depth_model = DPTDepthModel(
            path=checkpoint,
            backbone="vitb_rn50_384",
            non_negative=True,
        )
    elif model_name == 'mn':
        from midas.midas_net import MidasNet
        depth_model = MidasNet(checkpoint,
                               non_negative=True)
    elif model_name == 'mns':
        from midas.midas_net_custom import MidasNet_small
        depth_model = MidasNet_small(checkpoint,
                                     features=64, backbone="efficientnet_lite3", exportable=True,
                                     non_negative=False, blocks={'expand': True})
    return depth_model


def get_dataloaders(dataset_files, bs, dims, view):
    train_load = load_dataset_robomimic_torch(dataset_files, batch_size=bs, video_len=10, video_dims=dims, phase='train', depth=True, view=view)
    val_load = load_dataset_robomimic_torch(dataset_files, batch_size=bs, video_len=10, video_dims=dims, phase='valid', depth=True, view=view)
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


def scale_invariant_loss(pred, actual):
    mses = []
    for p, gt in zip(pred, actual):
        # invert ground truth
        actual = 1.0 / (actual + 1e-10)
        # align prediction based on least squares criterion
        total_size = np.prod(p.shape)
        import ipdb; ipdb.set_trace()
        inv_term = torch.tensor(
                [[ (p**2).sum(), p.sum()],
                 [  p.sum(),     total_size]]
            ).to(p)
        inv_term = torch.inverse(inv_term)
        least_squares = torch.mm(inv_term, torch.tensor([p*gt.sum(), gt.sum()]).to(p))

        p_aligned = p * least_squares[0] + least_squares[1]
        # invert aligned pred to get depth
        p_aligned_depth = 1.0 / (p_aligned + 1e-10)
        mse = F.mse_loss(pred, actual)
        mses.append(mse)
    return torch.stack(mse).mean()


def prep_batch(batch, upsample_factor):
    depth_images = batch['depth_video'][:, :, None]
    images = flatten_dims(batch['video'])
    if upsample_factor != 1:
        images = torch.nn.Upsample(scale_factor=upsample_factor)(images)
    images = (images - torch.Tensor([0.485, 0.456, 0.406])[..., None, None].to(images.device)) / (torch.Tensor([0.229, 0.224, 0.225])[..., None, None].to(images.device))
    assert 0 <= depth_images.min() < depth_images.max() <= 1, 'Depth image found which was OOB!'
    return images, depth_images


def log_preds(folder, rgb_images, true_images, preds, epoch, phase):
    preds = preds.detach().cpu().numpy()
    true_images = true_images.detach().cpu().numpy()
    rgb_images = rgb_images.detach().cpu().numpy()
    rgb_images = np.transpose(rgb_images, (0, 1, 3, 4, 2)) * 255
    for i, (rgb_image, pred, timg) in enumerate(zip(rgb_images, preds, true_images)):
        if i > 10:
            continue
        depth_video = depth_to_rgb_im(pred)
        true_image = depth_to_rgb_im(timg)
        video = np.concatenate([depth_video, true_image, rgb_image], axis=-2)
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
            images, depth_images = prep_batch(batch, args.upsample_factor)
            with torch.no_grad():
                preds = model(images)
                if args.upsample_factor != 1:
                    preds = torch.nn.functional.interpolate(preds[:, None], size=(64, 64))
                preds = preds.reshape(-1, traj_length, *preds.shape[1:])
                # preds = 1.0 / (preds + 1e-10) + 0.5
                # preds = 1 - preds
            val_loss = loss_fn(preds, depth_images)
            if i > val_steps_per_epoch:
                break
        print(f'Epoch {epoch} validation loss: {val_loss}')
        log_preds(output_folder, batch['video'], depth_images, preds, epoch, 'val')

        model.train()
        for i, batch in tqdm.tqdm(enumerate(train_loader)):
            # save_torch_img(batch['obs']['agentview_shift_2_image'][0][0], 'test_image')
            batch = dict_to_cuda(train_prep(batch))
            shape = batch['video'].shape
            traj_length = shape[1]
            images, depth_images = prep_batch(batch, args.upsample_factor)
            preds = model(images)
            if args.upsample_factor != 1:
                preds = torch.nn.functional.interpolate(preds[:, None], size=(64, 64))
            preds = preds.reshape(-1, traj_length, *preds.shape[1:])
            # preds = 1 - preds
            # preds = 1.0 / (preds + 1e-10) + 0.5
            optimizer.zero_grad()
            loss = loss_fn(preds, depth_images)
            loss.backward()
            torch.nn.utils.clip_grad_norm(model.parameters(), 1)
            optimizer.step()
            if i % 100 == 0:
                print(f'Train loss: {loss}')
            if i > train_steps_per_epoch:
                break
        log_preds(output_folder, batch['video'], depth_images, preds, epoch, 'train')
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
        '--upsample_factor', default=4, help='factor to upsample images')
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