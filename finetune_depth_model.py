import argparse
import tqdm
import torch
import torch.nn.functional as F

import midas
from fitvid.fitvid_torch import dict_to_cuda, normalize_depth
from fitvid.robomimic_data import load_dataset_robomimic_torch
from fitvid.depth_utils import normalize_depth, depth_to_rgb_im, save_moviepy_gif, DEFAULT_WEIGHT_LOCATIONS

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
                                     non_negative=True, blocks={'expand': True})
    return depth_model


def get_dataloaders(dataset_files, bs):
    train_load = load_dataset_robomimic_torch(dataset_files, batch_size=bs, video_len=12, phase='train', depth=True)
    val_load = load_dataset_robomimic_torch(dataset_files, batch_size=bs, video_len=12, phase='valid', depth=True)
    return train_load, val_load


def flatten_dims(img):
    shape = img.shape
    final_shape = [-1] + list(shape[-3:])
    img = torch.reshape(img, final_shape)
    return img


def loss_fn(pred, actual):
    pred = normalize_depth(pred, 1)
    actual = normalize_depth(actual, 1)
    return F.mse_loss(pred, actual)


def prep_batch(batch):
    depth_images = flatten_dims(batch['depth_video'])
    images = flatten_dims(batch['video'])
    images = torch.nn.Upsample(scale_factor=4)(images)
    images = (images - torch.Tensor([0.485, 0.456, 0.406])[..., None, None].to(images.device)) / (torch.Tensor([0.229, 0.224, 0.225])[..., None, None].to(images.device))
    return images, depth_images


def log_preds(preds, epoch, phase):
    preds = preds.detach().cpu().numpy()
    for i, pred in enumerate(preds):
        depth_video = depth_to_rgb_im(pred)
        save_moviepy_gif(list(depth_video), f'finetune_outputs/{phase}_epoch_{epoch}_pred_{i}')


def main(args):
    model = load_model(args.model_type, args.checkpoint)
    model = model.cuda()
    train_loader, val_loader = get_dataloaders(args.dataset_files, args.batch_size)
    train_loader, train_prep = train_loader
    val_loader, val_prep = val_loader
    print(len(train_loader))

    train_steps_per_epoch = 300
    val_steps_per_epoch = 300

    optimizer = torch.optim.Adam(model.parameters())

    for epoch in range(args.epochs):
        model.train()
        for i, batch in tqdm.tqdm(enumerate(train_loader)):
            batch = dict_to_cuda(train_prep(batch))
            traj_length = batch['video'].shape[1]
            images, depth_images = prep_batch(batch)
            preds = model(images)
            preds = torch.nn.functional.interpolate(preds[:, None], size=(64, 64))
            preds = 1 - normalize_depth(preds)
            optimizer.zero_grad()
            loss = loss_fn(preds, depth_images)
            loss.backward()
            optimizer.step()
            if i % 100 == 0:
                print(f'Train loss: {loss}')
            if i > train_steps_per_epoch:
                break
        preds = preds.reshape(-1, traj_length, *preds.shape[1:])
        log_preds(preds, epoch, 'train')
        print(f'Epoch {epoch} training loss: {loss}')
        model.eval()
        print('Running validation...')
        for i, batch in enumerate(val_loader):
            batch = dict_to_cuda(val_prep(batch))
            images, depth_images = prep_batch(batch)
            preds = model(images)
            preds = torch.nn.functional.interpolate(preds[:, None], size=(64, 64))
            preds = 1 - normalize_depth(preds)
            val_loss = loss_fn(preds, depth_images)
            if i > val_steps_per_epoch:
                break
        print(f'Epoch {epoch} validation loss: {val_loss}')
        preds = preds.reshape(-1, traj_length, *preds.shape[1:])
        log_preds(preds, epoch, 'val')

    torch.save(model.state_dict(), args.output_file)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Finetune MiDaS depth model.")
    parser.add_argument(
        '--checkpoint', default='', help='Model checkpoint to load')
    parser.add_argument(
        '--output_file', default='', required=True, help='Where to save final model params')
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