import argparse
import torch
import torch.nn.functional as F

import midas
from fitvid.fitvid_torch import dict_to_cuda, normalize_depth
from fitvid.robomimic_data import load_dataset_robomimic_torch

def load_model(model_name, path):
    default_locations = {
        'dpt': '/viscam/u/stian/perceptual-metrics/MiDaS/weights/dpt_hybrid-midas-501f0c75.pt',
        'mn' : '/viscam/u/stian/perceptual-metrics/MiDaS/weights/midas_v21-f6b98070.pt',
        'mns': '/viscam/u/stian/perceptual-metrics/MiDaS/weights/midas_v21_small-70d6b9c8.pt'
    }
    assert model_name in default_locations.keys(), f"Model name was {model_name} but must be in {list(default_locations.keys())}"
    checkpoint = path if path else default_locations['model_name']
    if model == 'dpt':
        depth_model = DPTDepthModel(
            path=checkpoint,
            backbone="vitb_rn50_384",
            non_negative=True,
        )
    elif model == 'mn':
        depth_model = MidasNet(checkpoint,
                               non_negative=True)
    elif model == 'mns':
        depth_model = MidasNet_small(checkpoint,
                                     features=64, backbone="efficientnet_lite3", exportable=True,
                                     non_negative=True, blocks={'expand': True})
    return depth_model


def get_dataloaders(dataset_files, bs, ):
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
    return images, depth_images

def main(args):
    model = load_model(args.model_type, args.checkpoint)
    train_loader, val_loader = get_dataloaders(args.dataset_files)
    train_loader, train_prep = train_loader
    val_loader, val_prep = val_loader

    optimizer = torch.optim.Adam(model.parameters())

    for epoch in range(args.num_epochs):
        for batch in train_loader:
            batch = dict_to_cuda(train_prep(batch))
            images, depth_images = prep_batch(batch)
            preds = model(images)
            preds = torch.nn.functional.interpolate(preds, size=(64, 64))
            optimizer.zero_grad()
            loss = loss_fn(preds, depth_images)
            loss.backward()
            optimizer.step()
        print(f'Epoch {epoch} training loss: {loss}')
        for batch in val_loader:
            batch = dict_to_cuda(val_prep(batch))
            images, depth_images = prep_batch(batch)
            preds = model(images)
            preds = torch.nn.functional.interpolate(preds, size=(64, 64))
            val_loss = loss_fn(preds, depth_images)
        print(f'Epoch {epoch} validation loss: {val_loss}')


if __name__ == '__main__':
    args = argparse.ArgumentParser(description="Finetune MiDaS depth model.")
    parser.add_argument(
        '--checkpoint', default='', help='Model checkpoint to load')
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