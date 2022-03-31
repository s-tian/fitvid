# import jax
import torch
import numpy as np

from robomimic.utils.dataset import SequenceDataset
import robomimic.utils.obs_utils as ObsUtils
from torch.utils.data import DataLoader, ConcatDataset
from torchvision.transforms import Resize


def get_image_name(cam):
    if not cam:
        return 'image'  # for the real world datasets
    else:
        return f'{cam}_image'


def get_data_loader(dataset_paths, batch_size, video_len, video_dims, phase, depth, view, cache_mode='lowdim', seg=True, only_depth=False):
    """
    Get a data loader to sample batches of data.
    """
    imageview_name = get_image_name(view)

    ObsUtils.initialize_obs_utils_with_obs_specs({
        "obs": {
            "rgb": [imageview_name],
            "depth": [f"{view}_depth"],
            #"scan": [f"{view}_segmentation_instance"]
            "scan": [f"{view}_seg"]
        }
    })

    all_datasets = []
    for i, dataset_path in enumerate(dataset_paths):
        #obs_keys = (f"{view}_image", f"{view}_segmentation_instance")
        obs_keys = tuple()
        if not only_depth:
            obs_keys = obs_keys + (imageview_name, )
        if depth or only_depth:
            obs_keys = obs_keys + (f"{view}_depth",)
        if seg:
            obs_keys = obs_keys + (f"{view}_seg",)

        dataset = SequenceDataset(
            hdf5_path=dataset_path,
            obs_keys=obs_keys,                      # observations we want to appear in batches
            dataset_keys=(                  # can optionally specify more keys here if they should appear in batches
                "actions", 
                "rewards", 
                "dones",
            ),
            load_next_obs=False,
            frame_stack=1,
            seq_length=video_len,                  # length-10 temporal sequences
            pad_frame_stack=True,
            pad_seq_length=False,            # pad last obs per trajectory to ensure all sequences are sampled
            get_pad_mask=False,
            goal_mode=None,
            hdf5_cache_mode=cache_mode,          # cache dataset in memory to avoid repeated file i/o
            hdf5_use_swmr=True,
            hdf5_normalize_obs=False,
            filter_by_attribute=phase,       # filter either train or validation data
            image_size=video_dims,
        )
        all_datasets.append(dataset)
        print(f"\n============= Created Dataset {i+1} out of {len(dataset_paths)} =============")
        print(dataset)
        print("")
    dataset = ConcatDataset(all_datasets)
    data_loader = DataLoader(
        dataset=dataset,
        sampler=None,       # no custom sampling logic (uniform sampling)
        batch_size=batch_size,     
        shuffle=True,
        num_workers=0,
        drop_last=True,      # don't provide last batch in dataset pass if it's less than 100 in size
    )
    return data_loader


def load_dataset_robomimic_torch(dataset_path, batch_size, video_len, video_dims, phase, depth, view='agentview', cache_mode='low_dim', seg=True, only_depth=False):
    assert phase in ['train', 'valid'], f'Phase is not one of the acceptable values! Got {phase}'

    loader = get_data_loader(dataset_path, batch_size, video_len, video_dims, phase, depth, view, cache_mode, seg, only_depth)

    def prepare_data(xs):
        if only_depth:
            # take depth video as the actual video
            data_dict = {
                'video': xs['obs'][f'{view}_depth'],
                'actions': xs['actions'],
            }
        else:
            data_dict = {
                'video': xs['obs'][get_image_name(view)],
                'actions': xs['actions'],
            }
        if f'{view}_seg' in xs['obs']:
            data_dict['segmentation'] = xs['obs'][f'{view}_seg']
            # zero out the parts of the segmentation which are not assigned label corresponding to object of interest
            # set the object label components to 1
            object_seg_index = 0 # Seg index is 0 on the iGibson data, and 1 on Mujoco data
            arm_seg_index = 1 # Seg index is 0 on the iGibson data, and 1 on Mujoco data
            seg_image = torch.zeros_like(data_dict['segmentation'])
            seg_image[data_dict['segmentation'] == object_seg_index] = 1
            seg_image[data_dict['segmentation'] == arm_seg_index] = 2
            not_either_mask = ~((data_dict['segmentation'] == object_seg_index) | (data_dict['segmentation'] == arm_seg_index))
            seg_image[not_either_mask] = 0
            data_dict['segmentation'] = seg_image
        else:
            data_dict['segmentation'] = None
        if depth and not only_depth:
            data_dict['depth_video'] = xs['obs'][f'{view}_depth']
        return data_dict

    return loader, prepare_data


def load_dataset_robomimic(dataset_path, batch_size, video_len, is_train, depth, view='agentview'):
    if is_train:
        phase = 'train'
    else:
        phase = 'valid'

    loader = get_data_loader(dataset_path, batch_size, video_len, phase, depth)
    local_device_count = jax.local_device_count()
    
    def prepare_data(xs):
        data_dict = {
            'video': torch.permute(xs['obs'][f'{view}_image'], (0, 1, 3, 4, 2)),
            'actions': xs['actions']
        }
        if depth:
            data_dict['depth_video'] = torch.permute(xs['obs'][f'{view}_depth'], (0, 1, 3, 4, 2))

        def _prepare(x):
            x = x.numpy()
            return jax.device_put(x.reshape((local_device_count, -1) + x.shape[1:]))
        prepared = jax.tree_map(_prepare, data_dict)

        return prepared
    
    def cycle(it):
        while True:
            for x in it:
                yield x
    iterable = cycle(loader)
    iterable = map(prepare_data, iterable)
    return iterable


if __name__ == '__main__':
    dataset_path = '/viscam/u/stian/perceptual-metrics/robomimic/datasets/lift/mg/image_and_depth.hdf5'
    dataset_path = '/viscam/u/stian/perceptual-metrics/robosuite/robosuite/models/assets/demonstrations/1644373519_3708425/1644373519_3708425_igibson_obs.hdf5'
    dataset_path = '/viscam/u/stian/perceptual-metrics/robosuite/robosuite/models/assets/policy_rollouts/pushcenter_osc_position_eval/igibson_obs.hdf5'

    dl, p = load_dataset_robomimic_torch([dataset_path], 16, 10, 'train', depth=False, view='agentview_shift_2')
    batch = next(iter(dl))
    p(batch)

    #dataloader = get_data_loader(dataset_path, 1, 10, 'train')
    #batch = next(iter(dataloader))

