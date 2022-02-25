import jax
import torch
import numpy as np

from robomimic.utils.dataset import SequenceDataset
import robomimic.utils.obs_utils as ObsUtils
from torch.utils.data import DataLoader, ConcatDataset


def get_data_loader(dataset_paths, batch_size, video_len, phase, depth, view):
    """
    Get a data loader to sample batches of data.
    """

    ObsUtils.initialize_obs_utils_with_obs_specs({
        "obs": {
            "rgb": [f"{view}_image"],
            "depth": [f"{view}_depth"],
            #"scan": [f"{view}_segmentation_instance"]
            "scan": [f"{view}_seg"]
        }
    })

    all_datasets = []
    for i, dataset_path in enumerate(dataset_paths):
        #obs_keys = (f"{view}_image", f"{view}_segmentation_instance")
        obs_keys = (f"{view}_image", f"{view}_seg")
        if depth:
            obs_keys = obs_keys + (f"{view}_depth",)

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
            hdf5_cache_mode="low_dim",          # cache dataset in memory to avoid repeated file i/o
            hdf5_use_swmr=True,
            hdf5_normalize_obs=False,
            filter_by_attribute=phase,       # filter either train or validation data
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
        num_workers=2,
        drop_last=True      # don't provide last batch in dataset pass if it's less than 100 in size
    )
    return data_loader


def load_dataset_robomimic_torch(dataset_path, batch_size, video_len, phase, depth, view='agentview'):
    assert phase in ['train', 'valid'], f'Phase is not one of the acceptable values! Got {phase}'

    loader = get_data_loader(dataset_path, batch_size, video_len, phase, depth, view)

    def prepare_data(xs):
        data_dict = {
            'video': xs['obs'][f'{view}_image'],
            'actions': xs['actions'],
            #'segmentation': xs['obs'][f'{view}_segmentation_instance']
            'segmentation': xs['obs'][f'{view}_seg']
        }
        # zero out the parts of the segmentation which are not assigned label corresponding to object of interest
        # set the object label components to 1
        object_seg_index = 0 # Seg index is 0 on the iGibson data, and 1 on Mujoco data
        seg_image = torch.zeros_like(data_dict['segmentation'])
        seg_image[data_dict['segmentation'] == object_seg_index] = 1
        seg_image[data_dict['segmentation'] != object_seg_index] = 0
        data_dict['segmentation'] = seg_image

        # segmentation = data_dict['segmentation'][0][0]
        # segmentation = torch.tile(segmentation, (3, 1, 1))
        # sample_image = data_dict['video'][0][0]
        # sample_image[segmentation != 1] = 0
        # sample_image[segmentation == 1] = 255
        # from perceptual_metrics.utils import save_np_img
        # save_np_img(np.transpose(sample_image.cpu().numpy(), (1, 2, 0)).astype(np.uint8), dataset_path[0].split('/')[-1])

        if depth:
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

    
