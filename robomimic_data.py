import jax
import torch

from robomimic.utils.dataset import SequenceDataset
import robomimic.utils.obs_utils as ObsUtils
from torch.utils.data import DataLoader, ConcatDataset


def get_data_loader(dataset_paths, batch_size, video_len, phase, depth):
    """
    Get a data loader to sample batches of data.
    """

    ObsUtils.initialize_obs_utils_with_obs_specs({
        "obs": {
            "rgb": ["agentview_image"],
            "depth": ["agentview_depth"],
        }
    })

    all_datasets = []

    for i, dataset_path in enumerate(dataset_paths):
        if depth:
            obs_keys = ("agentview_depth",)
        else:
            obs_keys = ("agentview_image",)
        dataset = SequenceDataset(
            hdf5_path=dataset_path,
            obs_keys=obs_keys,                      # observations we want to appear in batches
            dataset_keys=(                  # can optionally specify more keys here if they should appear in batches
                "actions", 
                "rewards", 
                "dones",
            ),
            load_next_obs=True,
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
        num_workers=4,
        drop_last=True      # don't provide last batch in dataset pass if it's less than 100 in size
    )
    return data_loader 


def load_dataset_robomimic(dataset_path, batch_size, video_len, is_train, depth):
    if is_train:
        phase = 'train'
    else:
        phase = 'valid'

    loader = get_data_loader(dataset_path, batch_size, video_len, phase, depth)
    local_device_count = jax.local_device_count()
    
    def prepare_data(xs):
        if depth:
            xs = {
                'video': torch.permute(xs['obs']['agentview_depth'], (0, 1, 3, 4, 2)),
                'actions': xs['actions']
            }
        else:
            xs = {
                'video': torch.permute(xs['obs']['agentview_image'], (0, 1, 3, 4, 2)),
                'actions': xs['actions']
            }

        def _prepare(x):
            x = x.numpy()
            return jax.device_put(x.reshape((local_device_count, -1) + x.shape[1:]))
        prepared = jax.tree_map(_prepare, xs)

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
    
    dataset = load_dataset_robomimic(dataset_path, 16, 10, True)
    batch = next(dataset)
    import ipdb; ipdb.set_trace()
    
    #dataloader = get_data_loader(dataset_path, 1, 10, 'train')
    #batch = next(iter(dataloader))

    
