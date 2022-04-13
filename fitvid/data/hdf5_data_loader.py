import torch.utils.data as data
import torch
import numpy as np
from PIL import Image
import glob
import h5py
import random
import pdb
import matplotlib.pyplot as plt
import imageio
from torchvision.transforms import Resize, RandomResizedCrop, ColorJitter
import imp
from torch.utils.data import DataLoader
import os
import moviepy.editor as mpy

class AttrDict(dict):
    __setattr__ = dict.__setitem__

    def __getattr__(self, attr):
        # Take care that getattr() raises AttributeError, not KeyError.
        # Required e.g. for hasattr(), deepcopy and OrderedDict.
        try:
            return self.__getitem__(attr)
        except KeyError:
            raise AttributeError("Attribute %r not found" % attr)

    def __getstate__(self): return self
    def __setstate__(self, d): self = d


def map_dict(fn, d):
    """takes a dictionary and applies the function to every element"""
    return type(d)(map(lambda kv: (kv[0], fn(kv[1])), d.items()))


def resize_video(video, size):
    transformed_video = np.stack([np.asarray(Resize(size, antialias=True)(torch.Tensor(im))) for im in video], axis=0) #resize in grayscale mode!
    return transformed_video


class BaseVideoDataset(data.Dataset):
    def __init__(self, data_dir, mpar, data_conf, phase, shuffle=True):
        """

        :param data_dir:
        :param mpar:
        :param data_conf:
        :param phase:
        :param shuffle: whether to shuffle within batch, set to False for computing metrics
        :param dataset_size:
        """

        self.phase = phase
        self.data_dir = data_dir
        self.data_conf = data_conf

        self.shuffle = shuffle
        self.img_sz = mpar.img_sz

        if shuffle:
            self.n_worker = 4
        else:
            self.n_worker = 1
        #todo debug:
        self.n_worker = 8

    def get_data_loader(self, batch_size):
        print('len {} dataset {}'.format(self.phase, len(self)))
        return DataLoader(self, batch_size=batch_size, shuffle=self.shuffle, num_workers=self.n_worker,
                                  drop_last=True)


class FixLenVideoDataset(BaseVideoDataset):
    """
    Variable length video dataset
    """

    def __init__(self, data_dir, mpar, data_conf, duplicates=1000, phase='train', shuffle=True):
        """

        :param data_dir:
        :param data_conf:
        :param data_conf:  Attrdict with keys
        :param phase:
        :param shuffle: whether to shuffle within batch, set to False for computing metrics
        :param dataset_size:
        """
        super().__init__(data_dir, mpar, data_conf, phase, shuffle)

        self.filenames = self._get_filenames()
        random.seed(1)
        random.shuffle(self.filenames)

        self._data_conf = data_conf
        self.duplicates_per_file = duplicates
        self.traj_per_file = self.get_traj_per_file(self.filenames[0])

        if hasattr(data_conf, 'T'):
            self.T = data_conf.T
        else: self.T = self.get_total_seqlen(self.filenames[0])

        self.flatten_im = False
        self.filter_repeated_tail = False

        self.cached_data = dict()

        # data augmentation
        #self.transform = RandomResizedCrop((self.img_sz[0], self.img_sz[1]), scale=(0.8, 1.0), ratio=(1., 1.))
        self.transform = ColorJitter(brightness=0.1, contrast=0.1)
        # Load data into RAM
        for file_index in range(len(self.filenames)):
            path = self.filenames[file_index]
            if path not in self.cached_data:
                print(f'Loading {path} into memory...')
                with h5py.File(path, 'r') as F:
                    ex_index = 0
                    key = 'traj{}'.format(ex_index)

                    # Fetch data into a dict
                    data_dict = AttrDict(images=(F[key + '/images'][:]))
                    for name in F[key].keys():
                        if name in ['states', 'actions', 'pad_mask']:
                            data_dict[name] = F[key + '/' + name][:].astype(np.float32)
                data_dict = self.process_data_dict(data_dict)
                # if self._data_conf.sel_len != -1:
                # data_dict = self.sample_rand_shifts(data_dict)
                self.cached_data[path] = data_dict

        print(phase)
        print(len(self.filenames))

    def _get_filenames(self):
        assert 'hdf5' not in self.data_dir, "hdf5 most not be contained in the data dir!"
        filenames = sorted(glob.glob(os.path.join(self.data_dir, os.path.join('hdf5', self.phase) + '/*')))
        if not filenames:
            raise RuntimeError('No filenames found in {}'.format(self.data_dir))
        return filenames

    def get_traj_per_file(self, path):
        with h5py.File(path, 'r') as F:
            return F['traj_per_file'][()]

    def get_total_seqlen(self, path):
        with h5py.File(path, 'r') as F:
            return F['traj0']['images'][:].shape[0]

    def __getitem__(self, index):
        file_index = (index // self.traj_per_file) // self.duplicates_per_file
        path = self.filenames[file_index]
        segment = self.sample_rand_shifts(self.cached_data[path])
        return {
            'video': self.apply_image_aug(segment.demo_seq_images),
            'actions': segment.actions
        }

    def apply_image_aug(self, x):
        x = torch.Tensor(x)
        return self.transform(x).numpy()

    def process_data_dict(self, data_dict):
        data_dict.demo_seq_images = self.preprocess_images(data_dict['images'])
        return data_dict

    def sample_rand_shifts(self, data_dict):
        """ This function processes data tensors so as to have length equal to max_seq_len
        by sampling / padding if necessary """
        offset = np.random.randint(0, self.T - self._data_conf.sel_len, 1)
        data_dict = map_dict(lambda tensor: self._croplen(tensor, offset, self._data_conf.sel_len), data_dict)
        #if 'actions' in data_dict:
        #    data_dict.actions = data_dict.actions[:-1]
        return data_dict

    def preprocess_images(self, images):
        # Resize video
        if len(images.shape) == 5:
            images = images[:, 0]  # Number of cameras, used in RL environments
        assert images.dtype == np.uint8, 'image need to be uint8!'
        images = np.transpose(images, [0, 3, 1, 2])  # convert to channel-first
        images = resize_video(images, (self.img_sz[0], self.img_sz[1]))
        images = images.astype(np.float32) / 255.
        assert images.dtype == np.float32, 'image need to be float32!'
        if self.flatten_im:
            images = np.reshape(images, [images.shape[0], -1])
        return images

    def __len__(self):
        return len(self.filenames) * self.traj_per_file * self.duplicates_per_file

    @staticmethod
    def _croplen(val, offset, target_length):
        """Pads / crops sequence to desired length."""
        return val[int(offset): int(offset) + target_length]
        val = val[int(offset):]
        len = val.shape[0]
        if len > target_length:
            return val[:target_length]
        elif len < target_length:
            raise ValueError("not enough length")
        else:
            return val

    @staticmethod
    def get_dataset_spec(data_dir):
        return imp.load_source('dataset_spec', os.path.join(data_dir, 'dataset_spec.py')).dataset_spec


def load_hdf5_data(data_dir, bs, data_type='train'):
    data_dir = data_dir[0]
    hp = AttrDict(img_sz=(64, 64),
                  sel_len=12,
                  T=600)
    loader = FixLenVideoDataset(data_dir, hp, hp, phase=data_type).get_data_loader(bs)
    return loader


if __name__ == '__main__':
    data_dir = '/viscam/u/stian/ff_data/ferro_600_3Hz/'
    hp = AttrDict(img_sz=(64, 64),
                  sel_len=12,
                  T=31)

    loader = FixLenVideoDataset(data_dir, hp, hp).get_data_loader(32)

    for i_batch, sample_batched in enumerate(loader):
        print(i_batch)
        images = np.asarray(sample_batched['video'])

        images = np.transpose((images + 1) / 2, [0, 1, 3, 4, 2])  # convert to channel-first
        actions = np.asarray(sample_batched['actions'])
        #print('actions', actions)

        plt.imshow(np.asarray(images[0, 0]))
        plt.show()

