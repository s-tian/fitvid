import os
import random
import glob
import cv2
import math
import pickle

import numpy as np
import torch
import torch.nn.functional as F



def short_side_scale(x, size):
    """
    Determines the shorter spatial dim of the video (i.e. width or height) and
    scales it to the given size. To maintain aspect ratio, the longer side is
    then scaled accordingly.
    """
    *_, h, w = x.shape
    if w < h:
        new_h = int(math.floor((float(h) / w) * size))
        new_w = size
    else:
        new_h = size
        new_w = int(math.floor((float(w) / h) * size))
    return F.interpolate(x, size=(new_h, new_w), mode="bilinear", align_corners=False)


def center_crop(x, size):
    *_, h, w = x.shape
    w_start = int(math.ceil((w - size) / 2))
    h_start = int(math.ceil((h - size) / 2))

    return x[:, :, h_start : h_start + size, w_start : w_start + size]


def load_img_list(image_paths, backend="pytorch"):
    """
    This function is to load images.
    Args:
        image_paths (list): paths of images needed to be loaded.
        retry (int, optional): maximum time of loading retrying. Defaults to 10.
        backend (str): `pytorch` or `cv2`.
    Returns:
        imgs (list): list of loaded images.
    """
    imgs = []
    for image_path in image_paths:
        img = cv2.imread(image_path, flags=cv2.IMREAD_COLOR)
        # bgr --> rgb
        img = img[..., ::-1]
        imgs.append(img)

    if all(img is not None for img in imgs):
        if backend == "pytorch":
            imgs = torch.as_tensor(np.stack(imgs))
        return imgs
    else:
        raise Exception("Failed to load images {}".format(image_paths))


def get_seq_frames(seq_len, frame_skip, video_length):
    """
    Sample seq_len frames for frame_list while skipping frame_skip number
    of frames. frame_skip = 1 means consecutive frames.
    """
    # For shorter videos reduce the frame_skip automatically to max possible value
    frame_skip = min(frame_skip, video_length // seq_len)

    # clip len including skipped frames
    clip_len = frame_skip * (seq_len - 1) + 1
    max_start = video_length - clip_len
    start = random.randint(0, max_start)
    seq = [start + i * frame_skip for i in range(seq_len)]
    return seq


def preprocess(video, resolution=64):
    video = video.permute(0, 3, 1, 2).float() / 255.0
    video = short_side_scale(video, resolution)
    video = center_crop(video, resolution)
    return video


def load_robonet_data(
    directory, metadata_path, mode, bs, sel_len, additional_finetune_data=False, only_finetune=False):
    dataset = RoboNetData(
        directory, metadata_path, mode, sel_len, additional_finetune_data, only_finetune
    )
    loader = torch.utils.data.DataLoader(
        dataset, batch_size=bs, shuffle=True, num_workers=4, drop_last=True
    )
    return loader

def get_sawyer_image_and_actions(data_path, filenames, infer_gripper):
    traj_paths = [os.path.join(data_path, "train", f) for f in filenames]
    all_image_paths, action_list, labels = list(), list(), list()
    if infer_gripper:
        print("Will infer gripper action from robot gripper state")
    else:
        print("Not inferring gripper actions! actions being padded to 5-dims")
    for traj_path in traj_paths:
        cam_path = os.path.join(traj_path, "images0")
        image_paths = glob.glob(os.path.join(cam_path, "*.jpg"))
        # The 3: index below slices an image which has name in the format ex. im_6.jpg and slices off
        # the "im" part.
        image_paths.sort(key=lambda x: int(os.path.splitext(os.path.basename(x))[0][3:]))
        action_path = os.path.join(traj_path, "policy_out.pkl")
        obs_path = os.path.join(traj_path, "obs_dict.pkl")
        # self._videos.append(image_paths)
        # self._videos_raw.append(image_paths)
        # self._labels.append("")
        # self._label_text.append("")
        all_image_paths.append(image_paths)
        labels.append("")
        with open(action_path, "rb") as f:
            per_step_policy_log = pickle.load(f, encoding="latin1")
            actions = np.stack([t["actions"] for t in per_step_policy_log])
        if infer_gripper:
            # Infer gripper action like in
            # https://github.com/SudeepDasari/visual_foresight/blob/4c79886cf6a01f2dba19fec8638073702eaf7ef9/visual_mpc/utils/file_2_record.py#L39
            with open(obs_path, "rb") as f:
                obs_dict = pickle.load(f, encoding="latin1")
                gripper_actions = []
                for i in range(len(actions)):
                    if obs_dict["state"][i + 1, -1] >= 0.5:
                        gripper_actions.append(np.array([-1.0]))
                    else:
                        gripper_actions.append(np.array([1.0]))
            gripper_actions = np.stack(gripper_actions).astype(actions.dtype)
            actions = np.concatenate((actions, gripper_actions), axis=-1)
        else:
            actions = np.concatenate(
                (actions, np.zeros_like(actions[:, -1])[..., None]), axis=-1
            )
        # Append dummy action to not break sampling, but this action should not be used!
        dummy_action = np.zeros(actions.shape[-1])
        actions = np.concatenate((actions, dummy_action[None]), axis=0).astype(np.float32)
        # self._actions.append(actions)
        action_list.append(actions)
    return all_image_paths, action_list, labels


class RoboNetData(torch.utils.data.Dataset):
    def __init__(
        self, directory, metadata_path, mode, sel_len, additional_finetune_data=None, only_finetune=False
    ):
        assert mode in [
            "train",
            "val",
            "test",
        ], "mode must be one of 'train', 'val', or 'test'"
        self.mode = mode
        self.mode_map = {
            "train": "train_trajectories.txt",
            "val": "test_256_test_trajectories.txt",
        }

        self.tars_sawyer_mode_map = {
            "train": "train_trajectories.txt",
            "val": "val_trajectories.txt",
        }

        self.tars_sawyer_metadata_path = "/viscam/u/stian/mfm/data/mfm_robot_data_09_16_meta/"
        self.tars_sawyer_data_path = "/svl/u/stian/tars_09_16/"
        self.infer_gripper = True
        self.metadata_path = metadata_path
        self.directory = directory
        self.additional_finetune_data = additional_finetune_data
        self.only_finetune = only_finetune
        self.seq_len = sel_len
        self.frame_skip = 1
        self.use_actions_cond = True

        self._construct_loader()

    def __getitem__(self, idx):
        label = self._labels[idx]
        label_text = self._label_text[idx]
        # Subsample frames
        frame_list = self._videos[idx]
        seq = get_seq_frames(
            self.seq_len,
            self.frame_skip,
            len(self._videos[idx]),
        )
        frames = load_img_list([frame_list[i] for i in seq])

        item = {}
        # Preprocess frames
        item["video"] = preprocess(frames)
        item["label"] = label
        if self.use_actions_cond:
            item["actions"] = self._actions[idx][seq]
        return item

    @property
    def num_videos(self):
        """
        Returns:
            (int): the number of videos in the dataset.
        """
        return len(self._videos)

    def __len__(self):
        """
        Returns:
            (int): the number of videos in the dataset.
        """
        return self.num_videos

    def _construct_loader(self, check_cache_from_cfg=False):
        self._videos = []
        self._videos_raw = []
        self._labels = []
        self._label_text = []
        self._latents = []
        self._actions = []
        if not self.only_finetune:
            # split_dir = os.path.join(self.cfg.data.data_path, "processed_data", split)
            split_file = self.mode_map[self.mode]
            with open(os.path.join(self.metadata_path, split_file), "r") as f:
                filenames = f.read().splitlines()

            traj_paths = [
                os.path.join(self.directory, f.replace(".hdf5", ""))
                for f in filenames
            ]
            for traj_path in traj_paths:
                cam_paths = glob.glob(os.path.join(traj_path, "cam_*"))
                for cam_path in cam_paths:
                    image_paths = glob.glob(os.path.join(cam_path, "*.jpg"))
                    image_paths.sort(
                        key=lambda x: int(os.path.splitext(os.path.basename(x))[0])
                    )
                    try:
                        action_path = os.path.join(traj_path, "actions.npy")
                        actions = np.load(action_path)
                    except Exception as e:
                        print(e)
                        continue
                    self._videos.append(image_paths)
                    self._videos_raw.append(image_paths)
                    self._labels.append("")
                    self._label_text.append("")
                    # Append dummy action to not break sampling, but this action should not be used!
                    dummy_action = np.zeros(actions.shape[-1])
                    actions = np.concatenate((actions, dummy_action[None]), axis=0)
                    actions = actions.astype(np.float32)
                    self._actions.append(actions)
        else:
            print("Only finetuning, not loading base RoboNet data")
        if self.additional_finetune_data:
            print("Adding our sawyer data!")
            split_file = self.tars_sawyer_mode_map[self.mode]
            with open(
                os.path.join(self.tars_sawyer_metadata_path, split_file), "r"
            ) as f:
                sawyer_filenames = f.read().splitlines()
            image_paths, action_list, labels = get_sawyer_image_and_actions(
                self.tars_sawyer_data_path,
                sawyer_filenames,
                self.infer_gripper,
            )
            self._videos.extend(image_paths)
            self._videos_raw.extend(image_paths)
            self._labels.extend(labels)
            self._label_text.extend(labels)
            self._actions.extend(action_list)
