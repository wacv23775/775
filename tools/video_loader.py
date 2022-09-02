from __future__ import print_function, absolute_import

import os
import torch
import functools
import torch.utils.data as data
from PIL import Image
import numpy as np
import os.path as osp
import random
import time

def pil_loader(path, mode):
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        with Image.open(f) as img:
            return img.convert(mode)


def accimage_loader(path):
    try:
        import accimage
        return accimage.Image(path)
    except IOError:
        # Potentially a decoding problem, fall back to PIL.Image
        return pil_loader(path)


def get_default_image_loader():
    from torchvision import get_image_backend
    if get_image_backend() == 'accimage':
        return accimage_loader
    else:
        return pil_loader


def image_loader(path):
    from torchvision import get_image_backend
    if get_image_backend() == 'accimage':
        print("accIMAGE")
        return accimage_loader(path)
    else:
        print("PIL")
        return pil_loader(path)


def video_loader(img_paths, mode, image_loader):
    video = []
    for image_path in img_paths:
        if os.path.exists(image_path):
            video.append(image_loader(image_path, mode))
        else:
            return video
    return video


def get_default_video_loader():
    image_loader = get_default_image_loader()
    return functools.partial(video_loader, image_loader=image_loader)


def read_image(path):
    """Reads image from path using ``PIL.Image``.

    Args:
        path (str): path to an image.

    Returns:
        PIL image
    """
    got_img = False
    if not osp.exists(path):
        raise IOError('"{}" does not exist'.format(path))
    while not got_img:
        try:
            img = Image.open(path).convert('RGB')
            got_img = True
        except IOError:
            print(
                'IOError incurred when reading "{}". Will redo. Don\'t worry. Just chill.'
                .format(path)
            )
    return img


class VideoDataset(data.Dataset):
    """Video Person ReID Dataset.
    Note:
        Batch data has shape N x C x T x H x W
    Args:
        dataset (list): List with items (img_paths, pid, camid)
        temporal_transform (callable, optional): A function/transform that  takes in a list of frame indices
            and returns a transformed version
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
        loader (callable, optional): A function to load an video given its path and frame indices.
    """

    def __init__(self, 
                 dataset, 
                 spatial_transform=None,
                 temporal_transform=None,
                 get_loader=get_default_video_loader,
                 sample_method="",
                 seq_len=10,
                 num_clips=1,
                 chunks=4):
        self.dataset = dataset
        self.spatial_transform = spatial_transform
        self.temporal_transform = temporal_transform
        self.loader = get_loader()
        self.sample_method = sample_method
        self.seq_len = seq_len
        self.chunks = chunks
        self.num_clips = num_clips

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (clip, pid, camid) where pid is identity of the clip.
        """
        img_paths, pid, camid = self.dataset[index]
        img_paths_for_re_sampling = img_paths
        num_imgs = len(img_paths)

        if self.num_clips == 1:

            if self.temporal_transform is not None:
                img_paths = self.temporal_transform(img_paths)

            clip = self.loader(img_paths, mode='RGB')

            if self.spatial_transform is not None:
                self.spatial_transform.randomize_parameters()
                clip = [self.spatial_transform(img) for img in clip]

            # trans T x C x H x W to C x T x H x W
            clip = torch.stack(clip, 0).permute(1, 0, 2, 3)

            return clip, pid, camid, img_paths_for_re_sampling, index

        else:
            indices = []
            size_clips_pool = int(num_imgs/self.num_clips)
            sampled_img_paths = []
            for i in range(self.num_clips):
                sampled_img_paths += self._tclnet_train_sampling(img_paths[i*size_clips_pool:(i+1)*size_clips_pool])
            for i, p in enumerate(sampled_img_paths):
                indices.append(img_paths.index(p))

        clips = self.loader(sampled_img_paths, mode='RGB')

        if self.spatial_transform is not None:
            self.spatial_transform.randomize_parameters()
            clips = [self.spatial_transform(img) for img in clips]
            # trans T x C x H x W to C x T x H x W
        clips = torch.stack(clips, 0).permute(1, 0, 2, 3)

        return clips, pid, camid, img_paths_for_re_sampling, index

    def _tclnet_train_sampling(self, img_paths):
        img_paths = list(img_paths)
        self.size = self.seq_len
        self.stride = 8
        if len(img_paths) >= self.size * self.stride:
            rand_end = len(img_paths) - (self.size - 1) * self.stride - 1
            begin_index = random.randint(0, rand_end)
            end_index = begin_index + (self.size - 1) * self.stride + 1
            out = img_paths[begin_index:end_index:self.stride]
        elif len(img_paths) >= self.size:
            index = np.random.choice(len(img_paths), size=self.size, replace=False)
            index.sort()
            out = [img_paths[index[i]] for i in range(self.size)]
        else:
            index = np.random.choice(len(img_paths), size=self.size, replace=True)
            index.sort()
            out = [img_paths[index[i]] for i in range(self.size)]

        return out


class ImageDataset(data.Dataset):
    """A base class representing ImageDataset.

    All other image datasets should subclass it.

    ``__getitem__`` returns an image given index.
    It will return ``img``, ``pid``, ``camid`` and ``img_path``
    where ``img`` has shape (channel, height, width). As a result,
    data in each batch has shape (batch_size, channel, height, width).
    """

    def __init__(self, dataset,
                 transform=None,
                 get_loader=get_default_video_loader,
                 sample_method="",
                 seq_len=10,
                 subtracklets=False,
                 chunks=4):
        self.loader = get_default_image_loader()
        self.dataset = dataset
        self.transform = transform

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        img_path, pid, camid = self.dataset[index]
        #img = read_image(img_path)
        img = self.loader(img_path, mode='RGB')
        if self.transform is not None:
            img = self.transform(img)

        return img, pid, camid, img_path, index
