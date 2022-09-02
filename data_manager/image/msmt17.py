from __future__ import division, print_function, absolute_import
import re
import glob
import os.path as osp
import warnings

##### Log #####
# 22.01.2019
# - add v2
# - v1 and v2 differ in dir names
# - note that faces in v2 are blurred
"""TRAIN_DIR_KEY = 'train_dir'
TEST_DIR_KEY = 'test_dir'
VERSION_DICT = {
    'MSMT17_V1': {
        TRAIN_DIR_KEY: 'train',
        TEST_DIR_KEY: 'test',
    },
    'MSMT17_V2': {
        TRAIN_DIR_KEY: 'mask_train_v2',
        TEST_DIR_KEY: 'mask_test_v2',
    }
}"""


class MSMT17(object):
    """MSMT17.

    Reference:
        Wei et al. Person Transfer GAN to Bridge Domain Gap for Person Re-Identification. CVPR 2018.

    URL: `<http://www.pkuvmc.com/publications/msmt17.html>`_

    Dataset statistics:
        - identities: 4101.
        - images: 32621 (train) + 11659 (query) + 82161 (gallery).
        - cameras: 15.
    """
    _junk_pids = [0, -1]
    dataset_dir = 'msmt17'
    dataset_url = ''

    def __init__(self, root='', market1501_500k=False, **kwargs):
        self.root = osp.abspath(osp.expanduser(root))
        self.dataset_dir = osp.join(self.root, self.dataset_dir)
        # self.download_dataset(self.dataset_dir, self.dataset_url)

        # allow alternative directory structure
        self.data_dir = self.dataset_dir
        data_dir = osp.join(self.data_dir, 'MSMT17_V1')
        if osp.isdir(data_dir):
            self.data_dir = data_dir
        else:
            warnings.warn(
                'The current data structure is deprecated. Please '
                'put data folders such as "bounding_box_train" under '
                '"Market-1501-v15.09.15".'
            )

        self.train_dir = osp.join(self.data_dir, 'bounding_box_train')
        self.query_dir = osp.join(self.data_dir, 'query')
        self.gallery_dir = osp.join(self.data_dir, 'bounding_box_test')
        self.extra_gallery_dir = osp.join(self.data_dir, 'images')
        self.market1501_500k = market1501_500k

        required_files = [
            self.data_dir, self.train_dir, self.query_dir, self.gallery_dir
        ]
        if self.market1501_500k:
            required_files.append(self.extra_gallery_dir)
        self.check_before_run(required_files)

        train, train_pids, train_camids, num_train_imgs = self.process_dir(self.train_dir, relabel=True)
        query, query_pids, query_camids, num_query_imgs = self.process_dir(self.query_dir, relabel=False)
        gallery, gallery_pids, gallery_camids, num_gallery_imgs = self.process_dir(self.gallery_dir, relabel=False)
        if self.market1501_500k:
            gallery += self.process_dir(self.extra_gallery_dir, relabel=False)

        total_pids = train_pids | query_pids | gallery_pids
        total_camids = train_camids | query_camids | gallery_camids
        num_total_imgs = num_train_imgs + num_query_imgs + num_gallery_imgs

        print("=> Market1501 loaded")
        print("Dataset statistics:")
        print("  ------------------------------")
        print("  subset   | # ids | # cams | # images")
        print("  ------------------------------")
        print("  train    | {:5d} | {:5d} | {:8d}".format(len(train_pids), len(train_camids), num_train_imgs))
        print("  query    | {:5d} | {:5d} | {:8d}".format(len(query_pids), len(query_camids), num_query_imgs))
        print("  gallery  | {:5d} | {:5d} | {:8d}".format(len(gallery_pids), len(gallery_camids), num_gallery_imgs))
        print("  ------------------------------")
        print("  total    | {:5d} | {:5d} | {:8d}".format(len(total_pids), len(total_camids), num_total_imgs))
        print("  ------------------------------")

        self.train, self.train_pids, self.train_camids, self.num_train_imgs = train, train_pids, train_camids, num_train_imgs
        self.query, self.query_pids, self.query_camids, self.num_query_imgs = query, query_pids, query_camids, num_query_imgs
        self.gallery, self.gallery_pids, self.gallery_camids, self.num_gallery_imgs = gallery, gallery_pids, gallery_camids, num_gallery_imgs

        self.total_camids = total_camids

    def process_dir(self, dir_path, relabel=False):
        img_paths = glob.glob(osp.join(dir_path, '*.jpg'))
        pattern = re.compile(r'([-\d]+)_c([\d]+)')  # for MSMT17

        pid_container = set()
        camid_container = set()
        for img_path in img_paths:
            pid, camid = map(int, pattern.search(img_path).groups())
            if pid == -1:
                continue  # junk images are just ignored
            pid_container.add(pid)
            camid_container.add(camid)
        pid2label = {pid: label for label, pid in enumerate(pid_container)}

        data = []
        for img_path in img_paths:
            pid, camid = map(int, pattern.search(img_path).groups())
            if pid == -1:
                continue  # junk images are just ignored
            assert 0 <= pid <= 4101  # pid == 0 means background
            assert 1 <= camid <= 15
            camid -= 1  # index starts from 0
            if relabel:
                pid = pid2label[pid]
            data.append((img_path, pid, camid))

        num_pids = pid_container
        num_camids = camid_container
        num_imgs = len(img_paths)

        return data, num_pids, num_camids, num_imgs

    def check_before_run(self, required_files):
        """Checks if required files exist before going deeper.

        Args:
            required_files (str or list): string file name(s).
        """
        if isinstance(required_files, str):
            required_files = [required_files]

        for fpath in required_files:
            if not osp.exists(fpath):
                raise RuntimeError('"{}" is not found'.format(fpath))


if __name__ == '__main__':
    # test
    dataset = MSMT17()
