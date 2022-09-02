from __future__ import division, print_function, absolute_import
import glob
import os.path as osp
import json
import os
import errno
import numpy as np

def read_json(fpath):
    """Reads json file from a path."""
    with open(fpath, 'r') as f:
        obj = json.load(f)
    return obj

def mkdir_if_missing(dirname):
    """Creates dirname if it is missing."""
    if not osp.exists(dirname):
        try:
            os.makedirs(dirname)
        except OSError as e:
            if e.errno != errno.EEXIST:
                raise

def write_json(obj, fpath):
    """Writes to a json file."""
    mkdir_if_missing(osp.dirname(fpath))
    with open(fpath, 'w') as f:
        json.dump(obj, f, indent=4, separators=(',', ': '))


class PRID2011(object):
    """PRID2011.

    Reference:
        Hirzer et al. Person Re-Identification by Descriptive and
        Discriminative Classification. SCIA 2011.

    URL: `<https://www.tugraz.at/institute/icg/research/team-bischof/lrs/downloads/PRID11/>`_

    Dataset statistics:
        - identities: 200.
        - tracklets: 400.
        - cameras: 2.
    """
    dataset_dir = 'prid2011'
    dataset_url = None

    def __init__(self, root='', split_id=0, **kwargs):
        self.root = osp.abspath(osp.expanduser(root))
        self.dataset_dir = osp.join(self.root, self.dataset_dir)
        #self.download_dataset(self.dataset_dir, self.dataset_url)

        self.split_path = osp.join(self.dataset_dir, 'splits_prid2011.json')
        self.cam_a_dir = osp.join(
            self.dataset_dir, 'prid_2011', 'multi_shot', 'cam_a'
        )
        self.cam_b_dir = osp.join(
            self.dataset_dir, 'prid_2011', 'multi_shot', 'cam_b'
        )

        required_files = [self.dataset_dir, self.cam_a_dir, self.cam_b_dir]
        self.check_before_run(required_files)

        splits = read_json(self.split_path)
        if split_id >= len(splits):
            raise ValueError(
                'split_id exceeds range, received {}, but expected between 0 and {}'
                    .format(split_id,
                            len(splits) - 1)
            )
        split = splits[split_id]
        train_dirs, test_dirs = split['train'], split['test']

        self.train = self.process_dir(train_dirs, cam1=True, cam2=True)
        self.query = self.process_dir(test_dirs, cam1=True, cam2=False)
        self.gallery = self.process_dir(test_dirs, cam1=False, cam2=True)

        self.num_train_pids = int(self.train.__len__()/2)
        self.num_train_tracklets = self.train.__len__()
        self.num_query_pids = self.query.__len__()
        self.num_query_tracklets = self.query.__len__()
        self.num_gallery_pids = self.gallery.__len__()
        self.num_gallery_tracklets = self.gallery.__len__()
        self.num_total_pids = self.num_train_pids + self.num_query_pids
        self.num_total_tracklets = self.num_train_tracklets + self.num_query_tracklets + self.num_gallery_tracklets

        self.total_camids = set(np.arange(2))

        print("=> PRID2011 loaded")
        print("Dataset statistics:")
        print("  ------------------------------")
        print("  subset   | # ids | # tracklets")
        print("  ------------------------------")
        print("  train    | {:5d} | {:8d}".format(self.num_train_pids, self.num_train_tracklets))
        print("  query    | {:5d} | {:8d}".format(self.num_query_pids, self.num_query_tracklets))
        print("  gallery  | {:5d} | {:8d}".format(self.num_gallery_pids, self.num_gallery_tracklets))
        print("  ------------------------------")
        print("  total    | {:5d} | {:8d}".format(self.num_total_pids, self.num_total_tracklets))
        print("  ------------------------------")

    def process_dir(self, dirnames, cam1=True, cam2=True):
        tracklets = []
        dirname2pid = {dirname: i for i, dirname in enumerate(dirnames)}

        for dirname in dirnames:
            if cam1:
                person_dir = osp.join(self.cam_a_dir, dirname)
                img_names = glob.glob(osp.join(person_dir, '*.png'))
                assert len(img_names) > 0
                img_names = tuple(img_names)
                pid = dirname2pid[dirname]
                tracklets.append((img_names, pid, 0))

            if cam2:
                person_dir = osp.join(self.cam_b_dir, dirname)
                img_names = glob.glob(osp.join(person_dir, '*.png'))
                assert len(img_names) > 0
                img_names = tuple(img_names)
                pid = dirname2pid[dirname]
                tracklets.append((img_names, pid, 1))

        return tracklets


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

