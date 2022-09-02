from __future__ import absolute_import
import os
import sys
import errno
import json
import os.path as osp
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from sklearn.manifold import TSNE
from scipy.stats import norm  #
import seaborn as sns  #
from torch.nn import functional as F
from collections import defaultdict
import torch
import numpy as np


def mkdir_if_missing(directory):
    if not osp.exists(directory):
        try:
            os.makedirs(directory)
        except OSError as e:
            if e.errno != errno.EEXIST:
                raise


class AverageMeter(object):
    """Computes and stores the average and current value.
       
       Code imported from https://github.com/pytorch/examples/blob/master/imagenet/main.py#L247-L262
    """
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def save_checkpoint(state, fpath='checkpoint.pth.tar'):
    mkdir_if_missing(osp.dirname(fpath))
    torch.save(state, fpath)
    """if is_best:
        shutil.copy(fpath, osp.join(osp.dirname(fpath), 'best_model.pth.tar'))"""


class Logger(object):
    """
    Write console output to external text file.
    Code imported from https://github.com/Cysu/open-reid/blob/master/reid/utils/logging.py.
    """
    def __init__(self, fpath=None, mode='w'):
        self.console = sys.stdout
        self.file = None
        if fpath is not None:
            mkdir_if_missing(os.path.dirname(fpath))
            self.file = open(fpath, mode)

    def __del__(self):
        self.close()

    def __enter__(self):
        pass

    def __exit__(self, *args):
        self.close()

    def write(self, msg):
        self.console.write(msg)
        if self.file is not None:
            self.file.write(msg)

    def flush(self):
        self.console.flush()
        if self.file is not None:
            self.file.flush()
            os.fsync(self.file.fileno())

    def close(self):
        self.console.close()
        if self.file is not None:
            self.file.close()


def read_json(fpath):
    with open(fpath, 'r') as f:
        obj = json.load(f)
    return obj


def write_json(obj, fpath):
    mkdir_if_missing(osp.dirname(fpath))
    with open(fpath, 'w') as f:
        json.dump(obj, f, indent=4, separators=(',', ': '))


def compute_distance_matrix(input1, input2, metric='euclidean'):
    """A wrapper function for computing distance matrix.

    Args:
        input1 (torch.Tensor): 2-D feature matrix.
        input2 (torch.Tensor): 2-D feature matrix.
        metric (str, optional): "euclidean" or "cosine".
            Default is "euclidean".

    Returns:
        torch.Tensor: distance matrix.

    Examples::
       >>> from torchreid import metrics
       >>> input1 = torch.rand(10, 2048)
       >>> input2 = torch.rand(100, 2048)
       >>> distmat = metrics.compute_distance_matrix(input1, input2)
       >>> distmat.size() # (10, 100)
    """
    # check input
    assert isinstance(input1, torch.Tensor)
    assert isinstance(input2, torch.Tensor)
    assert input1.dim() == 2, 'Expected 2-D tensor, but got {}-D'.format(
        input1.dim()
    )
    assert input2.dim() == 2, 'Expected 2-D tensor, but got {}-D'.format(
        input2.dim()
    )
    assert input1.size(1) == input2.size(1)

    if metric == 'euclidean':
        distmat = euclidean_squared_distance(input1, input2)
    elif metric == 'cosine':
        distmat = cosine_distance(input1, input2)
    else:
        raise ValueError(
            'Unknown distance metric: {}. '
            'Please choose either "euclidean" or "cosine"'.format(metric)
        )

    return distmat


def euclidean_squared_distance(input1, input2):
    """Computes euclidean squared distance.

    Args:
        input1 (torch.Tensor): 2-D feature matrix.
        input2 (torch.Tensor): 2-D feature matrix.

    Returns:
        torch.Tensor: distance matrix.
    """
    m, n = input1.size(0), input2.size(0)
    mat1 = torch.pow(input1, 2).sum(dim=1, keepdim=True).expand(m, n)
    mat2 = torch.pow(input2, 2).sum(dim=1, keepdim=True).expand(n, m).t()
    distmat = mat1 + mat2
    distmat.addmm_(1, -2, input1, input2.t())
    return distmat


def cosine_distance(input1, input2):
    """Computes cosine distance.

    Args:
        input1 (torch.Tensor): 2-D feature matrix.
        input2 (torch.Tensor): 2-D feature matrix.

    Returns:
        torch.Tensor: distance matrix.
    """
    input1_normed = F.normalize(input1, p=2, dim=1)
    input2_normed = F.normalize(input2, p=2, dim=1)
    distmat = 1 - torch.mm(input1_normed, input2_normed.t())
    return distmat


def show_distri(feat, num_instances, metric="euclidean", title="Title", savefile=None):
    s = torch.reshape(feat, (int(feat.shape[0]/num_instances), num_instances, feat.shape[1]))
    wcs = compute_distance_matrix(s[0], s[0], metric)
    bcs = compute_distance_matrix(s[0], s[1], metric)
    for i in s[1:]:
        wcs = torch.cat((wcs, compute_distance_matrix(i, i, metric)))
        for j in s:
            if not torch.equal(i, j):  # if j is not i:
                bcs = torch.cat((bcs, compute_distance_matrix(i, j, metric)))

    b_c = [x.cpu().detach().item() for x in bcs.flatten() if x > 0.000001]
    w_c = [x.cpu().detach().item() for x in wcs.flatten() if x > 0.000001]
    data_bc = norm.rvs(b_c)
    sns.distplot(data_bc, bins='auto', fit=norm, kde=False, label='from different class (between class)')
    data_wc = norm.rvs(w_c)
    sns.distplot(data_wc, bins='auto', fit=norm, kde=False, label='from the same class (within class)')
    plt.xlabel(metric + ' distance')
    plt.ylabel('Frequence of apparition')
    plt.title(title)
    plt.legend()
    if savefile is not None:
        plt.savefig(savefile)
    plt.show()


def T_SNE_computation(features, pids, camids, print=False, save_path="", title="Title", epoch=0):

    X_embedded = TSNE(n_components=2).fit_transform(features.cpu().numpy())

    legend_elements = [Patch(facecolor='red', edgecolor='r', label='c0'),
                       Patch(facecolor='blue', edgecolor='r', label='c1')]

    plt.figure(figsize=(20, 20))
    plt.axis([-50, 50, -50, 50])
    # loop through labels and plot each cluster
    for i, (pid, camid) in enumerate(zip(pids, camids)):
        # add data points
        if camid == 0:
            plt.text(x=X_embedded[:, 0][i],
                     y=X_embedded[:, 1][i],
                     s=str(pid),
                     ha="center",
                     va="center",
                     color="red",
                     )
        else:
            plt.text(x=X_embedded[:, 0][i],
                     y=X_embedded[:, 1][i],
                     s=str(pid),
                     ha="center",
                     va="center",
                     color="blue",
                     )

    plt.legend(handles=legend_elements)

    plt.title(title)
    if save_path != "":
        os.makedirs(save_path, exist_ok=True)
        plt.savefig(osp.join(save_path, "epoch_" + str(epoch)))

    if print:
        plt.show()
    plt.close()


def T_SNE_computation_Source_Target(features, pids, camids, print=False, save_path="", title="Title", epoch=0):

    X_embedded_ = TSNE(n_components=2).fit_transform(features.cpu().numpy())
    X_embedded = [X_embedded_[:len(pids[0])], X_embedded_[len(pids[0]):]]

    assert len(camids) == 2, "Must be a list with camids[0] containing Source camids and " \
                             "camid[1] containing Target camids"

    NUM_COLORS = len(list(set(camids[0]))) + len(list(set(camids[1])))
    colors = [["red", "orange"], ["blue", "green"]]

    legend_elements = []
    # Source
    for c in list(set(camids[0])):
        legend_elements.append(Patch(label='S_c' + str(c), color=colors[0][c]))
    # Target
    for c in list(set(camids[1])):
        legend_elements.append(Patch(label='T_c' + str(c), color=colors[1][c]))

    plt.figure(figsize=(20, 20))
    plt.axis([-50, 50, -50, 50])
    for d in range(len(pids)):
        for i, (pid, camid) in enumerate(zip(pids[d], camids[d])):
            # add data points
            plt.text(x=X_embedded[d][:, 0][i],
                     y=X_embedded[d][:, 1][i],
                     s=str(pid),
                     ha="center",
                     va="center",
                     color=colors[d][camid],
                     )

    plt.legend(handles=legend_elements)

    plt.title(title)
    if save_path != "":
        os.makedirs(save_path, exist_ok=True)
        plt.savefig(osp.join(save_path, "epoch_" + str(epoch)))

    if print:
        plt.show()
    plt.close()

    class MetricMeter(object):
        """A collection of metrics.

        Source: https://github.com/KaiyangZhou/Dassl.pytorch

        Examples::
            >>> # 1. Create an instance of MetricMeter
            >>> metric = MetricMeter()
            >>> # 2. Update using a dictionary as input
            >>> input_dict = {'loss_1': value_1, 'loss_2': value_2}
            >>> metric.update(input_dict)
            >>> # 3. Convert to string and print
            >>> print(str(metric))
        """

        def __init__(self, delimiter='\t'):
            self.meters = defaultdict(AverageMeter)
            self.delimiter = delimiter

        def update(self, input_dict):
            if input_dict is None:
                return

            if not isinstance(input_dict, dict):
                raise TypeError(
                    'Input to MetricMeter.update() must be a dictionary'
                )

            for k, v in input_dict.items():
                if isinstance(v, torch.Tensor):
                    v = v.item()
                self.meters[k].update(v)

        def __str__(self):
            output_str = []
            for name, meter in self.meters.items():
                output_str.append(
                    '{} {:.4f} ({:.4f})'.format(name, meter.val, meter.avg)
                )
            return self.delimiter.join(output_str)

class MetricMeter(object):
    """A collection of metrics.

    Source: https://github.com/KaiyangZhou/Dassl.pytorch

    Examples::
        >>> # 1. Create an instance of MetricMeter
        >>> metric = MetricMeter()
        >>> # 2. Update using a dictionary as input
        >>> input_dict = {'loss_1': value_1, 'loss_2': value_2}
        >>> metric.update(input_dict)
        >>> # 3. Convert to string and print
        >>> print(str(metric))
    """

    def __init__(self, delimiter='\t'):
        self.meters = defaultdict(AverageMeter)
        self.delimiter = delimiter

    def update(self, input_dict):
        if input_dict is None:
            return

        if not isinstance(input_dict, dict):
            raise TypeError(
                'Input to MetricMeter.update() must be a dictionary'
            )

        for k, v in input_dict.items():
            if isinstance(v, torch.Tensor):
                v = v.item()
            self.meters[k].update(v)

    def __str__(self):
        output_str = []
        for name, meter in self.meters.items():
            output_str.append(
                '{} {:.4f} ({:.4f})'.format(name, meter.val, meter.avg)
            )
        return self.delimiter.join(output_str)


def extract_features_for_clustering(model, loader, use_gpu, mode='video', seq_len=4, hm=False):
    ttf, ttf_pids, ttf_camids, ttf_imgs = [], [], [], []
    with torch.no_grad():
        model.eval()
        for batch_idx, (vids, pids, camids, img_paths, index) in enumerate(loader):
            #print(batch_idx)
            if isinstance(pids, list):
                vids = torch.cat(vids, 0)
                pids = torch.cat(pids, 0)
                camids = torch.cat(camids, 0)

            if vids.shape[2] != seq_len:
                assert vids.shape[2] % seq_len == 0
                num_clips = int(vids.shape[2] / seq_len)
                l_vids = torch.split(vids, seq_len, dim=2)
                vids = torch.cat(l_vids, dim=0)
                pids = torch.cat([pids for i in range(num_clips)], dim=0)
                camids = torch.cat([camids for i in range(num_clips)], dim=0)

            if use_gpu:
                vids = vids.cuda()
            feat = model(vids)  # [b, c * n]

            if mode == 'video':
                feat_list = torch.split(feat, 2048, dim=1)
                norm_feat_list = []
                for i, f in enumerate(feat_list):
                    if hm == False:
                        f = model.module.bn[i](f)  # [bs, c]
                        f = F.normalize(f, p=2, dim=1, eps=1e-12)
                    else:
                        f = F.normalize(f, p=2, dim=1, eps=1e-12)
                    norm_feat_list.append(f)
                feat = torch.cat(norm_feat_list, 1)

            else:
                feat = F.normalize(feat, p=2, dim=1, eps=1e-12)

            ttf.append(feat)
            ttf_pids.extend(pids)
            ttf_camids.extend(camids)
            ttf_imgs.append(img_paths)

        ttf = torch.cat(ttf, 0)
        ttf_pids = np.asarray(ttf_pids)
        ttf_camids = np.asarray(ttf_camids)

        return ttf, ttf_pids, ttf_camids, ttf_imgs

def MMD(x, y, kernel):
    """Emprical maximum mean discrepancy. The lower the result
       the more evidence that distributions are the same.

    Args:
        x: first sample, distribution P
        y: second sample, distribution Q
        kernel: kernel type such as "multiscale" or "rbf"
    """
    xx, yy, zz = torch.mm(x, x.t()), torch.mm(y, y.t()), torch.mm(x, y.t())
    rx = (xx.diag().unsqueeze(0).expand_as(xx))
    ry = (yy.diag().unsqueeze(0).expand_as(yy))

    dxx = rx.t() + rx - 2. * xx  # Used for A in (1)
    dyy = ry.t() + ry - 2. * yy  # Used for B in (1)
    dxy = rx.t() + ry - 2. * zz  # Used for C in (1)

    XX, YY, XY = (torch.zeros(xx.shape).cuda(),
                  torch.zeros(xx.shape).cuda(),
                  torch.zeros(xx.shape).cuda())

    if kernel == "multiscale":

        bandwidth_range = [0.2, 0.5, 0.9, 1.3]
        for a in bandwidth_range:
            XX += a ** 2 * (a ** 2 + dxx) ** -1
            YY += a ** 2 * (a ** 2 + dyy) ** -1
            XY += a ** 2 * (a ** 2 + dxy) ** -1

    if kernel == "rbf":

        bandwidth_range = [10, 15, 20, 50]
        for a in bandwidth_range:
            XX += torch.exp(-0.5 * dxx / a)
            YY += torch.exp(-0.5 * dyy / a)
            XY += torch.exp(-0.5 * dxy / a)

    return torch.mean(XX + YY - 2. * XY)
