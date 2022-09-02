from __future__ import absolute_import
import time
import torch
import torch.nn.functional as F
import numpy as np

from tools.eval_metrics import evaluate
from tools.utils import extract_features_for_clustering


def test(model, queryloader, galleryloader, use_gpu, is_target, ranks=[1, 5, 10, 20], writer=None, epoch=0,
         name_dataset=""):

    since = time.time()
    model.eval()

    qf, q_pids, q_camids = [], [], []
    for batch_idx, (vids, pids, camids, _, _) in enumerate(queryloader):
        if use_gpu:
            vids = vids.cuda()

        feat = model(vids)  # [b, c * n]

        feat_list = torch.split(feat, 2048, dim=1)
        norm_feat_list = []
        for i, f in enumerate(feat_list):
            f = model.module.bn[i](f)  # [bs, c]
            f = F.normalize(f, p=2, dim=1, eps=1e-12)
            norm_feat_list.append(f)
        feat = torch.cat(norm_feat_list, 1)

        qf.append(feat)
        q_pids.extend(pids)
        q_camids.extend(camids)

    qf = torch.cat(qf, 0)
    q_pids = np.asarray(q_pids)
    q_camids = np.asarray(q_camids)
    #print("Extracted features for query set, obtained {} matrix".format(qf.shape))

    gf, g_pids, g_camids = [], [], []
    for batch_idx, (vids, pids, camids, _, _) in enumerate(galleryloader):
        if use_gpu:
            vids = vids.cuda()
        feat = model(vids)

        feat_list = torch.split(feat, 2048, dim=1)
        norm_feat_list = []
        for i, f in enumerate(feat_list):
            f = model.module.bn[i](f)  # [bs, c]
            f = F.normalize(f, p=2, dim=1, eps=1e-12)
            norm_feat_list.append(f)
        feat = torch.cat(norm_feat_list, 1)

        gf.append(feat)
        g_pids.extend(pids)
        g_camids.extend(camids)

    gf = torch.cat(gf, 0)
    g_pids = np.asarray(g_pids)
    g_camids = np.asarray(g_camids)

    if name_dataset == 'mars':
        # gallery set must contain query set, otherwise 140 query imgs will not have ground truth.
        gf = torch.cat((qf, gf), 0)
        g_pids = np.append(q_pids, g_pids)
        g_camids = np.append(q_camids, g_camids)

    #print("Extracted features for gallery set, obtained {} matrix".format(gf.shape))

    time_elapsed = time.time() - since
    #print('Extracting features complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))

    #print("Computing distance matrix")

    distmat = - torch.mm(qf, gf.t())
    distmat = distmat.data.cpu()
    distmat = distmat.numpy()

    #print("Computing CMC and mAP")
    max_rank = 50
    num_q, num_g = distmat.shape
    if num_g < max_rank:
        max_rank = num_g
        print("Note: number of gallery samples is quite small, got {}".format(num_g))
    indices = np.argsort(distmat, axis=1)
    matches = (g_pids[indices] == q_pids[:, np.newaxis]).astype(np.int32)

    # compute cmc curve for each query
    all_cmc = []
    all_AP = []
    num_valid_q = 0.  # number of valid query
    for q_idx in range(num_q):
        # get query pid and camid
        q_pid = q_pids[q_idx]
        q_camid = q_camids[q_idx]

        # remove gallery samples that have the same pid and camid with query
        order = indices[q_idx]
        remove = (g_pids[order] == q_pid) & (g_camids[order] == q_camid)
        keep = np.invert(remove)

        # compute cmc curve
        orig_cmc = matches[q_idx][keep]  # binary vector, positions with value 1 are correct matches
        if not np.any(orig_cmc):
            # this condition is true when query identity does not appear in gallery
            continue

        cmc = orig_cmc.cumsum()
        cmc[cmc > 1] = 1

        all_cmc.append(cmc[:max_rank])
        num_valid_q += 1.

        # compute average precision
        # reference: https://en.wikipedia.org/wiki/Evaluation_measures_(information_retrieval)#Average_precision
        num_rel = orig_cmc.sum()
        tmp_cmc = orig_cmc.cumsum()
        tmp_cmc = [x / (i+1.) for i, x in enumerate(tmp_cmc)]
        tmp_cmc = np.asarray(tmp_cmc) * orig_cmc
        AP = tmp_cmc.sum() / num_rel
        all_AP.append(AP)

    assert num_valid_q > 0, "Error: all query identities do not appear in gallery"

    all_cmc = np.asarray(all_cmc).astype(np.float32)
    all_cmc = all_cmc.sum(0) / num_valid_q
    mAP = np.mean(all_AP)

    cmc, mAP = evaluate(distmat, q_pids, g_pids, q_camids, g_camids)

    print("Results ----------")
    print("mAP: {:.2%}".format(mAP))
    print("CMC curve")
    for r in ranks:
        print("Rank-{:<3}: {:.2%}".format(r, cmc[r - 1]))
    print("------------------")
    if writer is not None:
        writer.add_scalar(f'Test/{name_dataset}-{"target" if is_target else "source"}/rank1', cmc[0], epoch)
        writer.add_scalar(f'Test/{name_dataset}-{"target" if is_target else "source"}/mAP', mAP, epoch)
    return cmc[0]
