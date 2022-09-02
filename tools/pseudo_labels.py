from __future__ import division, print_function, absolute_import

from tools.rerank import compute_jaccard_distance
import collections
from sklearn.cluster import DBSCAN
import torch
import numpy as np
import matplotlib.pyplot as plt


def generate_labels_criterion(
        rerank_dist_jaccard, source_classes, dataset_target, eps, eps_gap, num_clips=1, majority_vote=False
        ):

    eps_tight = eps - eps_gap
    eps_loose = eps + eps_gap
    print(
        'Clustering criterion: eps: {:.3f}, eps_tight: {:.3f}, eps_loose: {:.3f}'.format(eps, eps_tight,
                                                                                         eps_loose))
    cluster = DBSCAN(eps=eps, min_samples=2, metric='precomputed', n_jobs=-1)
    cluster_tight = DBSCAN(eps=eps_tight, min_samples=2, metric='precomputed', n_jobs=-1)
    cluster_loose = DBSCAN(eps=eps_loose, min_samples=2, metric='precomputed', n_jobs=-1)

    # select & cluster images as training set of this epochs
    pseudo_labels = cluster.fit_predict(rerank_dist_jaccard)
    pseudo_labels_tight = cluster_tight.fit_predict(rerank_dist_jaccard)
    pseudo_labels_loose = cluster_loose.fit_predict(rerank_dist_jaccard)

    num_ids = len(set(pseudo_labels)) - (1 if -1 in pseudo_labels else 0)
    num_ids_tight = len(set(pseudo_labels_tight)) - (1 if -1 in pseudo_labels_tight else 0)
    num_ids_loose = len(set(pseudo_labels_loose)) - (1 if -1 in pseudo_labels_loose else 0)

    def generate_pseudo_labels(cluster_id, num):
        labels = []
        outliers = 0

        # --------- ADDED SPECIFICALLY FOR CLIPS) ---------------
        dataset_final = []
        for i in dataset_target.train:
            for r in range(0, num_clips):
                dataset_final.append(i)
        # ------------------------------------------------------

        for i, ((fname, _, cid), id) in enumerate(zip(sorted(dataset_final), cluster_id)):

            if id != -1:
                labels.append(source_classes + id)
            else:
                labels.append(source_classes + num + outliers)
                outliers += 1
        return torch.Tensor(labels).long()

    # --------------------------- ADDED SPECIFICALLY FOR CLIPS) ------------------------
    from collections import Counter

    def generate_pseudo_labels_clips(cluster_id, num_clips, rerank_dist_jaccard):

        pseudo_labels_test = torch.tensor(cluster_id).view(int(cluster_id.shape[0] / num_clips), num_clips)
        receptacle = torch.zeros(pseudo_labels_test.shape)
        count_clustered_samples = (pseudo_labels_test != -1).sum(dim=-1)  # 0 : no_outlier, 1: 1 outlier, 2: 2+ outliers

        cpt_outlier = pseudo_labels_test.max() + 1  #
        for i in range(0, len(count_clustered_samples)):

            # Consolidation of clips inside the same tracklet
            if count_clustered_samples[i] == 0:
                receptacle[i] = torch.ones(num_clips) * cpt_outlier
                cpt_outlier += 1

            # Assignation of all clips seen as outliers to the the same cluster of the unique clip clustered
            elif count_clustered_samples[i] == 1:
                pos_cluster = (pseudo_labels_test[i] != -1).nonzero()
                label_id = pseudo_labels_test[i][pos_cluster]
                receptacle[i] = torch.ones(num_clips) * label_id

            # Strategy to vote to assign pseudo_labels
            elif count_clustered_samples[i] > 1:

                if Counter(pseudo_labels_test[i].numpy()).most_common(1)[-1][0].item() != -1:
                    dominating_cluster = Counter(pseudo_labels_test[i].numpy()).most_common(1)[-1][0].item()
                else:
                    dominating_cluster = Counter(pseudo_labels_test[i].numpy()).most_common(2)[-1][0].item()

                pos_cluster = (pseudo_labels_test[i] == dominating_cluster).nonzero()[0]
                label_id = pseudo_labels_test[i][pos_cluster].item()
                receptacle[i] = torch.ones(num_clips) * label_id

        return receptacle.view(cluster_id.shape) + source_classes

    if majority_vote:
        pseudo_labels = generate_pseudo_labels_clips(pseudo_labels, num_clips, rerank_dist_jaccard)
        pseudo_labels_tight = generate_pseudo_labels_clips(pseudo_labels_tight, num_clips, rerank_dist_jaccard)
        pseudo_labels_loose = generate_pseudo_labels_clips(pseudo_labels_loose, num_clips, rerank_dist_jaccard)
    else:
        pseudo_labels = generate_pseudo_labels(pseudo_labels, num_ids)
        pseudo_labels_tight = generate_pseudo_labels(pseudo_labels_tight, num_ids_tight)
        pseudo_labels_loose = generate_pseudo_labels(pseudo_labels_loose, num_ids_loose)


    def manage_discontinuous_labels(pseudo_labels_discontinuous):
        # locate where is the issue
        pseudo_labels_final = pseudo_labels_discontinuous - source_classes

        ids_of_discoutinuous_cluster = []
        gap_in_discontinuity = []
        a = list(set((pseudo_labels_discontinuous - source_classes).int().tolist()))
        for i in range(0, len(a)):
            if a[i] - 1 != a[i - 1]: # -> find it is due to discontinuous labels
                ids_of_discoutinuous_cluster.append(a[i-1])
                gap_in_discontinuity.append(a[i] - a[i - 1])
        # Remove 0
        ids_of_discoutinuous_cluster = ids_of_discoutinuous_cluster[1:]
        gap_in_discontinuity = gap_in_discontinuity[1:]

        # Now remove discontinuous labels by substracting by the correspunding gap all ids after them
        for i in range(0, len(ids_of_discoutinuous_cluster)):
            idx = ((pseudo_labels_discontinuous - source_classes) > ids_of_discoutinuous_cluster[i]).nonzero()
            pseudo_labels_final[idx] -= gap_in_discontinuity[i] - 1

        return pseudo_labels_final + source_classes

    pseudo_labels = manage_discontinuous_labels(pseudo_labels_discontinuous=pseudo_labels)
    pseudo_labels_tight = manage_discontinuous_labels(pseudo_labels_discontinuous=pseudo_labels_tight)
    pseudo_labels_loose = manage_discontinuous_labels(pseudo_labels_discontinuous=pseudo_labels_loose)

    # ---------------------------------------------------------------------------------

    # compute R_indep and R_comp
    N = pseudo_labels.size(0)
    label_sim = pseudo_labels.expand(N, N).eq(pseudo_labels.expand(N, N).t()).float()
    label_sim_tight = pseudo_labels_tight.expand(N, N).eq(pseudo_labels_tight.expand(N, N).t()).float()
    label_sim_loose = pseudo_labels_loose.expand(N, N).eq(pseudo_labels_loose.expand(N, N).t()).float()

    R_comp = 1 - torch.min(label_sim, label_sim_tight).sum(-1) / torch.max(label_sim, label_sim_tight).sum(-1)
    R_indep = 1 - torch.min(label_sim, label_sim_loose).sum(-1) / torch.max(label_sim, label_sim_loose).sum(-1)
    assert ((R_comp.min() >= 0) and (R_comp.max() <= 1))
    assert ((R_indep.min() >= 0) and (R_indep.max() <= 1))

    cluster_R_comp, cluster_R_indep = collections.defaultdict(list), collections.defaultdict(list)
    cluster_img_num = collections.defaultdict(int)
    for i, (comp, indep, label) in enumerate(zip(R_comp, R_indep, pseudo_labels)):
        cluster_R_comp[label.item() - source_classes].append(comp.item())
        cluster_R_indep[label.item() - source_classes].append(indep.item())
        cluster_img_num[label.item() - source_classes] += 1

    print(eps, eps_gap)

    cluster_R_comp = [min(cluster_R_comp[i]) for i in sorted(cluster_R_comp.keys())]
    cluster_R_indep = [min(cluster_R_indep[i]) for i in sorted(cluster_R_indep.keys())]
    cluster_R_indep_noins = [iou for iou, num in zip(cluster_R_indep, sorted(cluster_img_num.keys())) if
                             cluster_img_num[num] > 1]

    return cluster_R_comp, cluster_R_indep, cluster_R_indep_noins, pseudo_labels, R_comp


def generate_pseudo_labeled_dataset(
                    epoch, cluster_R_comp, cluster_R_indep, cluster_R_indep_noins, dataset_target, pseudo_labels,
                    source_classes, R_comp, indep_thres, num_clips=1
                    ):

    pseudo_labeled_dataset = []
    outliers = 0

    # --------- ADDED SPECIFICALLY FOR CLIPS) ---------------
    dataset_final = []
    nb_repeat = num_clips
    for i in dataset_target.train:
        for r in range(0, nb_repeat):
            dataset_final.append(i)
    # ------------------------------------------------------

    for i, ((fname, _, cid), label) in enumerate(zip(sorted(dataset_final), pseudo_labels)):
        indep_score = cluster_R_indep[int(label.item() - source_classes)]
        comp_score = R_comp[i]
        if indep_score <= indep_thres and comp_score.item() <= cluster_R_comp[int(label.item()) - source_classes]:
            if i % num_clips == 0:
                pseudo_labeled_dataset.append((fname, label.item(), cid))
        else:
            if i % num_clips == 0:
                pseudo_labeled_dataset.append((fname, source_classes + len(cluster_R_indep) + outliers, cid))
            pseudo_labels[i] = source_classes + len(cluster_R_indep) + outliers
            outliers += 1

    # statistics of clusters and un-clustered instances
    index2label = collections.defaultdict(int)
    for label in pseudo_labels:
        index2label[label.item()] += 1
    index2label = np.fromiter(index2label.values(), dtype=float)
    print('==> Statistics for epoch {}: {} clusters, {} un-clustered instances, R_indep threshold is {}'
          .format(epoch, (index2label > 1).sum(), (index2label == 1).sum(), 1 - indep_thres))

    return pseudo_labeled_dataset, index2label, indep_thres, pseudo_labels


def print_nb_clusters_and_unclusters(eps_list, eps_gap_list, nb_cluster, nb_uncluster, savefig=None):
    fig, ax = plt.subplots()
    for i in range(0, len(eps_gap_list)):
        ax.plot(eps_list, nb_cluster[i], label=str(eps_gap_list[i]))
    plt.legend(loc='best', title='eps_gap')
    plt.xlabel('eps')
    plt.ylabel('nb clusters')
    plt.title("Number of clusters per epsilon.")
    if savefig is not None:
        plt.savefig("clusters_evolution")
    plt.show()
    plt.close()

    fig, ax = plt.subplots()
    for i in range(0, len(eps_gap_list)):
        ax.plot(eps_list, nb_uncluster[i], label=str(eps_gap_list[i]))
    plt.legend(loc='best', title='eps_gap')
    plt.xlabel('eps')
    plt.ylabel('nb samples unclustered')
    plt.title("Number of unclustered samples per epsilon.")
    if savefig is not None:
        plt.savefig(savefig)
    plt.show()
    plt.close()
