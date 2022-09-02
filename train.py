from __future__ import print_function, absolute_import

import os
seed = 1
import random
random.seed(seed)
import numpy as np
np.random.seed(seed)
import torch
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

import sys
import time
import datetime
import argparse
import os.path as osp
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim import lr_scheduler
import models
import data_manager
import transforms.spatial_transforms as ST
import transforms.temporal_transforms as TT
import csv
import numpy as np

from tools.video_loader import VideoDataset
from tools.losses import TripletLoss, DomainClassifierLoss, SupervisedContrastiveLoss, CameraLevelConfusionLoss, SupConLoss
from tools.utils import AverageMeter, Logger, save_checkpoint, show_distri, T_SNE_computation, T_SNE_computation_Source_Target, extract_features_for_clustering
from tools.samplers import RandomIdentitySampler, RandomDomainSampler, RandomMultipleGallerySampler
from tools.rerank import compute_jaccard_distance
from tools.pseudo_labels import generate_labels_criterion, generate_pseudo_labeled_dataset, print_nb_clusters_and_unclusters
from tools.train import train_da, train_source, train_target
from tools.test import test

from torch.utils.data.sampler import RandomSampler
from torch.utils.tensorboard import SummaryWriter
from models.hm import HybridMemory
from tools import transforms as T

from pathlib import Path

torch.multiprocessing.set_sharing_strategy('file_system')

parser = argparse.ArgumentParser(description='Train TCLNet')
# Datasets
parser.add_argument('--root', type=Path, default=Path('/export/livia/home/vision/dmekhazni/KD-Reid/reid-data'))
parser.add_argument('-d', '--dataset', type=str, default='mars', choices=data_manager.get_names())
parser.add_argument('-dt', '--dataset_target', type=str, default='ilidsvid', choices=data_manager.get_names())
parser.add_argument('-j', '--workers', default=0, type=int, help="number of data loading workers (default: 4)")
parser.add_argument('--height', type=int, default=256, help="height of an image (default: 256)")
parser.add_argument('--width', type=int, default=128, help="width of an image (default: 128)")

# Augment
parser.add_argument('--seq_len', type=int, default=4, help="number of images to sample in a tracklet")
parser.add_argument('--sample_stride', type=int, default=8, help="stride of images to sample in a tracklet")
parser.add_argument('--test_frames', default=4, type=int, help='frames/clip for test')

# Optimization options
parser.add_argument('--max_epoch', default=150, type=int, help="maximum epochs to run")
parser.add_argument('--start_epoch', default=0, type=int, help="manual epoch number (useful on restarts)")
parser.add_argument('--train_batch', default=24, type=int, help="train batch size")
parser.add_argument('--test_batch', default=32, type=int, help="has to be 1")
parser.add_argument('--lr', '--learning-rate', default=0.0003, type=float,
                    help="initial learning rate, use 0.0001 for rnn, use 0.0003 for pooling and attention")
parser.add_argument('--stepsize', default=[40, 80, 120], nargs='+', type=int, help="stepsize to decay learning rate")
parser.add_argument('--gamma', default=0.1, type=float, help="learning rate decay")
parser.add_argument('--weight_decay', default=5e-04, type=float, help="weight decay (default: 5e-04)")
parser.add_argument('--margin', type=float, default=0.4, help="margin for triplet loss")
parser.add_argument('--max_delta', default='0.2', type=float, help='Max value for delta parameter')
parser.add_argument('--eps', default=0.5, type=float, help='value of eps')
parser.add_argument('--eps_gap', default=0.02, type=float, help='value of eps_gap')

parser.add_argument('--num_instances', type=int, default=4, help="number of instances per identity")
parser.add_argument('--distance', type=str, default='consine', help="euclidean or consine")
parser.add_argument('--loss_co', type=str, default='triplet')
parser.add_argument('--loss_id', type=str, default='ce')
parser.add_argument('--loss_camcla', type=str, default='clc')
parser.add_argument('--loss_contr', type=str, default='contr', help="put contr to activate and no_contr to deactivate")

# Architecture
parser.add_argument('-a', '--arch', type=str, default='TCLNet')
parser.add_argument('--resume', type=str, default='', metavar='PATH')

# Miscs
parser.add_argument('--seed', type=int, default=1, help="manual seed")
parser.add_argument('--eval_step', type=int, default=3, help="run eval for every N epochs (set to -1 test after train)")
parser.add_argument('--start_eval', type=int, default=0, help="start to evaluate after specific epoch")
parser.add_argument('--use_cpu', action='store_true', help="use cpu")
parser.add_argument('--gpu_devices', default='3', type=str, help='gpu device ids for CUDA_VISIBLE_DEVICES')

parser.add_argument('--experiment_name', type=str, default='')
parser.add_argument('--multihead', default='all_head', type=str, help='Camera classifier on both Source and Target Camera'
                                                         'Must be t_head or all_head')
parser.add_argument('--domains', default='source_target', type=str, help='source or source_target or target')

parser.add_argument('--num_clips', default='2', type=int)
parser.add_argument('--sampling_target', default='RandomMultipleGallerySampler', type=str)
parser.add_argument('--epoch_record_perf', default=40, type=int)
#parser.add_argument('--weighting_criterion', default='static', type=str)
parser.add_argument('--majority_vote', action='store_true', help="majority vote for pseudo-labels assoc with clips")
parser.add_argument('--weight', action='store_true', help="add weight for self pace policy")
parser.add_argument('--test_init', action='store_true', help="test initialisation before the start of training")

parser.add_argument('--force_zero', action='store_true', help="force 0 jaccard distances of clips from same tracklet")

parser.add_argument('--strat_gamma', type=str, default='kNN')

args = parser.parse_args()

def main():

    def seed_worker(worker_id):
        worker_seed = torch.initial_seed() % 2 ** 32
        np.random.seed(worker_seed)
        random.seed(worker_seed)

    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_devices
    use_gpu = torch.cuda.is_available()
    if args.use_cpu: use_gpu = False

    experiment_params = ["ce", args.loss_co, args.loss_camcla]

    assert not (len(set([args.dataset, args.dataset_target])) > 1) & ((args.domains == 'source') | (args.domains == 'target')), "Cannot have -d and -dt different when --domains is set at source_target"
    assert not (len(set([args.dataset, args.dataset_target])) == 1) & ((args.domains == 'source_target')), "Set -d and -dt different to perform UDA when --domains is set at source_target"

    if not args.weight:
        args.strat_gamma = "None"

    exp_name = "log/"
    save_dir = osp.join(exp_name, "_to_".join([args.dataset, args.dataset_target, args.arch]),
                        args.loss_camcla + "_" + args.loss_contr, "_b" + str(args.train_batch) + "_eps"
                        + str(args.eps) + "_clips" + str(args.num_clips)
                        + "_weight=" + str(args.weight) + "_strat_gamma=" + args.strat_gamma + "_" + args.experiment_name
                        )

    print(save_dir)
    sys.stdout = Logger(osp.join(save_dir, 'log_train.txt'), mode='a')
    print("==========\nArgs:{}\n==========".format(args))

    writer = SummaryWriter(log_dir=save_dir)

    if use_gpu:
        print("Currently using GPU {}".format(args.gpu_devices))
        torch.cuda.manual_seed_all(args.seed)
    else:
        print("Currently using CPU (GPU is highly recommended)")

    print("Initializing dataset {}".format(args.dataset))
    dataset = data_manager.init_dataset(name=args.dataset, root=args.root)
    dataset_target = data_manager.init_dataset(name=args.dataset_target, root=args.root)

    # Data augmentation
    spatial_transform_train = [
        ST.Scale((args.height, args.width), interpolation=3),
        ST.RandomHorizontalFlip(),
        ST.ToTensor(),
        ST.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]
    spatial_transform_train = ST.Compose(spatial_transform_train)

    temporal_transform_train = TT.TemporalRandomCrop(size=args.seq_len, stride=args.sample_stride)

    spatial_transform_test = ST.Compose([
        ST.Scale((args.height, args.width), interpolation=3),
        ST.ToTensor(),
        ST.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    temporal_transform_test = TT.TemporalBeginCrop(size=args.test_frames)

    pin_memory = False

# ------------------------------------ DATAMANAGER ------------------------------------
    # Source

    dataset_train = dataset.train
    if args.dataset == 'duke':
        dataset_train = dataset.train_dense
        print('process duke dataset')

    trainloader = DataLoader(
        VideoDataset(dataset_train, spatial_transform=spatial_transform_train,
                     temporal_transform=temporal_transform_train, num_clips=1, seq_len=args.seq_len),
        sampler=RandomMultipleGallerySampler(dataset_train, num_instances=args.num_instances),
        batch_size=int(args.train_batch * args.num_clips), num_workers=args.workers,
        pin_memory=pin_memory, drop_last=True, worker_init_fn=seed_worker
    )

    queryloader = DataLoader(
        VideoDataset(dataset.query, spatial_transform=spatial_transform_test,
                     temporal_transform=temporal_transform_test),
        batch_size=args.test_batch, shuffle=False, num_workers=args.workers,
        pin_memory=pin_memory, drop_last=False
    )

    galleryloader = DataLoader(
        VideoDataset(dataset.gallery, spatial_transform=spatial_transform_test,
                     temporal_transform=temporal_transform_test),
        batch_size=args.test_batch, shuffle=False, num_workers=args.workers,
        pin_memory=pin_memory, drop_last=False
    )

    trainloader_tsne = DataLoader(
        VideoDataset(dataset_train, spatial_transform=spatial_transform_train,
                     temporal_transform=temporal_transform_train),
        sampler=RandomSampler(dataset_train),
        batch_size=args.train_batch, num_workers=args.workers,
        pin_memory=pin_memory, drop_last=True,
    )

    # Target

    dataset_train_target = dataset_target.train
    if args.dataset_target == 'duke':
        dataset_train_target = dataset_target.train_dense
        print('process duke dataset_target')

    queryloader_target = DataLoader(
        VideoDataset(dataset_target.query, spatial_transform=spatial_transform_test,
                     temporal_transform=temporal_transform_test),
        batch_size=args.test_batch, shuffle=False, num_workers=args.workers,
        pin_memory=pin_memory, drop_last=False
    )

    galleryloader_target = DataLoader(
        VideoDataset(dataset_target.gallery, spatial_transform=spatial_transform_test,
                     temporal_transform=temporal_transform_test),
        batch_size=args.test_batch, shuffle=False, num_workers=args.workers,
        pin_memory=pin_memory, drop_last=False
    )

    trainloader_target = DataLoader(
        VideoDataset(dataset_train_target, spatial_transform=spatial_transform_train,
                     temporal_transform=temporal_transform_train, sample_method="evenly", seq_len=args.seq_len,
                     num_clips=args.num_clips),
        sampler=RandomIdentitySampler(dataset_train_target, num_instances=4),
        batch_size=int(args.train_batch / args.num_clips), num_workers=args.workers,  #
        pin_memory=pin_memory, drop_last=True,
    )

    trainloader_target_tsne = DataLoader(
        VideoDataset(dataset_train_target, spatial_transform=spatial_transform_train,
                     temporal_transform=temporal_transform_train, sample_method="evenly", seq_len=args.seq_len,
                     num_clips=args.num_clips),
        batch_size=args.train_batch, num_workers=args.workers,
        pin_memory=pin_memory, drop_last=False, worker_init_fn=seed_worker
    )
    print("Majority vote: ", str(args.majority_vote))
    # -------------------------------------------------------------------------------------

    # --------------------------------- MODELs & Optimizers -------------------------------
    print("Initializing model: {}".format(args.arch))
    model = models.init_model(name=args.arch, use_gpu=use_gpu, num_classes=dataset.num_train_pids,
                              loss={'xent', 'htri'})

    if args.multihead == "all_head":
        num_cameras = len(dataset.total_camids) + len(dataset_target.total_camids)
        print("NUM CAMERAS: ", num_cameras)
        print(len(dataset.total_camids))
        print(len(dataset_target.total_camids))
        model_grl = models.init_model(name='grl', num_classes=num_cameras)
    else:
        print("NUM CAMERAS: ", dataset_target.total_camids)
        model_grl = models.init_model(name='grl', num_classes=len(dataset_target.total_camids))

    if use_gpu:
        model = model.cuda()
        model_grl = model_grl.cuda()

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    optimizer_grl = torch.optim.Adam(model_grl.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=args.stepsize, gamma=args.gamma)
    start_epoch = args.start_epoch

    if args.resume:
        print("Loading checkpoint from '{}'".format(args.resume))
        checkpoint = torch.load(args.resume)
        # Model
        model.load_state_dict(checkpoint['state_dict'])
        print('Loaded model')
        if model_grl is not None and 'model_grl' in checkpoint.keys():
            model_grl.load_state_dict(checkpoint["state_dict_grl"])
            print('Loaded model_grl')
        # Optimizer
        if optimizer is not None and 'optimizer' in checkpoint.keys():
            optimizer.load_state_dict(checkpoint['optimizer'])
            print('Loaded optimizer')
        if optimizer_grl is not None and 'optimizer_grl' in checkpoint.keys():
            optimizer_grl.load_state_dict(checkpoint['optimizer_grl'])
            print('Loaded optimizer_grl')

    if use_gpu:
        model = nn.DataParallel(model).cuda()
        model_grl = nn.DataParallel(model_grl).cuda()
    # -------------------------------------------------------------------------------------
    # ------------------------------------ HYBRID MEMORY ----------------------------------
    #args.loss_contr = None


    # Create hybrid memory
    memory = HybridMemory(4096, int(len(dataset_target.train) * args.num_clips), temp=0.05, momentum=0.2,
                          weighting=args.weight, strat_gamma=args.strat_gamma).cuda()

    if args.loss_contr != None:

        with torch.no_grad():
            ttf_t, ttf_pids_t, ttf_camids_t, ttf_imgs_t = extract_features_for_clustering(
                model=model, loader=trainloader_target_tsne, use_gpu=use_gpu, seq_len=args.seq_len, hm=True
            )

        # Initialize target-domain instance features
        print("==> Initialize instance features in the hybrid memory")
        features_t = ttf_t
        memory.features = features_t.cuda()
        del features_t

    # -------------------------------------------------------------------------------------

    # --------------------------------------- LOSSES --------------------------------------

    criterion_xent = nn.CrossEntropyLoss()

    AVAI_LOSS_CO = ['contrastive', 'triplet']
    if args.loss_co not in AVAI_LOSS_CO:
        raise ValueError(
            'Unsupported loss: {}. Must be one of {}'.format(
                args.loss_co, AVAI_LOSS_CO
            )
        )

    if args.loss_co == 'triplet':
        criterion_htri = TripletLoss(margin=args.margin, distance=args.distance, use_gpu=use_gpu)
        print("triplet")
    elif args.loss_co == 'contrastive':
        criterion_htri = SupervisedContrastiveLoss()
        print("contrastive")

    AVAI_LOSS_CAMCLA = ['no_camcla', 'ce', 'uce', 'buce', 'clc']
    if args.loss_camcla not in AVAI_LOSS_CAMCLA:
        raise ValueError(
            'Unsupported loss: {}. Must be one of {}'.format(
                args.loss_camcla, AVAI_LOSS_CAMCLA
            )
        )
    if args.loss_camcla == 'no_camcla':
        criterion_camcla = None
        print("No camcla")
    if args.loss_camcla == 'ce':
        criterion_camcla = DomainClassifierLoss()
        print("ce")
    elif args.loss_camcla == "clc":
        criterion_camcla = CameraLevelConfusionLoss()
        print("clc")

    criterion_contr = None
    if args.loss_contr == "contr":
        criterion_contr = SupConLoss()
    # -------------------------------------------------------------------------------------

    start_time = time.time()
    train_time = 0
    best_rank1 = -np.inf
    best_epoch = 0

    # --------------------------- TEST BASELINE --------------------------------
    if args.test_init:
        print("Test Baseline on Source")
        with torch.no_grad():
            rank1 = test(model, queryloader, galleryloader, use_gpu, is_target=False, writer=writer, epoch=0, name_dataset=args.dataset)

        print("Test Baseline on Target")
        with torch.no_grad():
            rank1 = test(model, queryloader_target, galleryloader_target, use_gpu, is_target=True, writer=writer, epoch=0, name_dataset=args.dataset_target)
    # --------------------------------------------------------------------------

    print("==> Start training")

    indep_thres = 0

    for epoch in range(start_epoch, args.max_epoch):

        # ------------------------------------ PSEUDO LABELS -----------------------------------------------

        if args.loss_contr != None:
            source_classes = int(len(trainloader.dataset) / 2)

            eps = args.eps
            eps_gap = args.eps_gap

            features = memory.features.clone()
            rerank_dist_jaccard = compute_jaccard_distance(features, k1=7, k2=3, print_flag=True, search_option=0,
                                                           use_float16=False)

            if args.force_zero:

                nb_clips = args.num_clips
                mask = torch.zeros(nb_clips, nb_clips)

                for i in range(0, rerank_dist_jaccard.shape[0], nb_clips):
                    rerank_dist_jaccard[i:i+nb_clips, i:i+nb_clips] = mask

            del features

            cluster_R_comp, cluster_R_indep, cluster_R_indep_noins, pseudo_labels, R_comp = \
                generate_labels_criterion(rerank_dist_jaccard=rerank_dist_jaccard, source_classes=source_classes,
                                          dataset_target=dataset_target, eps=eps, eps_gap=eps_gap, num_clips=args.num_clips,
                                          majority_vote=args.majority_vote)

            if epoch == 0:
                indep_thres = np.sort(cluster_R_indep_noins)[
                    min(len(cluster_R_indep_noins) - 1, np.round(len(cluster_R_indep_noins) * 0.9).astype('int'))]

            pseudo_labeled_dataset, index2label, indep_thres, pseudo_labels = generate_pseudo_labeled_dataset(
                epoch, cluster_R_comp, cluster_R_indep, cluster_R_indep_noins, dataset_target, pseudo_labels,
                source_classes, R_comp, indep_thres, num_clips=args.num_clips
            )

            memory.labels = pseudo_labels.cuda()

            if args.sampling_target == "RandomSampler":
                print("Sampling Target randomly")
                sampler_target = RandomSampler(pseudo_labeled_dataset)
            elif args.sampling_target == "RandomIdentitySampler":
                sampler_target = RandomIdentitySampler(dataset_train_target, num_instances=args.num_instances)
            elif args.sampling_target == "RandomMultipleGallerySampler":
                sampler_target = RandomMultipleGallerySampler(dataset_train_target, num_instances=args.num_instances)

            train_loader_target_pseudo_labels = \
                DataLoader(
                    VideoDataset(pseudo_labeled_dataset, spatial_transform=spatial_transform_train,
                    temporal_transform=temporal_transform_train, num_clips=args.num_clips, seq_len=args.seq_len),
                    sampler=sampler_target,
                    batch_size=args.train_batch, num_workers=args.workers, pin_memory=pin_memory, drop_last=True,
                    worker_init_fn=seed_worker
                )
        else:
            train_loader_target_pseudo_labels = trainloader_target

        # ----------------------------------------------------------------------------------------------

        # Start batch training

        start_train_time = time.time()

        if args.domains == "source":

            train_source(
                epoch=epoch, model=model, criterion_xent=criterion_xent, criterion_htri=criterion_htri,
                optimizer=optimizer, trainloader=trainloader, use_gpu=use_gpu
            )

        elif args.domains == "source_target":

            train_da(
                epoch=epoch, model=model, model_grl=model_grl, optimizer=optimizer, optimizer_grl=optimizer_grl,
                criterion_xent=criterion_xent, criterion_htri=criterion_htri, criterion_camcla=criterion_camcla,
                criterion_contr=criterion_contr, hmemory=memory,
                trainloader=trainloader, trainloader_target=train_loader_target_pseudo_labels,
                use_gpu=use_gpu, writer=writer, multihead=args.multihead, max_delta=args.max_delta, dataset_s=dataset,
                seq_len=args.seq_len
            )

        elif args.domains == "target":

            train_target(
                epoch=epoch, model=model, model_grl=model_grl, optimizer=optimizer, optimizer_grl=optimizer_grl,
                criterion_camcla=criterion_camcla, criterion_contr=criterion_contr, hmemory=memory,
                trainloader_target=train_loader_target_pseudo_labels,
                use_gpu=use_gpu, writer=writer, max_delta=args.max_delta, multihead=args.multihead, seq_len=args.seq_len
            )

        train_time += round(time.time() - start_train_time)

        scheduler.step()

        if (epoch + 1) >= args.start_eval and (epoch + 1) % args.eval_step == 0 or epoch == 0:

            if args.dataset_target == 'mars' and epoch < 15:
                pass

            print("==> Test Target")
            with torch.no_grad():
                rank1_target = test(model, queryloader_target, galleryloader_target, use_gpu, is_target=True,
                                    ranks=[1, 5], writer=writer, epoch=epoch+1, name_dataset=args.dataset_target)

            is_best = rank1_target > best_rank1
            if is_best:
                best_rank1 = rank1_target
                best_epoch = epoch + 1

                if use_gpu:
                    state_dict = model.module.state_dict()
                    state_dict_grl = model_grl.module.state_dict()
                    optimizer_state = optimizer.state_dict()
                    optimizer_grl_state = optimizer_grl.state_dict()

                else:
                    state_dict = model.state_dict()
                    state_dict_grl = model_grl.state_dict()
                    optimizer_state = optimizer.state_dict()
                    optimizer_grl_state = optimizer_grl.state_dict()

                """save_checkpoint({
                    'state_dict': state_dict,
                    'state_dict_grl': state_dict_grl,
                    'rank1': rank1_target,
                    'epoch': epoch,
                    'optimizer': optimizer_state,
                    'optimizer_grl': optimizer_grl_state
                }, fpath=osp.join(save_dir, 'best_model' + '.pth.tar'))"""

    print("==> Best Rank-1 {:.1%}, achieved at epoch {}".format(best_rank1, best_epoch))

    # CSV storage

    name_experiment = "_b=" + str(args.train_batch) + "_Vote=" + str(args.majority_vote) + "_Weight=" + str(args.weight)

    name_file = exp_name + "_to_".join([args.dataset, args.dataset_target, args.arch])
    header = ['Experiments setup', 'Rank-1', 'Best Epoch', 'Contr', 'Camcla', 'Clips', 'Eps', 'Strat Gamma', 'Init']
    data = [name_experiment, best_rank1, best_epoch, args.loss_contr, args.loss_camcla, args.num_clips, args.eps,
            args.strat_gamma, args.resume]

    with open(name_file + '.csv', 'a', encoding='UTF8', newline='') as f:
        writer_csv = csv.writer(f)
        # write the header
        writer_csv.writerow(header)
        # write multiple rows
        writer_csv.writerows([data])

    elapsed = round(time.time() - start_time)
    elapsed = str(datetime.timedelta(seconds=elapsed))
    train_time = str(datetime.timedelta(seconds=train_time))
    print("Finished. Total elapsed time (h:m:s): {}. Training time (h:m:s): {}.".format(elapsed, train_time))
    if writer is not None:
        writer.close()

    save_checkpoint({
        'state_dict': state_dict,
        'state_dict_grl': state_dict_grl,
        'rank1': rank1_target,
        'epoch': epoch,
        'optimizer': optimizer_state,
        'optimizer_grl': optimizer_grl_state
    }, fpath=osp.join(save_dir, 'best_model' + '.pth.tar'))


if __name__ == '__main__':
    main()
    print("")
    print("--------------------------")
    print("")
