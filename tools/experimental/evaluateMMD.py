import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ['CUDA_VISIBLE_DEVICES'] = "1"
import torch
import argparse
from torch.utils.data import DataLoader
from tools.video_loader import VideoDataset
import transforms.spatial_transforms as ST
import transforms.temporal_transforms as TT
from tools.utils import AverageMeter, Logger, save_checkpoint, show_distri, MMD
import data_manager
from torch.utils.data.sampler import RandomSampler
import models
from torch.nn import functional as F
import numpy as np

################

def handle_TCL_eval_features(feat, model):
    feat_list = torch.split(feat, 2048, dim=1)
    norm_feat_list = []
    for i, f in enumerate(feat_list):
        f = model.bn[i](f)  # [bs, c]
        f = F.normalize(f, p=2, dim=1, eps=1e-12)
        norm_feat_list.append(f)
    feat = torch.cat(norm_feat_list, 1)
    return feat


def run(args):
    dataset_f_1 = data_manager.init_dataset(name=args.dataset_1, root=args.root)
    dataset_f_2 = data_manager.init_dataset(name=args.dataset_2, root=args.root)

    spatial_transform_test = ST.Compose([
        ST.Scale((256, 128), interpolation=3),
        ST.ToTensor(),
        ST.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    temporal_transform_test = TT.TemporalBeginCrop(size=args.test_frames)

    dataset_1 = dataset_f_1.train
    dataset_2 = dataset_f_2.train

    loader_1 = DataLoader(
        VideoDataset(dataset_1, spatial_transform=spatial_transform_test,
                     temporal_transform=temporal_transform_test, seq_len=4,
                     num_clips=1),
        sampler=RandomSampler(dataset_1),
        batch_size=args.batch, num_workers=4,
        pin_memory=False, drop_last=True,
    )

    loader_2 = DataLoader(
        VideoDataset(dataset_2, spatial_transform=spatial_transform_test,
                     temporal_transform=temporal_transform_test, seq_len=4,
                     num_clips=1),
        sampler=RandomSampler(dataset_2),
        batch_size=args.batch, num_workers=4,
        pin_memory=False, drop_last=True,
    )

    model = models.init_model(name=args.arch, use_gpu=True, num_classes=dataset_f_1.num_train_pids,
                              loss={'xent', 'htri'}).cuda()

    if args.resume:
        print("Loading checkpoint from '{}'".format(args.resume))
        checkpoint = torch.load(args.resume)
        # Model
        model.load_state_dict(checkpoint['state_dict'])
        print('Loaded model')

    model.eval()
###########
    with torch.no_grad():

        a_features_1 = []
        a_features_2 = []
        a_cam_1 = []
        a_cam_2 = []
        model.eval()
        for batch_idx, ([vids_1, pids_1, camids_1, path1, _], [vids_2, pids_2, camids_2, path2, _]) in enumerate(
                    zip(loader_1, loader_2)):

            features_1 = model(vids_1.cuda())
            features_2 = model(vids_2.cuda())

            features_1 = handle_TCL_eval_features(features_1, model)
            features_2 = handle_TCL_eval_features(features_2, model)

            a_features_1.append(vids_1.cuda().flatten(1))
            a_features_2.append(vids_2.cuda().flatten(1)) #features_2

            a_cam_1.append(camids_1)
            a_cam_2.append(camids_2)

            paths_2_last = path2


        train_source_features = torch.cat(a_features_1, 0)
        train_target_features = torch.cat(a_features_2, 0)

        camids_1 = torch.cat(a_cam_1, 0).detach().cpu().numpy()
        camids_2 = torch.cat(a_cam_2, 0).detach().cpu().numpy()

        # train_target_features = F.normalize(train_target_features, p=2, dim=1)
        # train_source_features = F.normalize(train_source_features, p=2, dim=1)

        distmat_mmd_2 = np.zeros((len(np.unique(camids_1)),len(np.unique(camids_1))))
        for i in np.unique(camids_1):
            for j in np.unique(camids_1):

                f_ci = train_source_features[torch.from_numpy(camids_1  == i), :]
                f_cj = train_source_features[torch.from_numpy(camids_1 == j), :]

                min_id = min(f_ci.shape[0],f_cj.shape[0])
                distmat_mmd_2[i,j] = MMD(f_ci[:min_id,:],f_cj[:min_id,:],kernel = "multiscale").detach().cpu().numpy()

        mmd_1_2 = MMD(train_target_features, train_source_features, kernel="multiscale").detach().cpu().numpy()

        m_ic_mmd_t = np.mean(distmat_mmd_2[distmat_mmd_2 > 1e-6])
        max_ic_mmd_t = np.amax(distmat_mmd_2[distmat_mmd_2 > 1e-6])
        min_ic_mmd_t = np.amin(distmat_mmd_2[distmat_mmd_2 > 1e-6])

        print(distmat_mmd_2)
        print(" Mean inter-camera MMD Target : {}".format(m_ic_mmd_t))
        print(" Max inter-camera MMD Target : {}".format(max_ic_mmd_t))
        print(" Min inter-camera MMD Target : {}".format(min_ic_mmd_t))
        print("MMD Source <> Target : {}".format(mmd_1_2))

        # print("")
        # print(paths_2_last[-1][0:10])
        # print(camids_2[-50:-40])

###

    # with torch.no_grad():
    #
    #     a_features_1 = []
    #     a_features_2 = []
    #     a_cam_1 = []
    #     a_cam_2 = []
    #     model.eval()
    #     for batch_idx, ([vids_1, pids_1, camids_1, path1, _], [vids_2, pids_2, camids_2, path2, _]) in enumerate(
    #                 zip(loader_1, loader_2)):
    #
    #         features_1 = model(vids_1.cuda())
    #         features_2 = model(vids_2.cuda())
    #
    #         features_1 = handle_TCL_eval_features(features_1, model)
    #         features_2 = handle_TCL_eval_features(features_2, model)
    #
    #         print(vids_2.flatten(1).shape)
    #
    #         a_features_1.append(features_1)
    #         a_features_2.append(vids_2.flatten(1)) #features_2
    #
    #         a_cam_1.append(camids_1)
    #         a_cam_2.append(camids_2)
    #
    #         paths_2_last = path2
    #
    #
    #     train_source_features = torch.cat(a_features_1, 0)
    #     train_target_features = torch.cat(a_features_2, 0)
    #
    #     camids_1 = torch.cat(a_cam_1, 0).detach().cpu().numpy()
    #     camids_2 = torch.cat(a_cam_2, 0).detach().cpu().numpy()
    #
    #     # train_target_features = F.normalize(train_target_features, p=2, dim=1)
    #     # train_source_features = F.normalize(train_source_features, p=2, dim=1)
    #
    #     distmat_mmd_2 = np.zeros((len(np.unique(camids_2)),len(np.unique(camids_2))))
    #     for i in np.unique(camids_2):
    #         for j in np.unique(camids_2):
    #
    #             f_ci = train_source_features[ torch.from_numpy(camids_1  == i), :]
    #             f_cj = train_source_features[torch.from_numpy(camids_1 == j), :]
    #
    #             min_id = min(f_ci.shape[0],f_cj.shape[0])
    #             distmat_mmd_2[i,j] = MMD(f_ci[:min_id,:],f_cj[:min_id,:],kernel = "multiscale").detach().cpu().numpy()
    #
    #     mmd_1_2 = MMD(train_target_features, train_source_features, kernel="multiscale").detach().cpu().numpy()
    #
    #     # m_ic_mmd_t = np.mean(distmat_mmd_2[distmat_mmd_2 > 1e-6])
    #     # max_ic_mmd_t = np.amax(distmat_mmd_2[distmat_mmd_2 > 1e-6])
    #     # min_ic_mmd_t = np.amin(distmat_mmd_2[distmat_mmd_2 > 1e-6])
    #
    #     print(distmat_mmd_2)
    #     # print(" Mean inter-camera MMD Target : {}".format(m_ic_mmd_t))
    #     # print(" Max inter-camera MMD Target : {}".format(max_ic_mmd_t))
    #     # print(" Min inter-camera MMD Target : {}".format(min_ic_mmd_t))
    #     print("MMD Source <> Target : {}".format(mmd_1_2))
    #
    #     # print("")
    #     # print(paths_2_last[-1][0:10])
    #     # print(camids_2[-50:-40])

if __name__ == '__main__':
    ####################

    parser = argparse.ArgumentParser(description='Train TCLNet')
    parser.add_argument('--root', type=str, default='')
    parser.add_argument('-d', '--dataset_1', type=str, default='ilids-vid',
                        choices=data_manager.get_names())
    parser.add_argument('-dt', '--dataset_2', type=str, default='prid2011',
                        choices=data_manager.get_names())
    parser.add_argument('--test_frames', default=4, type=int, help='frames/clip for test')
    parser.add_argument('--batch', default=32, type=int, help="train batch size")
    parser.add_argument('-a', '--arch', type=str, default='TCLNet')
    parser.add_argument('--resume', default="", type=str,)

    args = parser.parse_args()

    run(args)