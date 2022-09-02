from __future__ import absolute_import
import time
import math

import torch
import torch.nn.functional as F

from tools.utils import MetricMeter, AverageMeter


def train_source(epoch, model, criterion_xent, criterion_htri, optimizer, trainloader, use_gpu):
    batch_xent_loss = AverageMeter()
    batch_htri_loss = AverageMeter()
    batch_loss = AverageMeter()
    batch_corrects = AverageMeter()
    batch_time = AverageMeter()
    data_time = AverageMeter()

    model.train()

    end = time.time()
    for batch_idx, (vids, pids, camids, _, _) in enumerate(trainloader):
        if use_gpu:
            vids, pids = vids.cuda(), pids.cuda()

        # measure data loading time
        data_time.update(time.time() - end)

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward
        outputs, features = model(vids)

        # combine hard triplet loss with cross entropy loss
        xent_loss, htri_loss = 0, 0
        for i in range(len(outputs)):
            xent_loss += criterion_xent(outputs[i], pids)
            htri_loss += criterion_htri(features[i], pids)

        loss = xent_loss + htri_loss

        # backward + optimize
        loss.backward()
        optimizer.step()

        # statistics
        _, preds = torch.max(outputs[1].data, 1)
        batch_corrects.update(torch.sum(preds == pids.data).float( ) /pids.size(0), pids.size(0))

        batch_xent_loss.update(xent_loss.item(), pids.size(0))
        batch_htri_loss.update(htri_loss.item(), pids.size(0))
        batch_loss.update(loss.item(), pids.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

    print('Epoch{0} '
          'Time:{batch_time.sum:.1f}s '
          'Data:{data_time.sum:.1f}s '
          'Loss:{loss.avg:.4f} '
          'Xent:{xent.avg:.4f} '
          'Htri:{htri.avg:.4f} '
          'Acc:{acc.avg:.2%} '.format(
        epoch +1, batch_time=batch_time,
        data_time=data_time, loss=batch_loss,
        xent=batch_xent_loss, htri=batch_htri_loss,
        acc=batch_corrects))



def train_da(
        epoch, model, model_grl, optimizer, optimizer_grl, trainloader, trainloader_target, writer,
        criterion_xent=None, criterion_htri=None, criterion_camcla=None, criterion_contr=None,
        use_gpu=True, max_delta=1, hmemory=None, multihead='target', dataset_s='', seq_len=4
            ):

    losses = MetricMeter()
    batch_xent_loss = AverageMeter()
    batch_htri_loss = AverageMeter()
    batch_camcla_loss = AverageMeter()
    batch_domcla_loss = AverageMeter()
    batch_contr_loss = AverageMeter()
    batch_loss = AverageMeter()
    batch_corrects = AverageMeter()
    batch_time = AverageMeter()
    data_time = AverageMeter()
    batch_camera_corrects = AverageMeter()

    model.train()

    end = time.time()

    for batch_idx, ([vids, pids, camids, imgs_paths, index], [vids_t, pids_t, camids_t, imgs_paths_t, index_t]) in\
            enumerate(zip(trainloader, trainloader_target)):

        #print(pids_t)

        if use_gpu:
            vids, pids, camids = vids.cuda(), pids.cuda(), camids.cuda()
            vids_t, pids_t, camids_t, index_t = vids_t.cuda(), pids_t.cuda(), camids_t.cuda(), index_t.cuda()

        if vids_t.shape[2] != seq_len:
            assert vids_t.shape[2] % seq_len == 0
            num_clips = int(vids_t.shape[2]/seq_len)
            l_vids = torch.split(vids_t, seq_len, dim=2)
            vids_t = torch.cat(l_vids, dim=0)
            pids_t = torch.cat([pids_t for i in range(num_clips)], dim=0)
            camids_t = torch.cat([camids_t for i in range(num_clips)], dim=0)
            index_t = torch.cat([index_t for i in range(num_clips)], dim=0)

        # measure data loading time
        data_time.update(time.time() - end)

        # forward
        outputs, features = model(vids)
        _, features_t = model(vids_t)

        # Initialization of losses
        camcla_loss = 0
        domcla_loss = torch.tensor(0)

        if criterion_camcla is None:
            camcla_loss = torch.tensor(0)

        else:
            if multihead == 'all_head':
                domcla_loss = torch.tensor(0)


        contr_loss = 0
        if criterion_contr is None:
            contr_loss = torch.tensor(0)

        xent_loss, htri_loss = 0, 0

        # ------------------- NORMALIZATION OF FEATURES --------------------

        # Target
        norm_feat_list_target = []
        for i, f_t in enumerate(features_t):
            # f_t = model.module.bn[i](f_t)  # [bs, c]
            f_t = F.normalize(f_t, p=2, dim=1, eps=1e-12)
            norm_feat_list_target.append(f_t)
        features_t = torch.cat(norm_feat_list_target, 1)

        # Source
        norm_feat_list_source = []
        for i, f_s in enumerate(features):
            # f_s = model.module.bn[i](f_s)  # [bs, c]
            f_s = F.normalize(f_s, p=2, dim=1, eps=1e-12)
            norm_feat_list_source.append(f_s)
        features_s = torch.cat(norm_feat_list_source, 1)

        # ------------------- SUPERVISED LOSS ON SOURCE --------------------

        for i in range(len(outputs)):

            if criterion_xent is not None:
                xent_loss += criterion_xent(outputs[i], pids)
            if criterion_htri is not None:
                htri_loss += criterion_htri(features[i], pids)

        # ------------------- CONTRASTIVE LOSS ----------------

        if criterion_contr is not None:

            contr_loss += hmemory(features_t, index_t)

        # -------------------CAMERA LOSS ------------------------
        acc_camcla = torch.tensor(0)

        gamma = 10
        max_epoch = 500
        p = epoch / max_epoch
        delta = max_delta * ((2 / (1 + math.exp(-gamma * p))) - 1)

        # delta = max_delta

        if criterion_camcla is not None:

            if multihead == 'all_head':

                f = torch.cat((features_t, features_s))
                c = torch.cat((camids, camids_t + len(dataset_s.total_camids)))

                camcla_loss_i, domcla_loss_i, preds_camcla = criterion_camcla(
                    grl_model=model_grl, features=f.cpu(), delta=delta, camid=c, multihead=multihead
                )

                camcla_loss += camcla_loss_i
                domcla_loss += domcla_loss_i
                #print(camcla_loss)

            else:

                camcla_loss_i, _, preds_camcla = criterion_camcla(
                        grl_model=model_grl, features=features_t.cpu(), delta=delta, camid=camids_t, multihead=multihead
                    )

                camcla_loss += camcla_loss_i

            # Cameras precision
            if multihead == 'all_head':
                _, preds_camera = torch.max(preds_camcla.data, 1)
                acc_camcla = torch.sum(
                    preds_camera.long().cpu() == torch.cat((camids.detach(), camids_t.detach() +
                                                            len(dataset_s.total_camids))).cpu()).float() / pids.size(0)
            else:
                _, preds_camera = torch.max(preds_camcla.data, 1)
                acc_camcla = torch.sum(preds_camera.long().cpu() ==
                                       camids_t.detach().cpu()).float() / pids.size(0)

        # ------------------ END LOSS -----------------------

        loss = xent_loss + htri_loss + camcla_loss + contr_loss

        # ------------------ BACKWARD AND OPTIMIZERS -------------------
        optimizer.zero_grad()
        optimizer_grl.zero_grad()

        loss.backward()

        optimizer_grl.step()
        optimizer.step()
        # ------------------------------------------------------------
        # statistics
        _, preds = torch.max(outputs[1].data, 1)

        batch_corrects.update(torch.sum(preds == pids.data).float() / pids.size(0), pids.size(0))
        batch_camera_corrects.update(acc_camcla, pids.size(0))
        batch_xent_loss.update(xent_loss.item(), pids.size(0))
        batch_htri_loss.update(htri_loss.item(), pids.size(0))
        batch_camcla_loss.update(camcla_loss.item(), pids.size(0))
        batch_domcla_loss.update(domcla_loss.item(), pids.size(0))
        batch_contr_loss.update(contr_loss.item(), pids.size(0))
        batch_loss.update(loss.item(), pids.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

    print('Epoch{0} '
          'Time:{batch_time.sum:.1f}s '
          'Data:{data_time.sum:.1f}s '
          'Loss:{loss.avg:.4f} '
          'Xent:{xent.avg:.4f} '
          'Htri:{htri.avg:.4f} '
          'Camcla:{camcla.avg:.4f} '
          'Domcla:{domcla.avg:.4f} '
          'Contr:{contr.avg:.4f} '
          'Acc:{acc.avg:.2%} '
          'Acc_Cam:{acc_camcla.avg:.2%} '
          'Lambda:{delta:.4f} '.format(
        epoch + 1, batch_time=batch_time,
        data_time=data_time, loss=batch_loss,
        xent=batch_xent_loss, htri=batch_htri_loss,
        camcla=batch_camcla_loss, domcla=batch_domcla_loss, contr=batch_contr_loss,
        acc=batch_corrects, acc_camcla=batch_camera_corrects, delta=delta)
    )

    if writer is not None:
        loss_summary = {}
        loss_summary['htri'] = htri_loss.item()
        loss_summary['ce_loss'] = xent_loss.item()
        loss_summary['cam_cla'] = camcla_loss.item()
        loss_summary['dom_cla'] = domcla_loss.item()
        loss_summary['contr'] = contr_loss.item()
        loss_summary['acc_cam_cla'] = acc_camcla
        losses.update(loss_summary)
        n_iter = epoch * len(trainloader) + batch_idx
        writer.add_scalar('Train/time', batch_time.avg, n_iter)
        writer.add_scalar('Train/data', data_time.avg, n_iter)
        for name, meter in losses.meters.items():
            writer.add_scalar('Train/' + name, meter.avg, n_iter)


def train_target(
        epoch, model, model_grl, criterion_camcla, criterion_contr,
        optimizer, optimizer_grl, trainloader_target, use_gpu, writer, hmemory, multihead, max_delta, seq_len=4
            ):
    losses = MetricMeter()
    batch_camcla_loss = AverageMeter()
    batch_contr_loss = AverageMeter()
    batch_loss = AverageMeter()
    batch_time = AverageMeter()
    data_time = AverageMeter()

    model.train()

    end = time.time()

    for batch_idx, (vids_t, pids_t, camids_t, imgs_paths_t, index_t) in enumerate(trainloader_target):

        if use_gpu:
            vids_t, pids_t, camids_t, index_t = vids_t.cuda(), pids_t.cuda(), camids_t.cuda(), index_t.cuda()

        if vids_t.shape[2] != seq_len:
            assert vids_t.shape[2] % seq_len == 0
            num_clips = int(vids_t.shape[2]/seq_len)
            l_vids = torch.split(vids_t, seq_len, dim=2)
            vids_t = torch.cat(l_vids, dim=0)
            pids_t = torch.cat([pids_t for i in range(num_clips)], dim=0)
            camids_t = torch.cat([camids_t for i in range(num_clips)], dim=0)

        pids = pids_t

        # measure data loading time
        data_time.update(time.time() - end)

        # forward
        _, features_t = model(vids_t)

        contr_loss = 0
        camcla_loss = 0

        # ------------------- CONTRASTIVE LOSS ----------------

        if criterion_contr is not None:

            norm_feat_list = []
            for i, f in enumerate(features_t):
                f = model.module.bn[i](f)  # [bs, c]
                f = F.normalize(f, p=2, dim=1, eps=1e-12)
                norm_feat_list.append(f)
            feat = torch.cat(norm_feat_list, 1)
            contr_loss += hmemory(feat, index_t)

        # -------------------CAMERA LOSS ------------------------

        if criterion_camcla is not None:

            gamma = 10
            max_epoch = 500
            p = epoch / max_epoch
            delta = max_delta * ((2 / (1 + math.exp(-gamma * p))) - 1)

            for i in range(len(features_t)):
                input = F.normalize(features_t[i], p=1).cpu()

                camcla_loss_i, _, preds_camcla = criterion_camcla(
                    grl_model=model_grl, features=input, delta=delta, camid=camids_t, multihead=multihead
                )

                camcla_loss += camcla_loss_i

        # ------------------ BACKWARD AND OPTIMIZERS -------------------

        loss = contr_loss + max_delta*camcla_loss

        optimizer.zero_grad()
        optimizer_grl.zero_grad()

        loss.backward()

        if criterion_camcla is not None:
            optimizer_grl.step()
        optimizer.step()
        # ------------------------------------------------------------

        batch_contr_loss.update(contr_loss.item(), pids.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

    print('Epoch{0} '
          'Time:{batch_time.sum:.1f}s '
          'Data:{data_time.sum:.1f}s '
          'Contr:{contr.avg:.4f} '
          'Camcla:{camcla.avg:.4f} '.format(
        epoch + 1, batch_time=batch_time,
        data_time=data_time, contr=batch_contr_loss, camcla=batch_camcla_loss)
    )

    if writer is not None:
        loss_summary = {}
        loss_summary['contr_loss'] = contr_loss.item()
        losses.update(loss_summary)
        n_iter = epoch * len(trainloader_target) + batch_idx
        writer.add_scalar('Train/time', batch_time.avg, n_iter)
        writer.add_scalar('Train/data', data_time.avg, n_iter)
        for name, meter in losses.meters.items():
            writer.add_scalar('Train/' + name, meter.avg, n_iter)
