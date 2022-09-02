from __future__ import absolute_import

import math
from tools.utils import MetricMeter, AverageMeter, Logger, save_checkpoint, show_distri, T_SNE_computation, T_SNE_computation_Source_Target
import time
import torch
import torch.nn.functional as F


def train_camcla(args,
        epoch, model, model_grl, criterion_xent, criterion_htri, criterion_camcla,
        optimizer, optimizer_grl, trainloader, trainloader_target, use_gpu, writer, balance_loss, max_cam_cla_loss=0,
                 seq_len=4, criterion_clip_clu=None,
                 ):

    losses = MetricMeter()
    batch_xent_loss = AverageMeter()
    batch_htri_loss = AverageMeter()
    batch_camcla_loss = AverageMeter()
    batch_domcla_loss = AverageMeter()
    batch_loss = AverageMeter()
    batch_corrects = AverageMeter()
    batch_time = AverageMeter()
    data_time = AverageMeter()
    batch_camera_corrects = AverageMeter()
    batch_clip_clu_loss = AverageMeter()
    num_clips = 1
    model.train()

    end = time.time()

    gamma = 10
    max_epoch = 500
    p = epoch / max_epoch
    delta = args.max_delta * ((2 / (1 + math.exp(-gamma * p))) - 1)

    for batch_idx, ([vids, pids, camids, imgs_paths, _], [vids_t, pids_t, camids_t, imgs_paths_t, _]) in enumerate(
            zip(trainloader, trainloader_target)):

        if use_gpu:
            vids, pids, camids = vids.cuda(), pids.cuda(), camids.cuda()
            vids_t, pids_t, camids_t = vids_t.cuda(), pids_t.cuda(), camids_t.cuda()

        if vids_t.shape[2] != seq_len:
            assert vids_t.shape[2] % seq_len == 0
            num_clips = int(vids_t.shape[2]/seq_len)
            l_vids = torch.split(vids_t, seq_len, dim=2)
            vids_t = torch.cat(l_vids, dim=0)
            pids_t = torch.cat([pids_t for i in range(num_clips)], dim=0)
            camids_t = torch.cat([camids_t for i in range(num_clips)], dim=0)

        # measure data loading time
        data_time.update(time.time() - end)

        # forward
        outputs, features = model(vids)
        _, features_t = model(vids_t)

        camcla_loss = 0
        clip_clu_loss = torch.tensor(0)
        domcla_loss = torch.tensor(0)
        clip_clu_losses = []

        if args.multihead == 'all_head':   #todo
            for i in range(len(features_t)):
                f = torch.cat((features[i], features_t[i]))
                c = torch.cat((camids, camids_t + 2))#len(dataset.total_camids)))

                camcla_loss_i, domcla_loss_i, preds_camcla = criterion_camcla(
                    grl_model=model_grl, features=f.cpu(), delta=delta, camid=c, multihead=args.multihead
                )
                camcla_loss += camcla_loss_i
                domcla_loss += domcla_loss_i

        else:
            for i in range(len(features_t)):
                input = F.normalize(features_t[i], p=2).cpu()
                camcla_loss_i, _, preds_camcla = criterion_camcla(
                    grl_model=model_grl, features=input, delta=delta, camid=camids_t, multihead=args.multihead
                )
                camcla_loss += camcla_loss_i
                if num_clips != 1 : #If we are using clips, use the clu_loss
                    l_vids = torch.split(features_t[i], int(features_t[i].shape[0]/num_clips), dim=0)
                    l_vids = [ tensor.unsqueeze(1) for tensor in l_vids]

                    clip_clu_losses.append(criterion_clip_clu(torch.cat(l_vids,dim=1)))

        # combine hard triplet loss with cross entropy loss
        encoder_loss = 0
        xent_loss, htri_loss = 0, 0
        for i in range(len(outputs)):
            xent_loss = criterion_xent(outputs[i], pids)
            htri_loss = criterion_htri(features[i], pids)
            if num_clips != 1:
                encoder_loss += xent_loss + htri_loss #+ 0.05 * clip_clu_losses[i] #balance_loss([xent_loss, htri_loss, clip_clu_losses[i]])
            else:
                encoder_loss += balance_loss([xent_loss, htri_loss], epoch=epoch)

        #loss = xent_loss + htri_loss + camcla_loss + encoder_loss
        loss = camcla_loss + encoder_loss
        # backward + optimize

        optimizer.zero_grad()
        optimizer_grl.zero_grad()
        loss.backward()

        optimizer_grl.step()
        optimizer.step()

        # statistics
        _, preds = torch.max(outputs[1].data, 1)

        #Cameras precision
        if args.multihead == 'all_head':
            _, preds_camera = torch.max(preds_camcla.data, 1)
            acc_camcla = torch.sum(  # TODO:The number of source cameras need to be fetched and put in place of the 2
                preds_camera.long().cpu() == torch.cat((camids.detach(), camids_t.detach() +
                                                        len(trainloader.total_camids))).cpu()).float() / pids.size(0)
        if args.multihead == 't_head':
            _, preds_camera = torch.max(preds_camcla.data, 1)
            acc_camcla = torch.sum(preds_camera.long().cpu() ==
                                   camids_t.detach().cpu()).float() / pids.size(0)


        #batch_clip_clu_loss.update(clip_clu_losses[0].item(), pids.size(0))
        batch_corrects.update(torch.sum(preds == pids.data).float() / pids.size(0), pids.size(0))
        batch_camera_corrects.update(acc_camcla, pids.size(0))
        batch_xent_loss.update(xent_loss.item(), pids.size(0))
        batch_htri_loss.update(htri_loss.item(), pids.size(0))
        batch_camcla_loss.update(camcla_loss.item(), pids.size(0))
        batch_domcla_loss.update(domcla_loss.item(), pids.size(0))
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
          'Acc:{acc.avg:.2%} '
          'Lambda:{delta:.4f} '
          'Loss Clu:{clu.avg:.4f} '
          'Acc Camcla:{acc_camcla.avg:.2f}'.format(
        epoch + 1, batch_time=batch_time,
        data_time=data_time, loss=batch_loss,
        xent=batch_xent_loss, htri=batch_htri_loss,
        camcla=batch_camcla_loss, domcla=batch_domcla_loss,
        acc=batch_corrects, delta=delta,clu=batch_clip_clu_loss,acc_camcla=batch_camera_corrects)
    )

    if writer is not None:
        loss_summary = {}
        loss_summary[args.loss_co] = htri_loss.item()
        loss_summary[args.loss_id] = xent_loss.item()
        loss_summary['cam_cla'] = camcla_loss.item()
        loss_summary['dom_cla'] = domcla_loss.item()
        losses.update(loss_summary)
        n_iter = epoch * len(trainloader) + batch_idx
        writer.add_scalar('Train/time', batch_time.avg, n_iter)
        writer.add_scalar('Train/data', data_time.avg, n_iter)
        for name, meter in losses.meters.items():
            writer.add_scalar('Train/' + name, meter.avg, n_iter)