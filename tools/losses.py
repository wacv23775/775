from __future__ import absolute_import

import torch
from torch import nn
from torch.autograd import Variable
from torch.nn import functional as F


"""
Shorthands for loss:
- CrossEntropyLabelSmooth: xent
- TripletLoss: htri
"""


class TripletLoss(nn.Module):
    """Triplet loss with hard positive/negative mining.

    Reference:
    Hermans et al. In Defense of the Triplet Loss for Person Re-Identification. arXiv:1703.07737.

    Code imported from https://github.com/Cysu/open-reid/blob/master/reid/loss/triplet.py.

    Args:
        margin (float): margin for triplet.
    """
    def __init__(self, margin=0.3, distance='consine', use_gpu=True):
        super(TripletLoss, self).__init__()
        if distance not in ['euclidean', 'consine']:
            raise KeyError("Unsupported distance: {}".format(distance))
        self.distance = distance
        self.margin = margin
        self.use_gpu = use_gpu
        self.ranking_loss = nn.MarginRankingLoss(margin=margin)

    def forward(self, inputs, targets):
        """
        Args:
            inputs: feature matrix with shape (batch_size, feat_dim)
            targets: ground truth labels with shape (num_classes)
        """
        n = inputs.size(0)

        # Compute pairwise distance, replace by the official when merged
        if self.distance == 'euclidean':
            dist = torch.pow(inputs, 2).sum(dim=1, keepdim=True).expand(n, n)
            dist = dist + dist.t()
            dist.addmm_(1, -2, inputs, inputs.t())
            dist = dist.clamp(min=1e-12).sqrt()  # for numerical stability
        elif self.distance == 'consine':
            fnorm = torch.norm(inputs, p=2, dim=1, keepdim=True)
            l2norm = inputs.div(fnorm.expand_as(inputs))
            dist = - torch.mm(l2norm, l2norm.t())

        if self.use_gpu: targets = targets.cuda()
        # For each anchor, find the hardest positive and negative
        mask = targets.expand(n, n).eq(targets.expand(n, n).t())
        dist_ap, dist_an = [], []
        for i in range(n):
            dist_ap.append(dist[i][mask[i]].max().unsqueeze(0))
            dist_an.append(dist[i][mask[i] == 0].min().unsqueeze(0))
        dist_ap = torch.cat(dist_ap)
        dist_an = torch.cat(dist_an)

        # Compute ranking hinge loss
        y = torch.ones_like(dist_an)
        loss = self.ranking_loss(dist_an, dist_ap, y)
        return loss


class DomainClassifierLoss(nn.Module):

    def __init__(self, use_gpu=True):
        super(DomainClassifierLoss, self).__init__()
        self.use_gpu = use_gpu
        self.loss_fn = nn.CrossEntropyLoss()

    def forward(self, features, grl_model, source=True, delta=1, camid=None, multihead='t_head'):

        domain_pred = grl_model(features, delta)
        loss_dom_c = 0

        if camid is not None:
            l_cam_c = self.loss_fn(domain_pred, camid.cuda())

            if multihead == 'all_head':

                # Domain based
                soft_layer = nn.Softmax(dim=1)
                domain_pred2 = soft_layer(domain_pred)
                loss_criterion = nn.BCELoss()  # Source or Target
                # Extract only predictions that it is target on both Source and Target samples
                prob_sum_target = domain_pred2[:, 2:].sum(dim=1)
                prob_sum_target = torch.clamp(prob_sum_target, min=1e-6, max=0.999999)
                # define the labels we intend to receive
                ns_per_dom = int(features.shape[0]/2)
                domain_label = torch.cat((torch.zeros(ns_per_dom), torch.ones(ns_per_dom))).int()
                loss_dom_c = loss_criterion(prob_sum_target, domain_label.float().cuda())

        return l_cam_c, loss_dom_c, domain_pred


class SupConLoss(nn.Module):
    """Supervised Contrastive Learning: https://arxiv.org/pdf/2004.11362.pdf.
    It also supports the unsupervised contrastive loss in SimCLR"""
    def __init__(self, temperature=0.07, contrast_mode='all',
                 base_temperature=0.07):
        super(SupConLoss, self).__init__()
        self.temperature = temperature
        self.contrast_mode = contrast_mode
        self.base_temperature = base_temperature

    def forward(self, features, labels=None, mask=None, confidence=None):
        """Compute loss for model. If both `labels` and `mask` are None,
        it degenerates to SimCLR unsupervised loss:
        https://arxiv.org/pdf/2002.05709.pdf

        Args:
            features: hidden vector of shape [bsz, n_views, ...].
            labels: ground truth of shape [bsz].
            mask: contrastive mask of shape [bsz, bsz], mask_{i,j}=1 if sample j
                has the same class as sample i. Can be asymmetric.
        Returns:
            A loss scalar.
        """
        device = (torch.device('cuda')
                  if features.is_cuda
                  else torch.device('cpu'))

        if len(features.shape) < 3:
            raise ValueError('`features` needs to be [bsz, n_views, ...],'
                             'at least 3 dimensions are required')
        if len(features.shape) > 3:
            features = features.view(features.shape[0], features.shape[1], -1)

        batch_size = features.shape[0]
        if labels is not None and mask is not None:
            raise ValueError('Cannot define both `labels` and `mask`')
        elif labels is None and mask is None:
            mask = torch.eye(batch_size, dtype=torch.float32).to(device)
        elif labels is not None:
            labels = labels.contiguous().view(-1, 1)
            if labels.shape[0] != batch_size:
                raise ValueError('Num of labels does not match num of features')
            mask = torch.eq(labels, labels.t()).float().to(device)
            #A = confidence
            #A = A.expand(mask.shape[0], mask.shape[0]).cuda()
            #mask = mask * A
        else:
            mask = mask.float().to(device)

        contrast_count = features.shape[1]
        contrast_feature = torch.cat(torch.unbind(features, dim=1), dim=0)
        if self.contrast_mode == 'one':
            anchor_feature = features[:, 0]
            anchor_count = 1
        elif self.contrast_mode == 'all':
            anchor_feature = contrast_feature
            anchor_count = contrast_count
        else:
            raise ValueError('Unknown mode: {}'.format(self.contrast_mode))

        # compute logits
        anchor_dot_contrast = torch.div(
            torch.matmul(anchor_feature, contrast_feature.t()),
            self.temperature)
        # for numerical stability
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()

        # tile mask
        mask = mask.repeat(anchor_count, contrast_count)
        # mask-out self-contrast cases
        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(batch_size * anchor_count).view(-1, 1).to(device),
            0
        )
        mask = mask * logits_mask

        # compute log_prob
        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))

        # compute mean of log-likelihood over positive
        mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1)

        # loss
        loss = - (self.temperature / self.base_temperature) * mean_log_prob_pos
        loss = loss.view(anchor_count, batch_size).mean()

        return loss


class SupervisedContrastiveLoss(nn.Module):
    """
        Supervised Contrastive Loss
    """
    def __init__(self, temperature=0.07):
        super(SupervisedContrastiveLoss, self).__init__()
        self.temperature = temperature

    def forward(self, inputs, targets):
        """
        Args:
            inputs (torch.Tensor): feature matrix with shape (batch_size, feat_dim).
            targets (torch.LongTensor): ground truth labels with shape (num_classes).
        """
        inputs = F.normalize(inputs, p=2, dim=1)

        n = inputs.size(0)

        mask = targets.expand(n, n).eq(targets.expand(n, n).t())

        mask = mask ^ torch.diag(mask.float().diag())

        anchor_dot_contrast = torch.matmul(inputs, inputs.t())/self.temperature

        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()

        dist = torch.exp(logits)
        ratio = dist/(torch.sum(dist-torch.diag(dist.diag()),1))

        log_ratio = -torch.log(ratio)

        #loss = torch.where(mask != 0, log_ratio, torch.zeros(1).cuda())

        loss = log_ratio * mask.float()
        loss = torch.sum(loss)

        return loss/n


class EntropyLoss(nn.Module):

    def __init__(self, use_gpu=True):
        super(EntropyLoss, self).__init__()
        self.use_gpu = use_gpu

    def forward(self, teacher_features, grl_model, source=True, delta=1, camid=None, da='da'):
        domain_pred = grl_model(teacher_features, delta)
        # print(domain_pred)
        domain_pred = domain_pred.clamp(min=1e-6,max=0.9999999)
        # print(domain_pred)
        # print(domain_pred * torch.log(domain_pred))
        # print((1-domain_pred) * torch.log(1-domain_pred))
        # print(torch.mean(domain_pred * torch.log(domain_pred) + (1-domain_pred) * torch.log(1-domain_pred)))
        return -torch.mean(domain_pred * torch.log(domain_pred) + (1-domain_pred) * torch.log(1-domain_pred)), domain_pred


# class GeneralUncertaintyCrossEntropyLoss(nn.Module):
class CameraLevelConfusionLoss(nn.Module):

    def __init__(self, use_gpu=True):
        super(CameraLevelConfusionLoss, self).__init__()
        self.use_gpu = use_gpu

    def forward(self, features, grl_model, source=True, delta=1, camid=None, multihead='t_head'):

        domain_pred = grl_model(features, delta) #
        if multihead == 'all_head':
            inverse_domain_pred = torch.sum(domain_pred, 1, keepdim=True) - domain_pred
            loss_fn = nn.CrossEntropyLoss()
            l_dc = loss_fn(domain_pred, camid.long())
            reversed_l_dc = loss_fn(inverse_domain_pred, camid.long())
            return l_dc * reversed_l_dc, 0, domain_pred

        else:
            sigmoid = torch.nn.Sigmoid()
            domain_pred = sigmoid(domain_pred[:, 1].unsqueeze(1))
            domain_label = camid.to(domain_pred.device).view(-1, 1)
            inverse_domain_pred = 1 - domain_pred
            #normalized_inverse_domain_pred = torch.norm(inverse_domain_pred, p=2, dim=1, keepdim=True)
            loss_fn = nn.BCELoss()
            l_dc = loss_fn(domain_pred, domain_label.float())
            reversed_l_dc = loss_fn(inverse_domain_pred, domain_label.float())
            return l_dc * reversed_l_dc, 0, domain_pred

        return l_dc, 0, domain_pred


class ClipCluLoss(nn.Module):

    def __init__(self, distance='cosine', use_gpu=True):
        super(ClipCluLoss, self).__init__()
        if distance not in ['cosine']:
            raise KeyError("Unsupported distance: {}".format(distance))
        self.distance = distance
        self.use_gpu = use_gpu

    def forward(self, inputs):
        inputs_normed = F.normalize(inputs, p=2, dim=2)
        mean_rep = torch.mean(inputs_normed, dim=1)
        dist = torch.zeros((inputs_normed.shape[0])).cuda()
        for i in range(inputs_normed.shape[1]):
            dist += 1 - torch.matmul(inputs_normed[:, i, :], mean_rep.t()).diag()
        return torch.mean(dist)


def LogSimoidHM(input):
    max_input = torch.nn.functional.relu(input)
    return input - (max_input + torch.log(torch.exp( - max_input) + torch.exp(+input - max_input)))


class MultipleBinaryCrossEntropy(nn.Module):

    def __init__(self, use_gpu=True):
        super(MultipleBinaryCrossEntropy, self).__init__()
        self.use_gpu = use_gpu
        self.sigmoid = torch.nn.LogSigmoid()

    def forward(self, features, grl_model, delta, camid):

        camera_pred = grl_model(features, delta)
        camid_one_hot = torch.nn.functional.one_hot(camid, num_classes=camera_pred.shape[1]).float()
        multiple_binary = torch.einsum('ij, ij ->i',camid_one_hot, self.sigmoid(camera_pred)) + torch.einsum('ij, ij ->i', 1 - camid_one_hot,torch.log((1 - torch.sigmoid(camera_pred)).clamp(min=1e-6,max=0.999999)))
        return -torch.mean(multiple_binary),0, camera_pred


if __name__ == '__main__':
    a = torch.randint(high=200, size=(4, 2)).float() - torch.randint(high=200, size=(4, 2)).float()
    print(LogSimoidHM(a))
    print(torch.log(torch.sigmoid(a)))
    print(torch.log(torch.nn.functional.sigmoid(a)))
    print("-----")
    print(torch.exp(LogSimoidHM(a)))
    print(torch.sigmoid(a))
