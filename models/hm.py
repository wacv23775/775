import numpy as np

import torch
import torch.nn.functional as F
from torch.nn import init
from torch import nn, autograd


class HM(autograd.Function):

    @staticmethod
    def forward(ctx, inputs, indexes, features, momentum):
        ctx.features = features
        ctx.momentum = momentum
        ctx.save_for_backward(inputs, indexes)
        outputs = inputs.mm(ctx.features.t())

        return outputs

    @staticmethod
    def backward(ctx, grad_outputs):
        inputs, indexes = ctx.saved_tensors
        grad_inputs = None
        if ctx.needs_input_grad[0]:
            grad_inputs = grad_outputs.mm(ctx.features)

        # momentum update
        for x, y in zip(inputs, indexes):
            ctx.features[y] = ctx.momentum * ctx.features[y] + (1. - ctx.momentum) * x
            ctx.features[y] /= ctx.features[y].norm()

        return grad_inputs, None, None, None


def hm(inputs, indexes, features, momentum=0.5):
    return HM.apply(inputs, indexes, features, torch.Tensor([momentum]).to(inputs.device))


class HybridMemory(nn.Module):
    def __init__(self, num_features, num_samples, temp=0.05, momentum=0.2, weighting=False, num_clips=2, strat_gamma="clips"):
        super(HybridMemory, self).__init__()
        self.num_features = num_features
        self.num_samples = num_samples
        self.num_clips = num_clips
        if self.num_clips > 2:
            assert "Need to implement for clips > 2"
        self.strat_gamma = strat_gamma

        self.momentum = momentum
        self.temp = temp

        self.weighting = weighting
        if self.weighting:
            print("weighting is ACTIVATED and in mode: ", self.strat_gamma)

        self.register_buffer('features', torch.zeros(num_samples, num_features))
        self.register_buffer('labels', torch.zeros(num_samples).long())

    def forward(self, inputs, indexes):
        # inputs: B*2048, features: L*2048
        if self.weighting:
            initial = inputs.detach()
            gamma_clips = initial[int(initial.shape[0] / self.num_clips):].t().diag()
            gamma_clips = gamma_clips/self.temp
            gamma_clips = torch.cat((gamma_clips, gamma_clips))

        inputs = hm(torch.reshape(inputs, [inputs.shape[0], inputs.shape[1]]), indexes, self.features, self.momentum)
        inputs /= self.temp
        B = inputs.size(0)

        def masked_softmax(vec, mask, dim=1, epsilon=1e-6):
            exps = torch.exp(vec)
            masked_exps = exps * mask.float().clone()
            masked_sums = masked_exps.sum(dim, keepdim=True) + epsilon
            return (masked_exps/masked_sums)

        targets = self.labels[indexes].long().clone()
        labels = self.labels.long().clone()
        idx_end_source = labels.min()

        sim = torch.zeros(labels.max()+1, B).float().cuda()
        sim.index_add_(0, labels, inputs.t().contiguous())  # Add features of the same labels at corresponding indexx = torch.ones(5, 3)
        nums = torch.zeros(labels.max()+1, 1).float().cuda()
        nums.index_add_(0, labels, torch.ones(self.num_samples, 1).float().cuda())  # Nums of samples per cluster
        mask = (nums > 0).float()
        sim /= (mask*nums+(1-mask)).clone().expand_as(sim)  # calculate the cluster centers by dividing the feature sum by the number of samples.
        mask = mask.expand_as(sim)
        masked_sim = masked_softmax(sim.t().contiguous(), mask.t().contiguous())  # log(masked_sim) is l_ij

        if self.weighting:

            sim_weight = masked_sim.detach()

            if self.strat_gamma == "kNN":
                top = torch.topk(sim_weight[:, idx_end_source:], 4, dim=1, largest=True)
                gamma = torch.max(top[0], dim=1)[0].view(B, 1)

            elif self.strat_gamma == "clips":
                gamma = torch.exp(gamma_clips).view(gamma_clips.shape[0], 1) + 1e-6

            w = 1 - (1/gamma)*sim_weight
            r_gamma = -gamma * w
            # r_gamma = gamma * (0.5*(w*w) - w)

            l = torch.log(masked_sim+1e-6)

            return F.nll_loss(w * l - r_gamma, targets)

        return F.nll_loss(torch.log(masked_sim+1e-6), targets)

