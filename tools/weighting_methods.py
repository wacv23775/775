from __future__ import absolute_import

import torch
from torch import nn


class StaticRatioWeighting(nn.Module):

    def forward(self, losses, epoch):
        total_losses = sum(losses).cuda()
        if epoch > 10:
            ratio = 1/len(losses)
            alphas = torch.zeros(len(losses)).cuda()
            for i, loss in enumerate(losses):
                alphas[i] = (ratio * total_losses/loss.clamp(min=1e-8)).cuda()

            scaling_factor = (len(losses)/torch.sum(alphas)).cuda()
            deltas = alphas * scaling_factor
            deltas = deltas.cuda()
            print(deltas)
            print(sum([deltas[i] * loss for i, loss in enumerate(losses)]).cuda())
            print(ratio)
            return sum([deltas[i] * loss for i, loss in enumerate(losses)]).cuda()
        else:
            return total_losses


class StaticWeighting(nn.Module):

    def forward(self, losses, epoch=0):
        return sum(losses)
