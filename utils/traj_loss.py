import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class TransLoss(nn.Module):
    def __init__(self, loss_weight=1.0):
        super().__init__()
        self.loss_weight = loss_weight
        self.frame_loss_weight = [1.0, 0.85, 0.7, 0.55, 0.4, 0.25]  # as default
        self.frame_loss_weight = torch.tensor(self.frame_loss_weight, dtype=torch.float32, device='cuda')

    def forward(self, pred_trajs, targ_trajs):
        bs, f = pred_trajs.shape[0], pred_trajs.shape[1]

        pred_trajs = torch.cumsum(pred_trajs, dim=1)
        targ_trajs = torch.cumsum(targ_trajs, dim=1)
        frame_loss_weight = self.frame_loss_weight[None].repeat(bs, 1)

        traj_loss = torch.sum((pred_trajs - targ_trajs) ** 2, dim=-1)
        traj_loss = traj_loss * frame_loss_weight[:, :f]
        traj_loss = traj_loss.mean(-1) / bs

        return traj_loss * self.loss_weight


class RotationLoss(nn.Module):
    def __init__(self, loss_weight=5.0):
        super().__init__()
        self.loss_weight = loss_weight

    def forward(self, pred_rotations, targ_rotations):
        bs = pred_rotations.shape[0]

        dot_product = torch.sum(pred_rotations * targ_rotations, dim=-1)
        dot_product = torch.clamp(dot_product, -1, 1)

        loss = 1.0 - torch.abs(dot_product)
        return loss.mean() * self.loss_weight
