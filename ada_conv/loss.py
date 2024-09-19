import torch
from torch import nn
import torch.nn.functional as F


class MSEContentLoss(nn.Module):
    def forward(self, content, pred):
        return F.mse_loss(content, pred)


class MomentMatchingStyleLoss(nn.Module):
    def forward(self, styles, preds):
        losses = []
        for style, pred in zip(styles, preds):
            style_std, style_mean = torch.std_mean(style, dim=[2, 3])
            pred_std, pred_mean = torch.std_mean(pred, dim=[2, 3])

            mean_loss = F.mse_loss(style_mean, pred_mean)
            std_loss = F.mse_loss(style_std, pred_std)

            losses.append(mean_loss + std_loss)

        return sum(losses)
