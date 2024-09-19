import torch.nn as nn


def init_weights(m):
    if isinstance(m, nn.Conv2d):
        if m.weight.requires_grad:
            nn.init.xavier_normal_(m.weight)
        if m.bias is not None and m.bias.requires_grad:
            nn.init.zeros_(m.bias)
    elif isinstance(m, nn.Linear):
        if m.weight.requires_grad:
            nn.init.xavier_normal_(m.weight)
        if m.bias is not None and m.bias.requires_grad:
            nn.init.zeros_(m.bias)
    elif isinstance(m, nn.LayerNorm):
        if m.weight.requires_grad:
            nn.init.ones_(m.weight)
        if m.bias is not None and m.bias.requires_grad:
            nn.init.zeros_(m.bias)
