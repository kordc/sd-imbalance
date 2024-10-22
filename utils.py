import torch


def save_checkpoint(state, filename='checkpoint.pth'):
    torch.save(state, filename)
