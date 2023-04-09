import torch

def get_pytorch_device():
    if torch.cuda.is_available():
        return 'cuda'
    elif torch.backends.mps.is_available():
        return 'mps'
    return 'cpu'

def train():
    pass

def evaluate():
    pass