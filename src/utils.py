import torch

FORCE_CPU = False
def get_pytorch_device():
    if not FORCE_CPU:
        if torch.cuda.is_available():
            return 'cuda'
        elif torch.backends.mps.is_available():
            return 'mps'
    
    return 'cpu'