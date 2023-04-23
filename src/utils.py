import torch
from timeit import default_timer as timer


def get_device():
    """Detects accelerator device. The possible devices are:
    1. cuda - this is preferred
    2. mps - this is for mac devices
    3. cpu - this is returned if there are no accelerator devices
    available.

    Returns:
        str: name of accelerator device available.
    """
    if torch.cuda.is_available():
        return 'cuda'
    elif torch.backends.mps.is_available():
        return 'mps'
    return 'cpu'

def classification_accuracy(y_true, y_pred):
    correct = torch.eq(y_true, y_pred).sum().item()
    return (correct/len(y_pred))*100

def print_train_time(start: float,
                     end: float,
                     device: torch.device = None):
    """Prints difference between start and end time"""
    total_time = end - start
    print(f"Train time on {device}: {total_time:.3f} seconds")