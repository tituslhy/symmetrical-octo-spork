"""
Contains functionality for creating PyTorch DataLoaders for
image classification tasks.
"""
import os
from typing import Tuple, List
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

NUM_WORKERS = os.cpu_count()

def create_dataloaders(
    train_dir: str,
    test_dir: str,
    transform: transforms.Compose,
    batch_size: int,
    num_workers: int = NUM_WORKERS
) -> Tuple[torch.utils.data.DataLoader, 
           torch.utils.data.DataLoader,
           List]:
    """Creates torch datasets and subsequently dataloaders for training
    and testing sets. 

    Args:
        train_dir (str): Filepath to train data
        test_dir (str): Filepath to test data
        transform (transforms.Compose): torch transforms.Compose object
        batch_size: Number of samples per batch in each of the datalaoders
        num_workers (int, optional): Defaults to NUM_WORKERS.
    
    Returns:
        Tuple of (train_dataloader, test_dataloader, class_names) where
        classnames is a list of the target classes.
    """
    train_data = datasets.ImageFolder(root = train_dir,
                                      transform = transform)
    class_names = train_data.classes
    test_data = datasets.ImageFolder(root = test_dir,
                                      transform = transform)
    
    train_dataloader = DataLoader(train_data,
                                  batch_size = batch_size,
                                  shuffle = True,
                                  num_workers = num_workers,
                                  pin_memory = True
                                  )
    
    test_dataloader = DataLoader(test_data,
                                  batch_size = batch_size,
                                  shuffle = False,
                                  num_workers = num_workers,
                                  pin_memory = True
                                  )
    
    return train_dataloader, test_dataloader, class_names
