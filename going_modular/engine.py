"""Contains functions for training and testing a PyTorch model
"""

import torch
from torch import nn
from typing import Dict, List, Tuple
from tqdm.auto import tqdm
from collections import defaultdict

def train_step(model: torch.nn.Module,
               dataloader: torch.utils.data.DataLoader,
               loss_fn: torch.nn.Module,
               optimizer: torch.optim.Optimizer,
               device: str) -> Tuple[float, float]:
    """Helper function to train pytorch model on device
    and acquire training metrics per epoch

    Args:
        model (torch.nn.Module): instantiated torch model
        dataloader (torch.utils.data.DataLoader)
        loss_fn (torch.nn.Module)
        optimizer (torch.optim.Optimizer) 
        device (str): Torch device

    Returns:
        Average training loss and training accuracy per epoch
    """
    
    train_loss, train_acc = 0,0
    model.train()

    for batch, (X, y) in enumerate(dataloader):
        
        # Forward pass
        X, y = X.to(device), y.to(device)
        y_pred = model(X) #logits
        loss = loss_fn(y_pred, y)
        train_loss += loss.item()

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Compute metric across all batches
        y_pred_class = torch.argmax(
            torch.softmax(y_pred, dim = 1),
            dim = 1
        )

        train_acc += (y_pred_class==y).sum().item()/len(y_pred)
    
    # Adjust metrics to get average loss and accuracy per batch
    train_loss = train_loss / len(dataloader)
    train_acc = train_acc/len(dataloader)

    return train_loss, train_acc

def test_step(model: torch.nn.Module,
              dataloader: torch.utils.data.DataLoader,
              loss_fn: torch.nn.Module,
              device: str) -> Tuple[float, float]:
    """Runs inference of trained model on test dataset per epoch
    and monitors model test metrics.

    Args:
        model (torch.nn.Module): instantiated torch model
        dataloader (torch.utils.data.DataLoader)
        loss_fn (torch.nn.Module)
        device (str, optional): _description_. Defaults to device.

    Returns:
        Average test loss and test accuracy per epoch
    """
    
    test_loss, test_acc = 0,0
    model.eval()
    
    with torch.inference_mode():
        for batch, (X, y) in enumerate(dataloader):
            
            # Forward pass
            X, y = X.to(device), y.to(device)
            test_pred = model(X) #logits

            # Compute metrics 
            loss = loss_fn(test_pred, y)
            test_loss += loss.item()
            test_pred_class = torch.argmax(
                torch.softmax(test_pred, dim = 1),
                dim = 1
            )
            test_acc += (test_pred_class == y).sum().item()/len(test_pred_class)
    
    # Adjust metrics to get average loss and accuracy per batch
    test_loss = test_loss/len(dataloader)
    test_acc = test_acc/len(dataloader)
    
    return test_loss, test_acc

def train(model: torch.nn.Module,
          train_dataloader: torch.utils.data.DataLoader,
          test_dataloader: torch.utils.data.DataLoader,
          optimizer: torch.optim.Optimizer,
          epochs: int,
          loss_fn: torch.nn.Module = nn.CrossEntropyLoss()): 
    """Wrapper function to train model over specified number of epochs,
    model, dataloaders, optimizer and loss function.

    Args:
        model (torch.nn.Module): instantiated torch model
        train_dataloader (torch.utils.data.DataLoader)
        test_dataloader (torch.utils.data.DataLoader)
        optimizer (torch.optim.Optimizer)
        epochs (int): Number of epochs for training
        loss_fn (torch.nn.Module, optional) Defaults to nn.CrossEntropyLoss().

    Returns:
        Dictionary of results
    """
    
    # Create storage results dictionary
    results = defaultdict(list)

    # Loop through training and testing steps for number of epochs
    for epoch in tqdm(range(epochs)):
        train_loss, train_acc = train_step(model = model,
                                           dataloader = train_dataloader,
                                           loss_fn = loss_fn,
                                           optimizer = optimizer)
        test_loss, test_acc = test_step(model = model,
                                        dataloader = test_dataloader,
                                        loss_fn = loss_fn)
        print(
            f"Epoch: {epoch+1} | "
            f"train_loss: {train_loss:.4f} | "
            f"train_acc: {train_acc:.4f} | "
            f"test_loss: {test_loss:.4f} | "
            f"test_acc: {test_acc:.4f}"
        )

        results["train_loss"].append(train_loss)
        results["train_acc"].append(train_acc)
        results["test_loss"].append(test_loss)
        results["test_acc"].append(test_acc)
    
    return results
