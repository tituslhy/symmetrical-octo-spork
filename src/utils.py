import torch
from tqdm.notebook import tqdm

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

def train_step(model: torch.nn.Module, 
               dataloader: torch.utils.data.DataLoader,
               loss_fn: torch.nn.Module,
               optimizer: torch.optim.Optimizer,
               device: torch.device,
               accuracy_fn = None):
    """Trains the model for each epoch

    Args:
        model (torch.nn.Module): Instantiated model object
        dataloader (torch.utils.data.Dataloader): Instantiated dataloader object
        optimizer (torch.optim.Optimizer): Optimizer function
        device (str): Device that the model is placed on.
        accuracy_fn: A defined function to compute classification accuracy. 
                    Defaults to "None"
    """
    model.train()
    train_acc, train_loss = 0, 0
    
    for batch, (X, y) in enumerate(tqdm(dataloader)):
        X, y = X.to(device), y.to(device)
        
        # Forward pass
        y_pred = model(X)
        
        # Calculate loss
        loss = loss_fn(y_pred, y)
        train_loss += loss #accumulate train loss
        
        if accuracy_fn is not None:
            train_acc += accuracy_fn(y_true = y,
                                     y_pred = y_pred.argmax(dim=1) #logits to pred labels
                                     )
        
        # Optimizer zero grad
        optimizer.zero_grad()
        
        # Loss backward
        loss.backward()
        
        # Optimizer step
        optimizer.step()
    
    train_loss/=len(dataloader)
    
    if train_acc != 0:
        train_acc/=len(dataloader)
        print(f"Train loss: {train_loss:.5f} | Train acc: {train_acc:.2f}%")
    
    else:
        print(f"Train loss: {train_loss:.5f}")

def test_step(model: torch.nn.Module,
              dataloader: torch.utils.data.DataLoader,
              loss_fn: torch.nn.Module,
              device: torch.device,
              accuracy_fn = None,):
    """Performs a testing loop step using model on dataloader

    Args:
        model (torch.nn.Module): Instantiated model object
        dataloader (torch.utils.data.Dataloader): Instantiated dataloader object
        device (str): Device that the model is placed on. Defaults to torch.device.
        accuracy_fn: A defined python function to compute classification accuracy. Defaults to "None"
    """
    test_loss, test_acc = 0, 0
    
    model.eval()
    
    with torch.inference_mode():
        for X, y in tqdm(dataloader):
            X, y = X.to(device), y.to(device)
            
            # Forward pass
            test_pred = model(X)
            
            # Calculate loss and accuracy
            test_loss += loss_fn(test_pred, y)
            
            if accuracy_fn is not None:
                test_acc += accuracy_fn(y_true = y,
                                        y_pred = test_pred.argmax(dim=1)
                                        )
            
        # Adjust metrics
        test_loss/=len(dataloader)
        
        if test_acc !=0:
            test_acc/=len(dataloader)
            print(f"Test loss: {test_loss:.5f} | Test acc: {test_acc:.2f}%")
        
        else:
            print(f"Test loss: {test_loss:.5f}")

def eval_model(model: torch.nn.Module,
               data_loader: torch.utils.data.DataLoader,
               loss_fn: torch.nn.Module,
               accuracy_fn,
               device: torch.device):
    """Returns a dictionary containing the results of model predicting
    on dataloader

    Args:
        model (torch.nn.Module): _description_
        loss_fn (torch.nn.Module): _description_
        accuracy_fn (_type_): _description_
        data_loader (_type_, optional): _description_. Defaults to torch.utils.data.DataLoader.
    """
    loss, acc = 0, 0
    model.eval()
    with torch.inference_mode():
        for X, y in tqdm(data_loader):
            X, y = X.to(device), y.to(device)
            # Make preds
            y_pred = model(X)
            
            # Accumulate the loss and acc values per batch
            loss += loss_fn(y_pred, y)
            acc += accuracy_fn(y_true = y,
                               y_pred = y_pred.argmax(dim=1))
        
        loss/= len(data_loader)
        acc /= len(data_loader)
    
    return {"model_name": model.__class__.__name__,
            "model_loss": loss.item(),
            "model_acc": acc}