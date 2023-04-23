import torch
from timeit import default_timer as timer
from tqdm.auto import tqdm

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

def eval_model(model: torch.nn.Module,
               data_loader: torch.utils.data.DataLoader,
               loss_fn: torch.nn.Module,
               accuracy_fn,
               SEED: int = 42):
    """Returns a dictionary containing the results of model predicting
    on dataloader

    Args:
        model (torch.nn.Module): _description_
        loss_fn (torch.nn.Module): _description_
        accuracy_fn (_type_): _description_
        data_loader (_type_, optional): _description_. Defaults to torch.utils.data.DataLoader.
    """
    torch.manual_seed(SEED)
    loss, acc = 0, 0
    model.eval()
    with torch.inference_mode():
        for X, y in tqdm(data_loader):
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