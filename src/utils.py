import os
from pathlib import Path
from typing import Tuple, Dict, List
from tqdm.notebook import tqdm

from PIL import Image
import torch

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

def make_predictions(model:torch.nn.Module,
                     data:list,
                     device: torch.device):
    """Generates a list of predictions given a list of data

    Args:
        model (torch.nn.Module): Model instance
        data (list): List of img tensors to run inference
        device (torch.device, optional): Defaults to device.

    Returns:
        torch.tensor: Tensor of probabilities
    """
    pred_probs = []
    model.eval()
    with torch.inference_mode():
        for sample in data:
            # Prep sample - add the batch dimension and pass to device
            sample = torch.unsqueeze(sample, dim=0).to(device)
            
            # Forward pass
            pred_logit = model(sample)
            
            #Get prediction orobability
            pred_prob = torch.softmax(pred_logit.squeeze(), dim = 0)
            
            pred_probs.append(pred_prob.cpu())
    
    #Stack pred_probs in list into tensor
    return torch.stack(pred_probs)

def save_model(root_dir: str,
               model_name: str,
               model: torch.nn.Module):
    """Saves a torch model to the given root directory

    Args:
        root_dir (str): Name of directory to store model
        model_name (str): Name of the model
        model (torch.nn.Module): Torch model object
    """
    MODEL_PATH = Path(root_dir)
    MODEL_PATH.mkdir(parents=True,
                     exist_ok = True)
    MODEL_NAME = model_name
    MODEL_SAVE_PATH = MODEL_PATH / MODEL_NAME

    torch.save(obj = model.state_dict(),
            f = MODEL_SAVE_PATH)
    print(f"Saved model to: {MODEL_SAVE_PATH}")

def find_classes(directory: str) -> Tuple[List[str], Dict[str, int]]:
    """Use os.scandir() to traverse a target directory and get class names
    Turn class names into a dict and returns it
    """
    classes = sorted(entry.name for entry in list(os.scandir(directory)) if entry.is_dir())
    if not classes:
        raise FileNotFoundError(f"Could not find any classes in {directory}...\nPlease check file structure")
    class_to_idx = {class_name:i for i, class_name in enumerate(classes)}
    return classes, class_to_idx

class ImageFolderCustom(torch.utils.data.Dataset):
    """Custom dataset object
    """
    def __init__(self, target_directory: str = None, transform = None):
        """Class constructor. Takes in a target directory filepath and
        optionally a torch transform.
        """
        self.transform = transform
        self.classes, self.class_to_idx = find_classes(target_directory)
        self.paths = list(Path(target_directory).glob("*/*.jpg"))
    def load_image(self, index: int) -> Image.Image:
        """Opens an image given a path and returns it
        """
        image_path = self.paths[index]
        return Image.open(image_path)
    def __len__(self):
        """Returns total number of samples
        """
        return len(self.paths)
    def __getitem__(self, index: int) -> Tuple[torch.Tensor, int]:
        """Overrides __getitem__ method to return a particular sample
        Returns one sample of data -> X and y. If a transform is specified,
        returns the transformed image and label. Otherwise, just returns
        original image and label.
        """
        img = self.load_image(index)
        class_name = self.paths[index].parent.name
        class_idx = self.class_to_idx[class_name]
        
        if self.transform:
            return self.transform(img), class_idx
        else:
            return img, class_idx