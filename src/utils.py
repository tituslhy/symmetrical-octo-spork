import os
from pathlib import Path
from typing import Tuple, Dict, List
from tqdm.notebook import tqdm
from collections import defaultdict

from PIL import Image
import torch
from torch import nn

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

# def train_step(model: torch.nn.Module,
#                dataloader: torch.utils.data.DataLoader,
#                loss_fn: torch.nn.Module,
#                optimizer: torch.optim.Optimizer,
#                device: str = get_device()) -> Tuple[float, float]:
#     """Helper function to train pytorch model on device
#     and acquire training metrics per epoch

#     Args:
#         model (torch.nn.Module): instantiated torch model
#         dataloader (torch.utils.data.DataLoader)
#         loss_fn (torch.nn.Module)
#         optimizer (torch.optim.Optimizer) 
#         device (str, optional): Defaults to device.

#     Returns:
#         Average training loss and training accuracy per epoch
#     """
    
#     train_loss, train_acc = 0,0
#     model.train()

#     for batch, (X, y) in enumerate(dataloader):
        
#         # Forward pass
#         X, y = X.to(device), y.to(device)
#         y_pred = model(X) #logits
#         loss = loss_fn(y_pred, y)
#         train_loss += loss.item()

#         # Backpropagation
#         optimizer.zero_grad()
#         loss.backward()
#         optimizer.step()

#         # Compute metric across all batches
#         y_pred_class = torch.argmax(
#             torch.softmax(y_pred, dim = 1),
#             dim = 1
#         )

#         train_acc += (y_pred_class==y).sum().item()/len(y_pred)
    
#     # Adjust metrics to get average loss and accuracy per batch
#     train_loss = train_loss / len(dataloader)
#     train_acc = train_acc/len(dataloader)

#     return train_loss, train_acc

# def test_step(model: torch.nn.Module,
#               dataloader: torch.utils.data.DataLoader,
#               loss_fn: torch.nn.Module,
#               device: str = get_device()) -> Tuple[float, float]:
#     """Runs inference of trained model on test dataset per epoch
#     and monitors model test metrics.

#     Args:
#         model (torch.nn.Module): instantiated torch model
#         dataloader (torch.utils.data.DataLoader)
#         loss_fn (torch.nn.Module)
#         device (str, optional): Defaults to device.

#     Returns:
#         Average test loss and test accuracy per epoch
#     """
    
#     test_loss, test_acc = 0,0
#     model.eval()
    
#     with torch.inference_mode():
#         for batch, (X, y) in enumerate(dataloader):
            
#             # Forward pass
#             X, y = X.to(device), y.to(device)
#             test_pred = model(X) #logits

#             # Compute metrics 
#             loss = loss_fn(test_pred, y)
#             test_loss += loss.item()
#             test_pred_class = torch.argmax(
#                 torch.softmax(test_pred, dim = 1),
#                 dim = 1
#             )
#             test_acc += (test_pred_class == y).sum().item()/len(test_pred_class)
    
#     # Adjust metrics to get average loss and accuracy per batch
#     test_loss = test_loss/len(dataloader)
#     test_acc = test_acc/len(dataloader)
    
#     return test_loss, test_acc


# def train(model: torch.nn.Module,
#           train_dataloader: torch.utils.data.DataLoader,
#           test_dataloader: torch.utils.data.DataLoader,
#           optimizer: torch.optim.Optimizer,
#           epochs: int,
#           loss_fn: torch.nn.Module = nn.CrossEntropyLoss(),) -> Dict(List):
#     """Wrapper function to train model over specified number of epochs,
#     model, dataloaders, optimizer and loss function.

#     Args:
#         model (torch.nn.Module): instantiated torch model
#         train_dataloader (torch.utils.data.DataLoader)
#         test_dataloader (torch.utils.data.DataLoader)
#         optimizer (torch.optim.Optimizer)
#         loss_fn (torch.nn.Module, optional) Defaults to nn.CrossEntropyLoss().
#         epochs (int): Number of epochs to train

#     Returns:
#         Dictionary of results
#     """
    
#     # Create storage results dictionary
#     results = defaultdict(list)

#     # Loop through training and testing steps for number of epochs
#     for epoch in tqdm(range(epochs)):
#         train_loss, train_acc = train_step(model = model,
#                                            dataloader = train_dataloader,
#                                            loss_fn = loss_fn,
#                                            optimizer = optimizer)
#         test_loss, test_acc = test_step(model = model,
#                                         dataloader = test_dataloader,
#                                         loss_fn = loss_fn)
#         print(
#             f"Epoch: {epoch+1} | "
#             f"train_loss: {train_loss:.4f} | "
#             f"train_acc: {train_acc:.4f} | "
#             f"test_loss: {test_loss:.4f} | "
#             f"test_acc: {test_acc:.4f}"
#         )

#         results["train_loss"].append(train_loss)
#         results["train_acc"].append(train_acc)
#         results["test_loss"].append(test_loss)
#         results["test_acc"].append(test_acc)
    
#     return results

# def make_predictions_batch(model:torch.nn.Module,
#                            data:list,
#                            device: torch.device):
#     """Generates a list of predictions given a list of data

#     Args:
#         model (torch.nn.Module): Model instance
#         data (list): List of img tensors to run inference
#         device (torch.device, optional): Defaults to device.

#     Returns:
#         torch.tensor: Tensor of probabilities
#     """
#     pred_probs = []
#     model.eval()
#     with torch.inference_mode():
#         for sample in data:
#             # Prep sample - add the batch dimension and pass to device
#             sample = torch.unsqueeze(sample, dim=0).to(device)
            
#             # Forward pass
#             pred_logit = model(sample)
            
#             #Get prediction orobability
#             pred_prob = torch.softmax(pred_logit.squeeze(), dim = 0)
            
#             pred_probs.append(pred_prob.cpu())
    
#     #Stack pred_probs in list into tensor
#     return torch.stack(pred_probs)

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