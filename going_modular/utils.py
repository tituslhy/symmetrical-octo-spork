"""Utility functions for torch driven computer vision projects
"""

from pathlib import Path
import torch

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
