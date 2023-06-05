"""Trains a PyTorch tinyVGG image classification model
using device agnostic code
"""
import os
import get_data, data_setup, engine, model_builder, utils

import sys
sys.path.append("../")
from src.utils import get_device

import torch
from torch import nn
from torchvision import transforms
from timeit import default_timer as timer

import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--seed",
                    type = int,
                    required = False,
                    help = "Random seed for reproducible experiments")
parser.add_argument("--epochs",
                    type = int,
                    required = True,
                    help = "Total number of epochs for training")
parser.add_argument("--batch_size",
                    type = int,
                    required = False,
                    default = 32,
                    help = "Batch size for model training")
parser.add_argument("--lr",
                    type = float,
                    required = True,
                    help = "Learning rate for torch optimizer")
parser.add_argument("--workers",
                    type = int,
                    required = False,
                    default = 0,
                    help = "Number of workers for training")
parser.add_argument("--url",
                    type = str,
                    required = False,
                    default = "https://github.com/mrdbourke/pytorch-deep-learning/raw/main/data/pizza_steak_sushi.zip",
                    help = "url for data download")

args = parser.parse_args()

#Hyperparameters
BATCH_SIZE = args.batch_size
LEARNING_RATE = args.lr
EPOCHS = args.epochs
NUM_WORKERS = args.workers
URL = args.url
if args.seed:
    SEED = args.seed
    torch.manual_seed(SEED)

HIDDEN_UNITS = 20

#Get data
data_path = "data"
image_dir = "pizza_steak_sushi"
return_ = get_data.main(url,
                        data_path,
                        image_dir)

#Setup directores
train_dir = f"{data_path}/{image_dir}/train"
test_dir = f"{data_path}/{image_dir}/test"

#Get device
device = get_device()
if device == "mps":
    torch.mps.manual_seed(SEED)
elif device == 'cuda':
    torch.cuda.manual_seed(SEED)

#Create transforms
data_transform = transforms.Compose([
    transforms.Resize((64,64)),
    transforms.ToTensor()
])

#Create dataloaders and get class names
train_dataloader, test_dataloader, class_names = data_setup.create_dataloaders(
    train_dir = train_dir,
    test_dir = test_dir,
    transform = data_transform,
    batch_size = BATCH_SIZE,
    num_workers = NUM_WORKERS
)

#Create model
model = model_builder.TinyVGG2(
    num_color_channels = 3,
    hidden_units = HIDDEN_UNITS,
    num_classes = 3
).to(device)

#Setup loss and optimizer
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(),
                             lr = LEARNING_RATE)

#Train model
start_time = timer()
engine.train(model = model,
             train_dataloader = train_dataloader,
             test_dataloader = test_dataloader,
             loss_fn = loss_fn,
             optimizer = optimizer,
             epochs = EPOCHS,
             device = device)
end_time = timer()
print(f"Total training time: {end_time - start_time:.3f} seconds")

#Save model to file
utils.save_model(model = model,
                 root_dir = "models",
                 model_name = "tinyVGG2_pizza_steak_sushi.pth")

print("Model saved in models directory as tinyVGG2_pizza_steak_sushi.pth")
