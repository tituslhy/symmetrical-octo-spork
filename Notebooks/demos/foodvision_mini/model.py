
import torch
import torchvision
from torch import nn

def create_effnetb2_model(num_classes: int = 3,
                          seed: int = 42):
    effnetb2_weights = torchvision.models.EfficientNet_B2_Weights.DEFAULT
    effnetb2_transforms = effnetb2_weights.transforms()
    effnetb2 = torchvision.models.efficientnet_b2(weights=effnetb2_weights)
    for param in effnetb2.parameters():
        param.requires_grad = False
    torch.manual_seed(seed)
    effnetb2.classifier = nn.Sequential(
        nn.Dropout(p = 0.3, inplace = True),
        nn.Linear(in_features = 1408, out_features = num_classes)
    )
    return effnetb2, effnetb2_transforms
