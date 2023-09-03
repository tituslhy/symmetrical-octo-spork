"""The main parts are:
1. Imports and class names setup
2. Model and transforms preparation
3. Write a predict function for gradio to use
4. Write the Gradio app and the launch command
"""
import os
from typing import Tuple, Dict
import PIL
import torch
import torchvision
import gradio as gr
from timeit import default_timer
from model import create_effnetb2_model

class_names = ['pizza', 'steak', 'sushi'] #hardcoded as a list
model, transforms = create_effnetb2_model(num_classes = len(class_names))

# Load saved weights into the model, and load the model onto the CPU
model.load_state_dict(torch.load(f = "finetuned_effnetb2_20percent.pth"),
                      map_location = torch.device('cpu'))

# Write function to run inference on gradio
def predict(img: PIL.Image, 
            model: nn.Module = model,
            transforms: torchvision.transforms = transforms,
            class_names: List[str] = class_names) -> Tuple[Dict, float]:
    """Function to predict image class on gradio

    Args:
        img (np.array): Image as a numpy array
        model (nn.Module, optional): Model. Defaults to vit.
        class_names (List[str], optional): List of class anmes. Defaults to class_names.

    Returns:
        Tuple[Dict, float]: Tuplefor further processing on gradio
    """
    start_time = timer()
    img = transforms(img).unsqueeze(0) #add batch dimension
    model.eval()
    with torch.inference_mode():
        pred_probs = torch.softmax(model(img), dim = 1)
    pred_labels_and_probs = {class_names[i]: float(pred_probs[0][i]) for i in range(len(class_names))}
    end_time = timer()
    pred_time = round(end_time - start_time, 4)
    return pred_labels_and_probs, pred_time

# Create example_list
example_list = [["examples/" + example] for example in os.listdir("examples")]

# Create Gradio App
title = 'FoodVision Mini 🍕🥩🍣'
description = "Using a [Vision Transformer](https://arxiv.org/abs/2010.11929) for Image Classification"
article = "Created by [Titus Lim](https://github.com/tituslhy)"

demo = gr.Interface(fn = predict,
                    inputs = gr.Image(type = "pil"),
                    outputs = [gr.Label(num_top_classes = 3, label = "Predictions"),
                               gr.Number(label = "Prediction time (s)")],
                    examples = example_list,
                    title = title,
                    description = description,
                    article = article)

# Launch demo
demo.launch(debug = False, #prints errors locally
            share = True # Generate a publicly shareable URL
            )
