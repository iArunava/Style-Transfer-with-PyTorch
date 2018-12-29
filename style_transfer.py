import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.optim as optim
import argparse

from PIL import Image
from torchvision import transforms, models

# Get the argument parset and take command line arguments
parser = argparse.ArgumentParser()
parser.add_argument('-c', '--content-image',
    type=str,
    required=True,
    help='The path to the content image')

parser.add_argument('-s', '--style-image',
    type=str,
    required=True,
    help='The path to the style image')

FLAGS, unparsed = parser.parse_known_args()

# Get the pretrained features of model
vgg = models.vgg19(pretrained=True).features

# Freeze the parameters as we won't be traning the model
for param in vgg.parameters():
    param.requires_grad(False)

# Move the model to GPU (if)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
vgg.to(device)

# Load the content image
content = load_image(FLAGS.content_image).to(device)

# Load the style image
style = load_image(FLAGS.style_image).to(device)
