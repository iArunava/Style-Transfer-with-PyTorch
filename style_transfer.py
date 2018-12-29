import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.optim as optim
import argparse

from PIL import Image
from torchvision import transforms, models
from utils import *

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

parser.add_argument('-cw', '--content-weight',
    type=float,
    default=1.,
    help='The weightage for the content loss')

parser.add_argument('-sw', '--style-weight',
    type=float,
    default=1e7,
    help='The weightage for the style loss')

parser.add_argument('-se', '--show-every',
    type=int,
    default=500,
    help='Interval in which the target image being generated is shown')

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

# Get the Content and feature representations of content and style image
content_feaures = get_features(content, vgg)
style_features = get_features(style, vgg)

# Calculate the gram matrices for each layer in style representation
style_grams = {layer: gram_matrix(style_features[layer]) for layer in style_features}

# Create the target image
target = content.clone().requires_grad_(True).to(device)

# Weights for each style layer
style_weights = {
    'conv1_1' : 1.,
    'conv2_1' : 0.8,
    'conv3_1' : 0.5,
    'conv4_1' : 0.3,
    'conv5_1' : 0.1
}

# Set the content_weight and style_weight
content_weight = FLAGS.content_weight
style_weight = FLAGS.style_weight

# To display the image generated, at some Interval
show_every = FLAGS.show_every
