import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.optim as optim
import argparse

from PIL import Image
from torchvision import transforms, models


# Get the pretrained features of model
vgg = models.vgg19(pretrained=True).features

# Freeze the parameters as we won't be traning the model
for param in vgg.parameters():
    param.requires_grad(False)

# Move the model to GPU (if)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
vgg.to(device)

# Load the content image
content = load_image('')
