import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.optim as optim

from PIL import Image
from torchvision import transforms, models

def load_image(img_path, max_size=500, shape=None):
    """
    Load image and perform transformations
    """

    # Load the img from path
    img = Image.open(img_path).convert('RGB')

    # Resize to max_size if larger
    # as large images will slow down processing
    if max(img.size) > max_size:
        size = max_size
    else:
        size = max(img.size)

    # Define the transformations
    in_transform = transforms.Compose([
                        transforms.Resize(size),
                        transforms.ToTensor(),
                        transforms.Normalize((0.485, 0.456, 0.406),
                                             (0.229, 0.224, 0.225))
                    ])

    # Discard the alpha channel (if)
    # Note: image now have channels first
    # And then unsqueeze to add the batch dimension
    img = in_transform(img)[:3, :, :].unsqueeze(0)

    return img

def im_convert(tensor):
    """
    converts tensors to images to display
    """

    # Move the tensor to cpu, clone the tensor and detach it from the computational graph
    img = tensor.to('cpu').clone().detach()

    # Remove the batch dimension
    img = img.squeeze()

    # Get the channels last
    img = img.transpose(1, 2, 0)

    # Apply the transformations applied before passing the image to the model
    img = img * np.array((0.229, 0.224, 0.225)) + np.array((0.485, 0.456, 0.406))

    # Clip the image b/w 1 and 0
    img = img.clip(0, 1)

    return img

def get_features(img, model, layers=None):
    """
    Get the feature representation for the set of layers
    """

    # The layers considered
    if layers == None:
        layers = {
            '0' : 'conv1_1',
            '5' : 'conv2_1',
            '8' : 'conv3_1',
            '19' : 'conv4_1',
            '21' : 'conv4_2',
            '28' : 'conv5_1'
        }

    # Get the output of the layers
    features = {}
    x = image
    for name, layer in model._modules.items():
        x = layer(x)
        if name in layers:
            features[layers[name]] = x

    return features
