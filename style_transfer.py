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

parser.add_argument('-e', '--epochs',
    type=int,
    default=3000,
    help='The number of epochs')

parser.add_argument('-si', '--show-image-at-intervals',
    type=bool,
    default=False,
    help='Show image at intervals')

parser.add_argument('-lr', '--learning-rate',
    type=float,
    default=0.003,
    help='Learning Rate')

FLAGS, unparsed = parser.parse_known_args()

# Get the pretrained features of model
print ('[INFO]Loading Model...')
vgg = models.vgg19(pretrained=True).features
print ('[INFO]Model Loaded Successfully!')

# Freeze the parameters as we won't be traning the model
for param in vgg.parameters():
    param.requires_grad(False)

# Move the model to GPU (if)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
vgg.to(device)

# Load the content image
content = load_image(FLAGS.content_image).to(device)
print ('[INFO]Loaded Content Image')

# Load the style image
style = load_image(FLAGS.style_image).to(device)
print ('[INFO]Loaded Style Image')

# Get the Content and feature representations of content and style image
content_feaures = get_features(content, vgg)
print ('[INFO]Got the Content Features.')
style_features = get_features(style, vgg)
print ('[INFO]Got the Style Features.')

# Calculate the gram matrices for each layer in style representation
style_grams = {layer: gram_matrix(style_features[layer]) for layer in style_features}
print ('[INFO]Calculated the Gram Matrices from the style layers')

# Create the target image
target = content.clone().requires_grad_(True).to(device)
print ('[INFO]Created target image')

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

# Iteration hyperparameters
optimizer = optim.Adam([target], lr=FLAGS.learning_rate)
epochs = FLAGS.epochs

print ('[INFO]Starting Training...')
for ii in range(1, epochs+1):
    # Get the features from target image
    target_features = get_features(target, vgg)

    # Calculate the content loss
    content_loss = torch.mean((target_features['conv4_2'] - content_feaures['conv4_2']) ** 2)

    # Calculating the style loss
    # Intitialize the style loss to 0
    style_loss = 0
    # Iterate over the layers
    for layer in style_weights:
        # Get the target style representation
        target_feature = target_features[layer]

        # Get the dimensions
        _, d, h, w = target_feature.shape

        # Calculate the target gram matrix
        target_gram = gram_matrix(target_feature)

        # Get the corresponding style gram
        style_gram = style_grams[layer]

        # Calculate the style loss
        layer_style_loss = style_weights[layer] * torch.mean((target_gram - style_gram) ** 2)

        # Add to the style loss
        style_loss += layer_style_loss / (d * w * h)

    # Calculate the total loss
    total_loss = (content_weight * content_loss) + (style_weight * style_loss)

    # update the target image
    optimizer.zero_grad()
    total_loss.backward()
    optimizer.step()

    print ('[INFO]Epoch {} Completed!'.format(ii))
    # Display if Interval
    if ii % show_every == 0:
        print ('Epochs: {} Loss: {}'.format(ii, total_loss.item()))
        im_image = im_convert(target)
        plt.imsave('./{}-{}.png'.format(ii, total_loss), im_image)
        if show_image_at_intervals:
            plt.imshow(im_image)
            plt.axis('off')
            plt.show()

print ('[INFO]Style Transfer Complete!')

print ('[INFO]Displaying final image!')
# Display final image
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))
ax1.imshow(im_convert(content))
ax2.imshow(im_convert(target))
ax1.axis('off'); ax2.axis('off')
plt.show()
