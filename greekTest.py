# Saugat Malla
# Task 3 (Testing)

# Importing necessary libraries
import sys
import torch
import torchvision 
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torchviz import make_dot

# Transform greek letter images
class GreekTransform:
    def __init__(self):
        pass

    def __call__(self, x):
        # Convert to grayscale
        x = torchvision.transforms.functional.rgb_to_grayscale(x)
        # Apply affine transformation
        x = torchvision.transforms.functional.affine(x, 0, (0,0), 36/128, 0)
        # Center crop to 28x28
        x = torchvision.transforms.functional.center_crop(x, (28, 28))
        # Invert colors
        return torchvision.transforms.functional.invert(x)

## Neural Network
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50,3)
    
    def forward(self, x):
        # Define the forward pass
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)

def main(argv):
    # Loading the pre-trained network
    network = Net()
    network_state_dict = torch.load('results/greek_model.pth')
    network.load_state_dict(network_state_dict)

    # Load and preprocess the test image
    image = Image.open("dataset/greek_train/gamma/gamma_002.png")
    transform = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
        GreekTransform(),
        torchvision.transforms.Normalize((0.1307,), (0.3081,))
    ])
    alphaImg = transform(image).unsqueeze(0)  # Add a batch dimension
    alphaImg.requires_grad = True  # Ensure gradients are calculated for the input

    # Forward pass through the network
    output = network(alphaImg)

    # Obtain the predicted class
    pred = output.data.max(1, keepdim=True)[1]
    if pred.item() == 0:
        print("Alpha")
    elif pred.item() == 1:
        print("Beta")
    else:
        print("Gamma")

    # Generate a computational graph
    dot = make_dot(output, params=dict(network.named_parameters()))
    dot.render("CNN", format="png", cleanup=True)

if __name__ == "__main__":
    main(sys.argv)
