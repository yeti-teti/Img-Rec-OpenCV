# Saugat Malla
# Task 1 [F]

# Loads the files from the directory and make prediction

# Importing necessary libraries
import sys
import torch
import torchvision 
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
import glob
import os


class ImgTransform:
    def __init__(self):
        pass

    def __call__(self, x):
        # Convert to grayscale
        x = torchvision.transforms.functional.rgb_to_grayscale(x)

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
        self.fc2 = nn.Linear(50,10)
    
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

    random_seed = 1
    torch.manual_seed(random_seed)

    # Loading the pre-trained network
    network = Net()
    network_state_dict = torch.load('results/model.pth')
    network.load_state_dict(network_state_dict)

    idx = 0
    for filename in glob.glob("dataset/additional_numbers/*"):

        # Load and preprocess the test image

        image = Image.open(filename)
        transform = torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(),
            ImgTransform(),
            torchvision.transforms.Normalize((0.1307,), (0.3081,))
        ])
        img = transform(image).unsqueeze(0)  # Add a batch dimension
        img.requires_grad = True  # Ensure gradients are calculated for the input

        # Forward pass through the network
        output = network(img)

        # Obtain the predicted class
        pred = output.data.max(1, keepdim=True)[1]

        target = os.path.basename(filename)[0]
        

        plt.subplot(4, 3, idx+1)
        plt.tight_layout()
        
        plt.title("Ground Truth: {}".format(target))
        plt.imshow(image, cmap='gray', interpolation='none')
        plt.xticks([])
        plt.yticks([])

        idx += 1
    
    plt.show()

if __name__ == "__main__":
    main(sys.argv)
