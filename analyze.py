# Saugat Malla
# Task 2 [A-B]

# Importing necessary libraries
import torch 
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import cv2
import numpy as np

# Define batch size and random seed
batch_size_train = 64
random_seed = 1
torch.manual_seed(random_seed)

# Define the network architecture
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

# Load the trained network
network = Net()
network_state_dict = torch.load('results/model.pth')
network.load_state_dict(network_state_dict)

# Print the model structure
print(network)

# Analyze the first layer
print("Filter Weight",network.conv1.weight)
print("Filter shape", network.conv1.weight.shape)

# Getting weights from the first layer
filters = network.conv1.weight

# Normalizing filter values for visualization. [0-1]
f_min, f_max = filters.min(), filters.max()
filters = (filters - f_min) / (f_max - f_min)

# Visualization of the filters
n_filters = 10
fig = plt.figure()
for i in range(n_filters):
    plt.subplot(3,4,i+1)
    plt.title("Filter {}".format(i))
    plt.imshow(filters[i][0].detach().numpy())
    plt.xticks([])
    plt.yticks([])
plt.show()

# Getting the dataset
train_loader = torch.utils.data.DataLoader(
    torchvision.datasets.MNIST('dataset', train=True, download=True,   
                               transform=torchvision.transforms.Compose([
                                   torchvision.transforms.ToTensor(), 
                                   torchvision.transforms.Normalize(
                                       (0.1307,),(0.3081,) 
                                   )
                               ])),
    batch_size = batch_size_train, shuffle=True
)

# Applying filters to the first training example image
kernel1 = np.ones((5,5), np.float32) / 30

fig = plt.figure(figsize=(10, 8))
with torch.no_grad():
    for images, labels in train_loader:
        img = images[0][0]  # Extracting the first image from the batch
        print(labels[0])
        for i in range(n_filters):
            filtered_img = cv2.filter2D(src=img.numpy(), ddepth=-1, kernel=filters[i][0].numpy())
            plt.subplot(2, 5, i+1)
            plt.imshow(filtered_img, cmap='gray')
            plt.title("Filter {}".format(i))
            plt.xticks([])
            plt.yticks([])
        break  # Break after processing the first batch
plt.show()
