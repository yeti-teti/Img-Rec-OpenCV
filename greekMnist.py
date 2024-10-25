# Saugat Malla
# Task 3 (Training)

# Importing necessary libraries
import sys
import torch
import torchvision 
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from PIL import Image

# Define a custom transformation for the Greek letter images
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

# Train function
def train(network, greek_train, optimizer, criterion):
    network.train()
    running_loss = 0.0
    correct = 0
    total = 0
    for batch_idx, (data, target) in enumerate(greek_train):
        optimizer.zero_grad()
        output = network(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        
        _, predicted = torch.max(output.data, 1)
        total += target.size(0)
        correct += (predicted == target).sum().item()

    accuracy = 100 * correct / total
    return running_loss / len(greek_train), accuracy

# Main function
def main(argv):
    # Hyperparameters
    n_epochs = 50
    batch_size_train = 5
    learning_rate = 0.001
    torch.manual_seed(1)

    # Getting the dataset ( DataLoader for the Greek dataset)
    greek_train = torch.utils.data.DataLoader(
            torchvision.datasets.ImageFolder( 'dataset/greek_train',
                                            transform = torchvision.transforms.Compose( [torchvision.transforms.ToTensor(),
                                                                                        GreekTransform(),
                                                                                        torchvision.transforms.Normalize(
                                                                                            (0.1307,), (0.3081,) ) ] ) ),
            batch_size = batch_size_train,
            shuffle = True )

    # Loading the pre-trained network and finetuning the final layer
    network = Net()
    network_state_dict = torch.load('results/model.pth')
    network.load_state_dict(network_state_dict)

    # Freezing the parameters for the whole network (except the last layer)
    for param in network.parameters():
        param.requires_grad = False

    # Enabling weight updates only in the last layer
    for param in network.fc2.parameters():
        param.requires_grad = True

    network.fc2 = nn.Linear(50,3)  # Changing the output layer to match the Greek dataset

    # Initial Analysis
    examples = enumerate(greek_train)
    batch_idx, (example_data, example_target) = next(examples)

    print("Data shape:", example_data.shape)

    # Visualization
    fig = plt.figure()

    for i in range(batch_size_train):
        plt.subplot(2, 3, i+1)
        plt.tight_layout()
        
        plt.title("Ground Truth: {}".format(example_target[i]))
        plt.imshow(example_data[i][0], cmap='gray', interpolation='none')
        plt.xticks([])
        plt.yticks([])
    plt.show()

    # Define optimizer and loss function
    optimizer = optim.Adam(network.fc2.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss()

    # Training loop
    train_losses = []
    train_accuracies = []
    for epoch in range(n_epochs):
        train_loss, train_accuracy = train(network, greek_train, optimizer, criterion)
        train_losses.append(train_loss)
        train_accuracies.append(train_accuracy)
        print(f"Epoch {epoch + 1}/{n_epochs}, Loss: {train_loss}, Accuracy: {train_accuracy}%")

    # Evaluating the model's performance
    fig = plt.figure()
    # Plot training loss
    plt.plot(train_losses, label='Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()

if __name__ == "__main__":
    main(sys.argv)
