# Saugat Malla

# Task 1 [A - D]

# Importing necessary libraries
import sys
import torch
import torchvision
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

## Neural Network
class MyNetwork(nn.Module):
    def __init__(self):
        super(MyNetwork, self).__init__()
        # Define convolutional layers and fully connected layers
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
def train(network, train_loader, epoch, train_losses, train_counter, optimizer, log_interval):
    # Set the model to training mode
    network.train()

    for batch_idx, (data, target) in enumerate(train_loader):
        optimizer.zero_grad()
        output = network(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()

        if batch_idx % log_interval == 0:
            # Print training statistics
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(epoch, batch_idx * len(data), len(train_loader.dataset),100. * batch_idx / len(train_loader), loss.item()))
        
        # Record training loss and iteration
        train_losses.append(loss.item())
        train_counter.append((batch_idx*64) + ((epoch-1)*len(train_loader.dataset)))
        # Save model and optimizer state
        torch.save(network.state_dict(), 'results/model.pth')
        torch.save(optimizer.state_dict(), 'results/optimizer.pth')

# Test function
def test(network, test_loader, test_losses):
    # Set the model to evaluation mode
    network.eval()
    test_loss = 0
    correct = 0

    with torch.no_grad():
        for data, target in test_loader:
            output = network(data)
            test_loss += F.nll_loss(output, target, size_average=False).item()
            pred = output.data.max(1, keepdim=True)[1]
            correct += pred.eq(target.data.view_as(pred)).sum()
    
    test_loss /= len(test_loader.dataset)
    test_losses.append(test_loss)

    # Print test set results
    print('\nTest set: Avg. loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(test_loss, correct, len(test_loader.dataset),100. * correct / len(test_loader.dataset)))


# Main function
def main(argv):
    # Hyperparameters
    n_epochs = 5
    batch_size_train = 64
    batch_size_test = 1000
    learning_rate = 0.01
    momentum = 0.5
    log_interval = 10
    torch.backends.cudnn.enabled = False
    random_seed = 1
    torch.manual_seed(random_seed)

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

    test_loader = torch.utils.data.DataLoader(
        torchvision.datasets.MNIST('dataset', train=False,download=True,
                                transform=torchvision.transforms.Compose([
                                    torchvision.transforms.ToTensor(),
                                    torchvision.transforms.Normalize(
                                        (0.1307,),(0.3081,)
                                    )
                                ])),
        batch_size = batch_size_test, shuffle=True                               
    )

    # Initial Analysis
    examples = enumerate(test_loader)
    batch_idx, (example_data, example_target) = next(examples)

    print("Data shape:", example_data.shape)


    # Visualization
    fig = plt.figure()

    for i in range(6):
        plt.subplot(2, 3, i+1)
        plt.tight_layout()
        
        plt.title("Ground Truth: {}".format(example_target[i]))
        plt.imshow(example_data[i][0], cmap='gray', interpolation='none')
        plt.xticks([])
        plt.yticks([])
    plt.show()

    # Initialize the neural network and optimizer
    network = MyNetwork()
    optimizer = optim.SGD(network.parameters(), lr=learning_rate, momentum=momentum)

    # Training the model
    train_losses = []
    train_counter = []
    test_losses = []
    test_counter = [i*len(train_loader.dataset) for i in range(n_epochs + 1)]

    # Evaluate the model before training
    test(network, test_loader, test_losses)
    
    # Perform training epochs
    for epoch in range(1, n_epochs + 1):
        train(network, train_loader, epoch, train_losses, train_counter, optimizer, log_interval)
        test(network, test_loader, test_losses)

    # Plotting training and test losses
    fig = plt.figure()
    plt.plot(train_counter, train_losses, color='blue')
    plt.scatter(test_counter, test_losses, color='red')
    plt.legend(['Train Loss', 'Test Loss'], loc = 'upper right')
    plt.xlabel('number of training examples seen')
    plt.ylabel('negative log likelihood loss')
    plt.show()
    fig

    # Visualize predictions
    with torch.no_grad():
        output = network(example_data)

    fig = plt.figure()
    for i in range(6):
        plt.subplot(2, 3, i+1)
        plt.tight_layout()
        plt.imshow(example_data[i][0], cmap='gray', interpolation='none')
        plt.title("Prediction: {}".format(output.data.max(1, keepdim=True)[1][i].item()))
        plt.xticks([])
        plt.yticks([])
    plt.show()

if __name__ == "__main__":
    main(sys.argv)
