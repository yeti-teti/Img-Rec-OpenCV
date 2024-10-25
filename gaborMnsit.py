# Saugat Malla
# Extension

# Importing necessary libraries
import sys
import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np
import cv2

# Define Gabor filter bank
def gabor_filter_bank():
    filters = []
    theta = [0, np.pi/4, np.pi/2, 3*np.pi/4]
    frequencies = [0.1, 0.3, 0.5]
    for t in theta:
        for f in frequencies:
            kernel = cv2.getGaborKernel((5, 5), 1.0, t, f, 0.5, 0, ktype=cv2.CV_32F)
            filters.append(kernel)
    return np.array(filters)

# Neural Network
class MyNetwork(nn.Module):
    def __init__(self):
        super(MyNetwork, self).__init__()
        # Define the Gabor filter bank layer
        self.gabor_filter = nn.Conv2d(1, 10, kernel_size=5)  # 10 Gabor filters
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50,10)
    
    def forward(self, x):
        # Apply Gabor filter bank
        x = F.relu(F.max_pool2d(self.gabor_filter(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)

# Train function
def train(network, train_loader, epoch, train_losses, train_counter, optimizer, log_interval):
    network.train()

    for batch_idx, (data, target) in enumerate(train_loader):
        optimizer.zero_grad()
        output = network(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()

        if batch_idx % log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(epoch, batch_idx * len(data), len(train_loader.dataset),100. * batch_idx / len(train_loader), loss.item()))
        
        train_losses.append(loss.item())
        train_counter.append((batch_idx*64) + ((epoch-1)*len(train_loader.dataset)))
        torch.save(network.state_dict(), 'results/gabor_model.pth')
        torch.save(optimizer.state_dict(), 'results/gabor_optimizer.pth')

# Test function
def test(network, test_loader, test_losses):
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

    network = MyNetwork()
    optimizer = optim.SGD(network.parameters(), lr=learning_rate, momentum=momentum)

    # Training the model
    train_losses = []
    train_counter = []
    test_losses = []
    test_counter = [i*len(train_loader.dataset) for i in range(n_epochs + 1)]

    test(network, test_loader, test_losses)
    for epoch in range(1, n_epochs + 1):
        train(network, train_loader, epoch, train_losses, train_counter, optimizer, log_interval)
        test(network, test_loader, test_losses)

    # Evaluating the models performance
    fig = plt.figure()
    plt.plot(train_counter, train_losses, color='blue')
    plt.scatter(test_counter, test_losses, color='red')
    plt.legend(['Train Loss', 'Test Loss'], loc = 'upper right')
    plt.xlabel('number of training examples seen')
    plt.ylabel('negative log likelihood loss')
    plt.show()
    fig

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
