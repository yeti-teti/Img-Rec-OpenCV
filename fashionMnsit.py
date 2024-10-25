# Saugat Malla
# Task 4

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
    def __init__(self, num_layers=2, num_neurons=120, dropout_rate=0.2):
        super(MyNetwork, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, kernel_size=5)
        self.pool = nn.MaxPool2d(2,2)
        self.conv2 = nn.Conv2d(6, 16, kernel_size=5)
        self.conv2_drop = nn.Dropout2d(p=dropout_rate)
        self.fc1 = nn.Linear(16*4*4, num_neurons)
        self.fc2 = nn.Linear(num_neurons,84)
        self.fc3 = nn.Linear(84,10)
    
    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.conv2_drop(x)
        x = x.view(-1, 16*4*4)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.dropout(x, training=self.training)
        x = self.fc3(x)
        return F.log_softmax(x, dim=1)

# Train function
def train(network, train_loader, loss_fn, epoch, train_losses, train_counter, optimizer, log_interval):

    network.train()

    for batch_idx, (data, target) in enumerate(train_loader):

        optimizer.zero_grad()

        output = network(data)

        loss = loss_fn(output, target)
        loss.backward()

        optimizer.step()

        if batch_idx % log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(epoch, batch_idx * len(data), len(train_loader.dataset),100. * batch_idx / len(train_loader), loss.item()))
        
        train_losses.append(loss.item())
        train_counter.append((batch_idx*64) + ((epoch-1)*len(train_loader.dataset)))
        
        

# Test function
def test(network, loss_fn, test_loader, test_losses):
    network.eval()
    correct = 0
    running_loss = 0.0
    with torch.no_grad():
        for data, target in test_loader:
            output = network(data)
            test_loss = loss_fn(output, target)
            running_loss += test_loss.item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()

    running_loss /= len(test_loader.dataset)
    test_losses.append(running_loss)
    accuracy = 100. * correct / len(test_loader.dataset)
    print('\nTest set: Avg. loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        running_loss, correct, len(test_loader.dataset), accuracy))
    
    return accuracy


# Main function
def main(argv):
    # Hyperparameters
    n_epochs = 5
    batch_size_train = 64
    batch_size_test = 64
    learning_rate = 0.01
    momentum = 0.5
    log_interval = 10
    torch.backends.cudnn.enabled = False
    random_seed = 1
    torch.manual_seed(random_seed)

    # Getting the dataset
    train_loader = torch.utils.data.DataLoader(
        torchvision.datasets.FashionMNIST('dataset', train=True, download=True,   
                                transform=torchvision.transforms.Compose([
                                    torchvision.transforms.ToTensor(), 
                                    torchvision.transforms.Normalize(
                                        (0.5,),(0.5,) 
                                    )
                                ])),
        batch_size = batch_size_train, shuffle=True
    )

    test_loader = torch.utils.data.DataLoader(
        torchvision.datasets.FashionMNIST('dataset', train=False,download=True,
                                transform=torchvision.transforms.Compose([
                                    torchvision.transforms.ToTensor(),
                                    torchvision.transforms.Normalize(
                                        (0.5,),(0.5,)
                                    )
                                ])),
        batch_size = batch_size_test, shuffle=True                               
    )

    # Class labels
    classes = ('T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
        'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle Boot')
    
    # Report split sizes
    print('Training set has {} instances'.format(len(train_loader)))
    print('Validation set has {} instances'.format(len(test_loader)))

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
    # %%
    plt.show()


    # Variations to explore
    # num_layers_list = [2,3,5]
    # num_neurons_list = [20,120,150]
    # dropout_rate_list = [0.2,0.3,0.5]
    # learning_rate_list = [0.01,0.001, 0.001]
    # regularization_list = [None,'dropout']

    num_layers_list = [5]
    num_neurons_list = [120]
    dropout_rate_list = [0.3,]
    learning_rate_list = [0.01]
    regularization_list = [None]

    variations = []
    for num_layers in num_layers_list:
        for num_neurons in num_neurons_list:
            for dropout_rate in dropout_rate_list:
                for learning_rate in learning_rate_list:
                    for regularization in regularization_list:
                        variations.append({
                            'num_layers': num_layers,
                            'num_neurons': num_neurons,
                            'dropout_rate': dropout_rate,
                            'learning_rate': learning_rate,
                            'regularization': regularization
                        })


    # Training and evaluating variations
    results = []
    for i, variation in enumerate(variations):
        print(f'Variation {i+1}/{len(variations)}:')
        print(variation)

        network = MyNetwork(
            num_layers=variation['num_layers'],
            num_neurons=variation['num_neurons'],
            dropout_rate=variation['dropout_rate']
        )

        if variation['regularization'] == 'dropout':
            network.conv2_drop = nn.Dropout2d(p=variation['dropout_rate'])
        elif variation['regularization'] == 'l2':
            # Add L2 regularization
            weight_decay = 1e-5
            optimizer = optim.SGD(network.parameters(), lr=variation['learning_rate'], momentum=momentum, weight_decay=weight_decay)
        else:
            optimizer = optim.SGD(network.parameters(), lr=variation['learning_rate'], momentum=momentum)

        loss_fn = torch.nn.CrossEntropyLoss()

        # Training the model
        train_losses = []
        train_counter = []
        test_losses = []
        test_counter = []
        accuracies = []

        for epoch in range(1, n_epochs + 1):
            train(network, train_loader,loss_fn, epoch, train_losses, train_counter, optimizer, log_interval)
            accuracy = test(network, loss_fn, test_loader, test_losses)
            test_counter.append(epoch * len(train_loader.dataset))
            accuracies.append(accuracy)

        results.append({
            'variation': variation,
            'train_losses': train_losses,
            'test_losses': test_losses,
            'accuracies': accuracies
        })

    # Display sample predictions
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

    # Rank variations based on average accuracy
    ranked_results = sorted(results, key=lambda x: max(x['accuracies']), reverse=True)

    # Print or save ranked results
    for i, result in list(enumerate(ranked_results))[:5]:
        print(f"Rank {i+1}:")
        print(result['variation'])

if __name__ == "__main__":
    main(sys.argv)
