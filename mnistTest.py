# Saugat Malla
# Task 1 [E]

# Importing necessary libraries
import torch 
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt

# Setting random seed for reproducibility
random_seed = 1
torch.manual_seed(random_seed)

# Network definition
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

# Loading the dataset
batch_size_test = 1000
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

# Getting one batch of test data
examples = enumerate(test_loader)
batch_idx, (example_data, example_target) = next(examples)

# Loading the pre-trained network
network = Net()
network_state_dict = torch.load('results/model.pth')
network.load_state_dict(network_state_dict)
network.eval()

# Iterating through first 10 examples in the test set
for batch_idx, (data, target) in enumerate(test_loader):
    if batch_idx >= 11:
        break  # Stop after the first 10 examples
    
    output = network(data)
    pred = output.data.max(1, keepdim=True)[1]
    
    # Printing the network output values, index of max output value, and correct label
    print("Example", batch_idx + 1)
    print("Output values:", [f"{val:.2f}" for val in output[0]])
    print("Predicted label:", pred[0].item())
    print("Correct label:", target[0].item())
    print()
    
    # Plotting the first 10 digits with predictions
    if batch_idx < 10:
        plt.subplot(4, 3, batch_idx + 1)
        plt.tight_layout()
        plt.imshow(data[0][0], cmap='gray')
        plt.title(f"Prediction: {pred[0].item()}")
        plt.axis('off')
        plt.xticks([])
        plt.yticks([])

plt.show()
