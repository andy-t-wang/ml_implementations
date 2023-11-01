
import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import torch.nn as nn
from torch.nn import functional as F
import matplotlib.pyplot as plt

# Define transformation
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])

# Load MNIST dataset
train_set = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
test_set = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform)

# Create data loaders
train_loader = DataLoader(train_set, batch_size=64, shuffle=True)
test_loader = DataLoader(test_set, batch_size=64, shuffle=False)

class CNN(nn.Module):
  def __init__(self, numFeatures):
    super(CNN, self).__init__()
    self.net = nn.Sequential(
      nn.Conv2d(numFeatures, 20, 5),
      nn.ReLU(),
      nn.MaxPool2d(kernel_size=2),

      nn.Conv2d(20, 50, 5),
      nn.ReLU(),
      nn.MaxPool2d(kernel_size=2),

      nn.Flatten(),
      nn.Linear(800, 500),
      nn.ReLU(),
      nn.Linear(500, 10),
    )

  def forward(self, x):
    x = self.net(x)
    return x

learning_rate = 1e-4
epochs = 5

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = CNN(1).to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
criterion = nn.CrossEntropyLoss()

# Loop through the data
for epoch in range(epochs):
  for batch_idx, (data, target) in enumerate(train_loader):
      data, target = data.to(device), target.to(device)
      output = model(data)
      loss = criterion(output, target.long())
      optimizer.zero_grad(set_to_none=True)
      loss.backward()
      optimizer.step()
  print(f"Epoch {epoch + 1}, Loss: {loss.item()}")

total_correct = 0

with torch.no_grad():  # Disable gradient calculation to save memory and speed up
  for batch_idx, (data, target) in enumerate(test_loader):
    model.eval()
    data, target = data.to(device), target.to(device)
    output = model(data)
     # Compute other metrics, e.g., accuracy
    _, predicted = torch.max(output.data, 1)
    total_correct += (predicted == target).sum().item()
    if batch_idx == 0:
      plt.imshow(data[0].cpu().squeeze().numpy(), cmap='gray')
      plt.title(f'Actual: {target[0].item()}, Predicted: {predicted[0].item()}')
      plt.show()
avg_accuracy = total_correct / len(test_loader.dataset)

print(avg_accuracy)