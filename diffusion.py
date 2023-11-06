import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Subset
import torch.nn as nn
from torch.nn import functional as F
import matplotlib.pyplot as plt
import random


class Diffusion(nn.Module):
    def __init__(self, num_channels):
        super(Diffusion, self).__init__()
        self.model = nn.Sequential(
            nn.Conv2d(num_channels, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            # Image data so you want to add some non linearity
            nn.ReLU(),
            nn.Conv2d(64, num_channels, kernel_size=3, padding=1)
        )

    def forward(self, x):
        return self.model(x)


transform = transforms.Compose([transforms.ToTensor()])
# Load MNIST dataset
train_set = torchvision.datasets.MNIST(
    root='./data', train=True, download=True, transform=transform)
test_set = torchvision.datasets.MNIST(
    root='./data', train=False, download=True, transform=transform)
indices = random.sample(range(len(train_set)), 1000)

# Use the Subset class to get a subset of the dataset
subset_dataset = Subset(train_set, indices)
# Create data loaders
train_loader = DataLoader(subset_dataset, batch_size=1, shuffle=True)
test_loader = DataLoader(test_set, batch_size=1, shuffle=False)

channels = train_set[0][0].shape[0]
lr = 1e-3
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = Diffusion(channels).to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
criterion = nn.MSELoss()
epochs = 6
diffusion_steps = 5

for epoch in range(epochs):
    for image_num, (data, _) in enumerate(train_loader):
        data = data.to(device)
        noisy_images = []
        noisy = data.clone()
        for i in range(diffusion_steps):
            noisy = noisy + torch.randn_like(data) * 0.15
            noisy_images.append(noisy.squeeze(1))
        stacked_noisy_images = torch.stack(noisy_images, dim=0).to(device)
        actual_images = torch.cat(
            [data, stacked_noisy_images[:diffusion_steps - 1,]], dim=0).to(device)
        delta = model(stacked_noisy_images)
        loss = criterion(stacked_noisy_images - delta, actual_images)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
    print(f"Epoch {epoch} loss {loss}")

for epoch in range(epochs):
    for image_num, (data, _) in enumerate(train_loader):
        data = data.to(device)

        # Initialize the current noisy version as the original data
        current_noised = data.clone()

        optimizer.zero_grad(set_to_none=True)

        total_loss = 0
        for i in range(diffusion_steps):
            # Previous state of the image before adding noise
            prev_image = current_noised.clone()

            # Add noise
            current_noised = current_noised + torch.randn_like(data) * 0.5

            # Predict the delta
            delta = model(current_noised)

            # Calculate loss for this step
            loss = criterion(current_noised - delta, prev_image)
            total_loss += loss

            # Backpropagate the loss
            loss.backward()

        # Update model weights
        optimizer.step()

        # Print average loss for the diffusion steps
    print(f"Epoch {epoch}, Average Loss {total_loss/diffusion_steps:.6f}")

model.eval()
f, axarr = plt.subplots(3, 1)
axarr[0].imshow(test_set[1][0][0].squeeze().numpy())
test_item = test_set[1][0].to(device) + torch.randn_like(data.squeeze())
axarr[1].imshow(test_item.cpu().squeeze().numpy(), )
with torch.no_grad():
    for i in range(diffusion_steps):
        test_item -= model(test_item)
model.train()
axarr[2].imshow(test_item.cpu().squeeze().numpy(), )
plt.show()
