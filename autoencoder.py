import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

transform = transforms.Compose(
    [transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
train_dataset = datasets.MNIST(
    '.', train=True, download=True, transform=transform)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(train_dataset, batch_size=32, shuffle=False)


class AutoEncoder(nn.Module):
    def __init__(self, n_embed):
        super(AutoEncoder, self).__init__()
        self.Encoder = nn.Sequential(
            nn.Linear(n_embed, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
        )
        self.Decoder = nn.Sequential(
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, n_embed),
            # Simoid here since we have pixels between 0-1 and want the final pixels of size n_embed to be between 0-1 sigmoid does this RELU does [0, inf]
            nn.Sigmoid(),
        )

    def forward(self, x):
        encoded = self.Encoder(x)
        decoded = self.Decoder(encoded)
        return decoded


lr = 1e-3
epochs = 5

model = AutoEncoder(784).to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
criterion = nn.MSELoss()
for epoch in range(epochs):
    for batch, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        img = data.view(data.size(0), -1)
        y = model(img)

        loss = criterion(y, img)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
    print(f'Epoch {epoch} loss {loss}')

model.eval()
with torch.no_grad():
    for batch, (data, target) in enumerate(test_loader):
        f, axarr = plt.subplots(2, 2)
        if batch < 2:
            img = data.view(data.size(0), -1)  # 32, 784
            output = model(img)
            axarr[0, 0].imshow(img.view(-1, 28, 28)[0].squeeze())
            axarr[0, 1].imshow(output.view(-1, 28, 28)[0].squeeze())
        else:
            break

img = next(iter(train_loader))
print(len(img))
print(img[1][0])
plt.imshow(img[0][0].squeeze())
