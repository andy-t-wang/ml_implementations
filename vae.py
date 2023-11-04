import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

transform = transforms.Compose(
    [transforms.ToTensor()])
train_dataset = datasets.MNIST(
    '.', train=True, download=True, transform=transform)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(train_dataset, batch_size=32, shuffle=False)


class VAE(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(VAE, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU(),
        )
        self.mean = nn.Linear(128, output_dim)
        self.distribution = nn.Linear(128, output_dim)
        self.decoder = nn.Sequential(
            nn.Linear(output_dim, 128),
            nn.ReLU(),
            nn.Linear(128, input_dim),
            nn.Sigmoid(),
        )

    def reparameterization(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return mu + eps*std

    def decode(self, z):
        return self.decoder(z)

    def forward(self, x):
        encoded = self.encoder(x)
        mu = self.mean(encoded)
        logvar = self.distribution(encoded)
        z = self.reparameterization(mu, logvar)
        return self.decode(z), mu, logvar


lr = 1e-3
epochs = 5

img_dim = next(iter(train_loader))[0][0].view(-1).size(0)
model = VAE(img_dim, img_dim).to(device)
optimizer = optim.AdamW(model.parameters(), lr=lr)
criterion = nn.BCELoss()


def VAELoss(output, target, mu, logvar):
    BCE = criterion(output, target)
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return BCE + KLD


for epoch in range(epochs):
    for i, (data, _) in enumerate(train_loader):
        data = data.to(device)
        optimizer.zero_grad()
        # flatten the images
        output, mu, logvar = model(data.view(-1, img_dim))
        loss = VAELoss(output, data.view(-1, img_dim), mu, logvar)
        loss.backward()
        optimizer.step()
        if i % 100 == 0:
            print(f"Epoch {epoch}, step {i}, loss {loss}")

model.eval()
with torch.no_grad():
    for batch, (data, target) in enumerate(test_loader):
        f, axarr = plt.subplots(2, 2)
        if batch < 2:
            img = data.view(data.size(0), -1)  # 32, 784
            output, _, d = model(img)
            axarr[0, 0].imshow(img.view(-1, 28, 28)[0].squeeze())
            axarr[0, 1].imshow(output.view(-1, 28, 28)[0].squeeze())
        else:
            break

img = next(iter(train_loader))
print(len(img))
print(img[1][0])
plt.imshow(img[0][0].squeeze())
