import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.datasets as dset
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

transform = transforms.Compose(
    [transforms.ToTensor(), transforms.Normalize([0.5], [0.5])])
mnist = dset.MNIST('./data', train=True, transform=transform, download=True)
dataloader = DataLoader(mnist, batch_size=128, shuffle=True)

noise_dim = 100


class Generator(nn.Module):
    def __init__(self, n_embed):
        super(Generator, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(noise_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Linear(256, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Linear(512, n_embed),
            nn.Tanh()
        )

    def forward(self, x):
        return self.layers(x)


class Discriminator(nn.Module):
    def __init__(self, n_embed):
        super(Discriminator, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(n_embed, 512),
            nn.ReLU(0),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.layers(x)


lr = 1e-4
epochs = 7
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
batch = next(iter(dataloader))
# print(batch[0][0].view(-1).size(0))
img_dim = batch[0][0].view(-1).size(0)
d = Discriminator(img_dim).to(device)
g = Generator(img_dim).to(device)

d_optimizer = optim.AdamW(params=d.parameters(), lr=lr)
g_optimizer = optim.AdamW(params=g.parameters(), lr=lr)

# output will always be the labels (output_label, real_label)
criterion = nn.BCELoss()

for epoch in range(epochs):
    for batch, (data, target) in enumerate(dataloader):
        # Discriminator
        real_img = data.view(data.size(0), -1).to(device)
        real_label = torch.ones(data.size(0), 1).to(device)  # 32 by 1
        fake_label = torch.zeros(data.size(0), 1).to(device)  # 32 by 1
        real_out = d(real_img)
        d_loss_real = criterion(real_out, real_label)

        noise = torch.randn((data.size(0), noise_dim)).to(device)
        new_img = g(noise)
        # Need to detach here otherwise the generator could be updated
        fake_out = d(new_img.detach())
        d_loss_fake = criterion(fake_out, fake_label)

        d_loss = d_loss_fake + d_loss_real
        d_optimizer.zero_grad(set_to_none=True)
        d_loss.backward()
        d_optimizer.step()

        # Generator
        noise = torch.randn((data.size(0), noise_dim)).to(device)
        fake_img = g(noise)
        fake_out = d(fake_img)

        g_loss = criterion(fake_out, real_label)
        g_optimizer.zero_grad(set_to_none=True)
        g_loss.backward()
        g_optimizer.step()
    print(f'Epoch {epoch} disciminator loss {d_loss} generator loss {g_loss}')

g.eval()
with torch.no_grad():
    noise = torch.randn((data.size(0), noise_dim)).to(device)
    new_img = g(noise)
    print(new_img.shape)
    plt.imshow(new_img[0].cpu().view(28, 28))

latent_dim = 100  # Size of noise vector
num_images = 10   # Number of images to generate

# Generate noise vectors
noise = torch.randn(num_images, latent_dim, device=device)
g.eval()
# Generate images
with torch.no_grad():
    generated_images = g(noise)
    generated_images = generated_images.view(-1, 28, 28)

# Create figure and axis objects
fig, axes = plt.subplots(1, 10, figsize=(20, 2))

for i, ax in enumerate(axes):
    ax.imshow(generated_images[i].cpu())
    ax.axis('off')

plt.show()
