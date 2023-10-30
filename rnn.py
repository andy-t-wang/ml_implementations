import torch
import torch.nn as nn
from torch.nn import functional as F

# Hyperparameters
embed_dim = 64
hidden_dim = 128
seq_length = 30
batch_size = 32
learning_rate = 0.001
epochs = 10
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# wget https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt
with open('input.txt', 'r', encoding='utf-8') as f:
    text = f.read()

torch.manual_seed(1337)

# Assuming you've loaded the data in the `text` variable.
chars = sorted(list(set(text)))
vocab_size = len(chars)
stoi = {ch: i for i, ch in enumerate(chars)}
itos = {i: ch for i, ch in enumerate(chars)}


def encode(s): return [stoi[c] for c in s]
def decode(l): return ''.join([itos[i] for i in l])


data = torch.tensor(encode(text), dtype=torch.long)
n = int(0.9*len(data))
train_data = data[:n]
val_data = data[n:]


def get_batch(data, seq_length, batch_size):
    start_indices = torch.randint(0, len(data) - seq_length - 1, (batch_size,))
    sequences = [data[i: i+seq_length] for i in start_indices]
    next_chars = [data[i+1: i+seq_length+1] for i in start_indices]
    return torch.stack(sequences), torch.stack(next_chars)

# Model


class CharRNN(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim):
        super(CharRNN, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.rnn = nn.RNN(embed_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, vocab_size)

    def forward(self, x, hidden_state=None):
        x = self.embedding(x)
        out, h = self.rnn(x, hidden_state)
        out = self.fc(out)
        return out, h

    @torch.no_grad()
    def generate_text(self, seed, length=100):
        model.eval()  # Set the model to evaluation mode
        generated = seed
        seq = torch.tensor(encode(generated)).unsqueeze(0)

        hidden = None
        for _ in range(length):
            out, hidden = self(seq, hidden)
            prob = F.softmax(out[:, -1, :], dim=-1).detach()
            next_char = torch.multinomial(prob, num_samples=1)
            generated += itos[next_char.item()]
            seq = torch.cat(
                [seq[:, 1:], torch.tensor([[next_char.item()]])], dim=1)
        model.train()
        return generated


model = CharRNN(vocab_size, embed_dim, hidden_dim).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# Training loop
for epoch in range(epochs):
    model.train()
    for _ in range(1000):  # Assuming 1000 batches per epoch
        x, y = get_batch(train_data, seq_length, batch_size)
        optimizer.zero_grad()
        output, _ = model(x)
        loss = criterion(output.transpose(1, 2), y)
        loss.backward()
        optimizer.step()
    print(f"Epoch {epoch + 1}, Loss: {loss.item()}")

# Generate text
model.eval()
seed = "To be or not to"
generated_text = model.generate_text(seed, 200)
print(generated_text)
