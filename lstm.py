import torch
import torch.nn as nn
from torch.nn import functional as F
import torch.nn.init as init

# n_embed represents the internal state size
n_embed = 128
block_size = 8  # what is the maximum context length for predictions?
batch_size = 32  # how many independent sequences will we process in parallel?
device = 'cuda' if torch.cuda.is_available() else 'cpu'
token_size = 65

torch.manual_seed(1337)

# wget https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt
with open('input.txt', 'r', encoding='utf-8') as f:
    text = f.read()

# here are all the unique characters that occur in this text
chars = sorted(list(set(text)))
vocab_size = len(chars)
# create a mapping from characters to integers
stoi = {ch: i for i, ch in enumerate(chars)}
itos = {i: ch for i, ch in enumerate(chars)}
# encoder: take a string, output a list of integers
def encode(s): return [stoi[c] for c in s]
# decoder: take a list of integers, output a string
def decode(l): return ''.join([itos[i] for i in l])


# Train and test splits
data = torch.tensor(encode(text), dtype=torch.long)
n = int(0.9*len(data))  # first 90% will be train, rest val
train_data = data[:n]
val_data = data[n:]


def get_batch(split):
    # generate a small batch of data of inputs x and targets y
    data = train_data if split == 'train' else val_data
    ix = torch.randint(token_size, (batch_size,))
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])
    x, y = x.to(device), y.to(device)
    return x, y


class RNN(nn.Module):
    def __init__(self, n_embed, vocab_size):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, n_embed)  # Embedding layer
        self.LSTM = nn.LSTM(n_embed, n_embed, batch_first=True)
        self.linear = nn.Linear(n_embed, vocab_size)

    def forward(self, x, h_prev=None):
        # After tokenization always need to embed
        x = self.embed(x)  # Convert token indices to embeddings
        output, (h_n, c_n) = self.LSTM(x, h_prev)
        y = self.linear(output)
        return output, y, (h_n, c_n)

    def generate_text(self, seed, length=100, temperature=1.0):
        model.eval()  # Set the model to evaluation mode
        # IT wont' train the model anymore but you still need to do each step RNN
        with torch.no_grad():  # No need to track gradients during generation
            generated_text = seed

            # we don't need the hidden and cell state since this is done running already
            for _ in range(length):
                print(_)
                x = torch.tensor(encode(generated_text)).to(device).unsqueeze(
                    0)
                h = (torch.zeros(1, token_size, n_embed).to(
                    device), torch.zeros(1, token_size, n_embed).to(
                    device))
                output, y, (h_n, c_n) = model(x, h)
                # Apply temperature to logits and get probabilities
                probs = F.softmax(y / temperature, dim=-1).squeeze()

                # Sample the next character
                next_token = torch.multinomial(probs, num_samples=1).item()
                generated_text += itos[next_token]

                # Use the predicted character as the next input
                # x = torch.tensor([[next_token]]).to(device)

            return generated_text


model = RNN(n_embed=n_embed, vocab_size=token_size)
m = model.to(device)

# training loop
lr = 0.001
criterion = nn.CrossEntropyLoss()

optimizer = torch.optim.Adam(model.parameters(), lr=lr)
eval_iters = 100
n_iters = 500

for i in range(n_iters):
    xb, yb = get_batch('train')
    print(xb.shape)
    h_prev = (torch.zeros(1, batch_size, n_embed).to(
        device), torch.zeros(1, batch_size, n_embed).to(
        device))  # Initial hidden state

    optimizer.zero_grad()

    # Forward pass
    output, y, (h_n, c_n) = model(xb, h_prev)
    loss = criterion(y.view(-1, vocab_size), yb.view(-1))

    # Backward pass and optimization
    loss.backward()
    optimizer.step()

    # Print loss every `eval_iters` iterations
    if (i+1) % eval_iters == 0:
        print(f'Iteration {i+1}/{n_iters}, Loss: {loss.item()}')

# def call


# Example usage:
seed = "To be or not to"
predicted_text = m.generate_text(seed, length=200, temperature=0.8)
print(predicted_text)