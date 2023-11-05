import torch
import torch.nn.functional as F
import torch_geometric.nn as nn
from torch_geometric.datasets import Planetoid

# Load the Cora dataset
dataset = Planetoid(root='/tmp/Cora', name='Cora', split="public")
data = dataset[0]

# The masks for train, val, and test are already provided
train_mask = data.train_mask
val_mask = data.val_mask
test_mask = data.test_mask


class GNN(torch.nn.Module):
    # This NN takes in nodes and predicts a class for the node. Uses the edges to learn
    def __init__(self, nodes, classes):
        super(GNN, self).__init__()
        # First convolutional layer
        self.conv1 = nn.GCNConv(dataset.num_node_features, 156)
        self.conv2 = nn.GCNConv(156, dataset.num_classes)

    def forward(self, x, edges):
        x = self.conv1(x, edges)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)

        x = self.conv2(x, edges)
        return F.log_softmax(x, dim=1)


nodes = dataset.num_node_features
edges = dataset.num_edge_features
classes = dataset.num_classes
print(nodes)
lr = 1e-2
epochs = 200
# print(dataset.get(0))
criterion = torch.nn.NLLLoss()
model = GNN(nodes, classes)
optimizer = torch.optim.AdamW(params=model.parameters(), lr=lr)

for epoch in range(epochs):
    x = data.x
    targets = data.y
    edges = data.edge_index
    output = model(x, edges)
    optimizer.zero_grad(set_to_none=True)

    loss = criterion(output[train_mask], targets[train_mask])
    loss.backward()
    optimizer.step()
    print(f"Epoch {epoch} the loss is {loss}")

# Test
x = data.x
targets = data.y
edges = data.edge_index
output = model(x, edges)
tensor1 = targets[test_mask]
tensor2 = output[test_mask].argmax(dim=1)
print(tensor1, tensor2)
differences = (tensor1 != tensor2).sum().item()
print((len(test_mask) - differences) / len(test_mask) * 100)
