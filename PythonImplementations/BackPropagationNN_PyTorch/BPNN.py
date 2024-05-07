import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from BPNNDataset import BPNNDataset

# class FeedforwardNeuralNetModel(nn.Module):
#     def __init__(self, input_dim, hidden_dim, output_dim):
#         super(FeedforwardNeuralNetModel, self).__init__()
#         self.fc1 = nn.Linear(input_dim, hidden_dim)
#         self.fc2 = nn.Linear(hidden_dim, hidden_dim)  # Adding an extra hidden layer
#         self.fc3 = nn.Linear(hidden_dim, output_dim)
#         self.relu = nn.ReLU()
#
#     def forward(self, x):
#         x = self.relu(self.fc1(x))
#         x = self.relu(self.fc2(x))  # Adding another layer with ReLU activation
#         x = self.fc3(x)
#         return x

class FeedforwardNeuralNetModel(nn.Module):
    def __init__(self):
        super(FeedforwardNeuralNetModel, self).__init__()
        self.fc1 = nn.Linear(2, 64)
        self.fc2 = nn.Linear(64, 3)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x



def train(dataloader, model, criterion, optimizer):
    model.train()
    for batch_idx, (data, target) in enumerate(dataloader):
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()

def validate(dataloader, model, criterion):
    model.eval()
    val_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in dataloader:
            output = model(data)
            val_loss += criterion(output, target).item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.argmax(dim=1, keepdim=True)).sum().item()

    val_loss /= len(dataloader.dataset)
    accuracy = correct / len(dataloader.dataset)
    print(f"Validation: Average loss: {val_loss:.4f}, Accuracy: {100 * accuracy:.2f}%")

def main():
    train_data = ((0.111, 0.935), (0.155, 0.958), (0.151, 0.960), (0.153, 0.955),
                  (0.715, 0.924), (0.758, 0.964), (0.725, 0.935), (0.707, 0.913),
                  (0.167, 0.079), (0.215, 0.081), (0.219, 0.075), (0.220, 0.078))

    train_labels = ((1.0, 0.0, 0.0), (1.0, 0.0, 0.0), (1.0, 0.0, 0.0), (1.0, 0.0, 0.0),
                    (0.0, 1.0, 0.0), (0.0, 1.0, 0.0), (0.0, 1.0, 0.0), (0.0, 1.0, 0.0),
                    (0.0, 0.0, 1.0), (0.0, 0.0, 1.0), (0.0, 0.0, 1.0), (0.0, 0.0, 1.0))

    tensor_x = torch.tensor(train_data, dtype=torch.float32)
    tensor_y = torch.tensor(train_labels, dtype=torch.float32)

    dataset_train = BPNNDataset(tensor_x, tensor_y)
    dataloader_train = DataLoader(dataset_train, batch_size=4, shuffle=True)

    model = FeedforwardNeuralNetModel()
    criterion = nn.MSELoss()
    optimizer = optim.SGD(model.parameters(), lr=0.01)

    num_epochs = 100
    for epoch in range(num_epochs):
        print(f"Epoch {epoch + 1}/{num_epochs}")
        train(dataloader_train, model, criterion, optimizer)
        validate(dataloader_train, model, criterion)


if __name__ == "__main__":
    main()
