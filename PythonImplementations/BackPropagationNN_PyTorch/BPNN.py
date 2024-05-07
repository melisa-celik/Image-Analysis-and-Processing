import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from BPNNDataset import BPNNDataset

class FeedforwardNeuralNetModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(FeedforwardNeuralNetModel, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

def train(dataloader, model, criterion, optimizer, num_epochs=100):
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for inputs, labels in dataloader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {running_loss / len(dataloader)}")

def validate(dataloader, model, criterion):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in dataloader:
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == torch.argmax(labels, dim=1)).sum().item()
    accuracy = 100 * correct / total
    print(f"Validation Accuracy: {accuracy:.2f}%")


def main():
    train_data = [
        [0.111, 0.935], [0.155, 0.958], [0.151, 0.960], [0.153, 0.955],
        [0.715, 0.924], [0.758, 0.964], [0.725, 0.935], [0.707, 0.913],
        [0.167, 0.079], [0.215, 0.081], [0.219, 0.075], [0.220, 0.078]
    ]

    train_labels = [
        [1.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0], [0.0, 1.0, 0.0], [0.0, 1.0, 0.0], [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0], [0.0, 0.0, 1.0], [0.0, 0.0, 1.0], [0.0, 0.0, 1.0]
    ]

    tensor_x = torch.stack([torch.tensor(x, dtype=torch.float32) for x in train_data])
    tensor_y = torch.stack([torch.tensor(y, dtype=torch.float32) for y in train_labels])

    dataset = BPNNDataset(tensor_x, tensor_y)
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True)

    model = FeedforwardNeuralNetModel()
    criterion = nn.MSELoss()
    optimizer = optim.SGD(model.parameters(), lr=0.01)

    train(dataloader, model, criterion, optimizer)
    validate(dataloader, model, criterion)


if __name__ == "__main__":
    main()
