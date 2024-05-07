import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

class BackPropagationNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(BackPropagationNN, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_size)
        self.sigmoid = nn.Sigmoid()
        self.lambda_val = 1.0
        self.eta = 0.1

    def forward(self, x):
        x = self.fc1(x)
        x = self.sigmoid(x)
        x = self.fc2(x)
        x = self.sigmoid(x)
        return x

def train_nn(model, n_samples=1000, n_iterations=10000):
    optimizer = optim.SGD(model.parameters(), lr=model.eta)
    criterion = nn.MSELoss()

    training_set = []
    for i in range(n_samples):
        classA = i % 2
        in_data = np.random.rand(model.fc1.in_features) * 0.1 + (0.6 if classA else 0.2)
        target = np.array([1.0, 0.0]) if classA else np.array([0.0, 1.0])
        training_set.append((in_data, target))

    error = 1.0
    iteration = 0

    while error > 0.001 and iteration < n_iterations:
        in_data, target = training_set[iteration % n_samples]
        input_tensor = torch.tensor(in_data, dtype=torch.float32)
        target_tensor = torch.tensor(target, dtype=torch.float32)

        optimizer.zero_grad()
        output = model(input_tensor)
        loss = criterion(output, target_tensor)
        loss.backward()
        optimizer.step()

        error = loss.item()
        iteration += 1
        print(f"\rError: {error:.3f}", end="")

    print(f" ({iteration} iterations)")

def test_nn(model, num_samples):
    num_err = 0
    for _ in range(num_samples):
        classA = bool(np.random.randint(2))
        in_data = np.random.rand(model.fc1.in_features) * 0.1 + (0.6 if classA else 0.2)
        input_tensor = torch.tensor(in_data, dtype=torch.float32)

        output = model(input_tensor)
        predicted_class = torch.argmax(output).item()

        if predicted_class != (0 if classA else 1):
            num_err += 1

        print(f"Predicted: {predicted_class} (Ground Truth: {0 if classA else 1})")

    test_error = num_err / num_samples
    print(f"Test Error: {test_error:.2f}")


def main():
    input_size = 2
    hidden_size = 4
    output_size = 2

    model = BackPropagationNN(input_size, hidden_size, output_size)
    train_nn(model)
    test_nn(model, num_samples=10)

if __name__ == "__main__":
    main()

