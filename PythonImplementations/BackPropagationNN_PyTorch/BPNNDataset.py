from torch.utils.data import Dataset, DataLoader

class BPNNDataset(Dataset):
    def __init__(self, features, labels):
        self.labels = labels
        self.features = features

    def __len__(self):
        return len(self.features)

    def __getitem__(self, index):
        X = self.features[index]
        y = self.labels[index]
        return X, y

