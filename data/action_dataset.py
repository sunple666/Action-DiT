from torch.utils.data import Dataset

class ActionDataset(Dataset):
    def __init__(self):
        super().__init__()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]