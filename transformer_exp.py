from torch.utils.data import Dataset
import json

class MSPSPDataset(Dataset):
    def __init__(self, file_path):
        with open(file_path, 'r') as f:
            self.data = json.load(f)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


