from torch.utils.data import Dataset
import torch
import pickle

class SMSDataset(Dataset):
    def __init__(self) -> None:
        super().__init__()
        with open('train.pkl','rb') as f:
            self.x = torch.tensor(pickle.load(f))

        with open('test.pkl','rb') as f:
            self.y = torch.tensor(pickle.load(f))
    
    def __getitem__(self, index):
        # No y since we're only finding a representation. 
        return self.x[index]
    
    def __len__(self):
        return self.x.shape[0]

     
