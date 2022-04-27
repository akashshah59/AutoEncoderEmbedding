import torch

class AE(torch.nn.Module):
    def __init__(self):
        super().__init__()
        
        self.encoder = torch.nn.Sequential(
            torch.nn.Linear(8, 4),
            torch.nn.ReLU(),
            torch.nn.Linear(4, 2)
        )
        
        self.decoder = torch.nn.Sequential(
            torch.nn.Linear(2, 4),
            torch.nn.ReLU(),
            torch.nn.Linear(4, 8),
            #torch.nn.Sigmoid() -> Probably for classification. This is Ordinal so we may not need this.
        )
    def forward(self,x):
        encoded = self.encoder(x) # batch_size * 2
        decoded = self.decoder(encoded) # batch_size * 2
        return decoded
        