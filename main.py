import numpy as np
import torch
from torch.utils.data import DataLoader 
import os

from module import AE
from dataset import SMSDataset


os.environ['KMP_DUPLICATE_LIB_OK']='True'

model = AE().double()

# Validation using MSE Loss function
loss_function = torch.nn.MSELoss()
  
# Using an Adam Optimizer with lr = 0.1
optimizer = torch.optim.Adam(model.parameters(),
                             lr = 1e-2,
                             weight_decay = 1e-8)

data = SMSDataset()

loader = DataLoader(data,batch_size=32,shuffle=True)

epochs = 20
outputs = []
losses = []

for epoch in range(epochs):
    for batch_x in loader:
        batch_x = batch_x.double()
        reconstructed = model(batch_x)
        
        loss = loss_function(reconstructed, batch_x)

        optimizer.zero_grad() # gradients to 0
        loss.backward() # Backpropogation
        optimizer.step() 

        losses.append(loss.item())

    print('Epoch number %d Loss %f'%(epoch,np.mean(losses)))
    outputs.append((epochs,batch_x,reconstructed))

torch.save(model,'auto_encoder.pkl')
