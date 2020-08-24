import torch.nn as nn
import torch
import pandas as pd
import pickle
import matplotlib.pyplot as plt
from tqdm import tqdm
from apex import amp
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import glob
from scipy.stats import pearsonr
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

# create nn module for ConvNet model
class ContactModel(nn.Module):
    def __init__(self,dr=0.1):
        super(ContactModel, self).__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(in_channels=2, out_channels=8, kernel_size=6),
            nn.ReLU(),
            nn.Conv2d(in_channels=8, out_channels=32, kernel_size=3),
            nn.ReLU(),
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Dropout(dr))
        
        self.linear = nn.Sequential(
            nn.Linear(32*27*27,64),
            nn.ReLU(),
            nn.Dropout(dr),
            nn.Linear(64,1))
    def forward(self,x):
        x = self.cnn(x)
        x = x.view(x.size(0), -1) # where x.size(0) is the batch size
        x = self.linear(x)
        return x

class ContactDataset(torch.utils.data.Dataset):
    def __init__(self, images, dock_score):
        'Initialization'
        self.images = images
        self.dock_score = dock_score

    def __len__(self):
        'Denotes the total number of samples'
        return len(self.dock_score)

    def __getitem__(self, index):
        'Generates one sample of data'
        X = self.images[index]
        y = self.dock_score[index]
        return X, y
   

# load data
def load_data():
    with open('dset.pkl', 'rb') as pickle_file:
        x,y = pickle.load(pickle_file)
    scaler = MinMaxScaler()
    y = scaler.fit_transform(y.reshape(-1,1))
    x = torch.FloatTensor(x)
    x = np.transpose(x,(0,3,1,2))
    y = torch.FloatTensor(y)
    return x, y, scaler

# main
x,y,scaler  = load_data()

x_train, x_val, y_train, y_val = train_test_split(x,y,test_size=0.2, shuffle=True)
train_dataset = ContactDataset(x_train,y_train)
test_dataset = ContactDataset(x_val,y_val)

train_dataloader = torch.utils.data.DataLoader(train_dataset, shuffle=True, batch_size=64, num_workers=0)
test_dataloader = torch.utils.data.DataLoader(test_dataset, shuffle=True, batch_size=64, num_workers=0)

use_cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if use_cuda else "cpu")
print(use_cuda)
model = ContactModel()

# create loss function and optimizer
opt = torch.optim.AdamW(model.parameters(), lr=1e-5)
criterion = nn.MSELoss()
model.to(device)

model, opt = amp.initialize(model, opt, opt_level='O2')
# epoch loop
num_epochs=50
loss_train_store=[]
loss_test_store=[]
r2_train_store=[]
r2_test_store=[]
for e in range(num_epochs):
    model.train()
    # Training
    loss_acc=0
    iters=0
    y_pred_values=[]
    y_test_values=[]
    gen_train = tqdm(enumerate(train_dataloader), total=int(len(train_dataloader.dataset) / train_dataloader.batch_size),
                       desc='training')
    for g in gen_train:
        i, (local_batch, local_labels) = g
        # Transfer to GPU
        local_batch, local_labels = local_batch.to(device), local_labels.to(device)

        #train
        y_pred = model(local_batch)
        loss_train = criterion(y_pred.flatten(),local_labels.flatten())
        
        #backprop + update
        opt.zero_grad()
        with amp.scale_loss(loss_train, opt) as scaled_loss:
            scaled_loss.backward()
        opt.step()
        
        loss_acc+=loss_train.item()
        iters+=1
        y_pred_values.append(y_pred)
        y_test_values.append(local_labels.cpu())
        
    y_pred_values = [item for sublist in y_pred_values for item in sublist]   
    y_test_values = [item for sublist in y_test_values for item in sublist]    

    r2_epoch = r2_score(y_test_values, y_pred_values)
    loss_train_store.append(loss_acc/iters)
    r2_train_store.append(r2_epoch)
    
    gen_test = tqdm(enumerate(test_dataloader), total=int(len(test_dataloader.dataset) / test_dataloader.batch_size),
                   desc='testing')
    # Testing
    loss_acc=0
    iters=0
    y_pred_values=[]
    y_test_values=[]
    with torch.no_grad():
        for g in gen_test:
            i, (local_batch, local_labels) = g
            # Transfer to GPU
            local_batch, local_labels = local_batch.to(device), local_labels.to(device)
            #train
            y_pred = model(local_batch)
            loss_test = criterion(y_pred.flatten(),local_labels.flatten())
            loss_acc+=loss_test.item()
            iters+=1
            y_pred_values.append(y_pred)
            y_test_values.append(local_labels.cpu())
            
    y_pred_values = [item for sublist in y_pred_values for item in sublist]    
    y_test_values = [item for sublist in y_test_values for item in sublist]    

    r2_epoch = r2_score(y_test_values, y_pred_values)  
    loss_test_store.append(loss_acc/iters)
    r2_test_store.append(r2_epoch)
        
        
    print(e, loss_train_store[-1], loss_test_store[-1], r2_train_store[-1], r2_test_store[-1])
    
fig ,(ax1,ax2,ax3) = plt.subplots(3, figsize=(5,7))
ax1.plot(loss_train_store,label="train loss")
ax1.plot(loss_test_store, label="test loss")
ax1.set_xlabel("Epochs")
ax1.set_ylabel("MSE")
ax1.legend()

ax2.plot(r2_train_store,label="train R2")
ax2.plot(r2_test_store, label="test R2")
ax2.set_xlabel("Epochs")
ax2.set_ylabel("R2")
ax2.legend()

test = scaler.inverse_transform(np.array(y_test_values).reshape(-1,1))
pred = scaler.inverse_transform(np.array(y_pred_values).reshape(-1,1))

ax3.hist(test, label="test", alpha=0.5)
ax3.hist(pred, label="pred", alpha=0.5)
ax3.legend()

plt.savefig("metrics.torch.png")

    