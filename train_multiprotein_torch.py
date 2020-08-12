import torch.nn as nn
import torch
import pandas as pd
import pickle
import matplotlib.pyplot as plt
from tqdm import tqdm

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import glob
from scipy.stats import pearsonr
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

# TODO
# r2 metric

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
    data_images =[]
    data_scores = []
    # look for all the protein matrix and score files
    files_m = glob.glob('./**/*.high.matrices.npy', recursive=True)
    files_s = glob.glob('./**/*.scores', recursive=True)
    for m,s in zip(files_m, files_s):
        print(m,s)
        data_images.append(np.load(m)[0:5000])
        data_scores.append(pd.read_csv(s).iloc[0:5000])
    x = np.vstack(data_images)[:,:,:,0:2]
    y = pd.concat(data_scores)['Chemgauss4'].values
    scaler = MinMaxScaler()
    y = scaler.fit_transform(y.reshape(-1,1))
    x = x
    x = torch.FloatTensor(x).view(x.shape[0],2,64,64)
    print(x.shape)
    y = torch.FloatTensor(y)
    return x, y

# main
x,y = load_data()

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
criterion = nn.MSELoss(reduction='sum')

# epoch loop
num_epochs=50
loss_train_store=[]
loss_test_store=[]
r2_train_store=[]
r2_test_store=[]
model.to(device)
for e in range(num_epochs):
    model.train()
    # Training
    loss_acc=0
    iters=0
    y_pred_values=[]
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
        loss_train.backward()
        opt.step()
        
        loss_acc+=loss_train.item()
        iters+=1
        y_pred_values.append(y_pred)
        
    y_pred_values = [item for sublist in y_pred_values for item in sublist]    
    r2_epoch = r2_score(y_pred_values, y_train)
    loss_train_store.append(loss_acc/iters)
    r2_train_store.append(r2_epoch)
    
    gen_test = tqdm(enumerate(test_dataloader), total=int(len(test_dataloader.dataset) / test_dataloader.batch_size),
                   desc='testing')
    # Testing
    loss_acc=0
    iters=0
    y_pred_values=[]
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
            
        y_pred_values = [item for sublist in y_pred_values for item in sublist]    
        r2_epoch = r2_score(y_pred_values, y_val)  
        loss_test_store.append(loss_acc/iters)
        r2_test_store.append(r2_epoch)
        
        
    print(e, loss_train_store[-1], loss_test_store[-1], r2_train_store[-1], r2_test_store[-1])
    
fig ,(ax1,ax2) = plt.subplots(2)
ax1.plot(loss_train_store,label="train loss")
ax1.plot(loss_test_store, label="test loss")
ax1.set_xlabel("Epochs")
ax1.set_ylabel("MSE")

ax2.plot(r2_train_store,label="train R2")
ax2.plot(r2_test_store, label="test R2")
ax2.set_xlabel("Epochs")
ax2.set_ylabel("R2")


plt.savefig("metrics.torch.png")

    