import torch.nn as nn
import torch
import pandas as pd
import pickle
import matplotlib.pyplot as plt
from tqdm import tqdm
import os
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import glob
from scipy.stats import pearsonr
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import argparse
import torch.multiprocessing as mp

import torch.distributed as dist
from apex.parallel import DistributedDataParallel as DDP
from apex import amp
class ConvNet(nn.Module):
    def __init__(self, num_classes=10):
        super(ConvNet, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        self.layer2 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        self.fc = nn.Linear(7*7*32, num_classes)

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = out.reshape(out.size(0), -1)
        out = self.fc(out)
        return out
    
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
        out = self.cnn(x)
        out = out.reshape(out.size(0), -1) # where x.size(0) is the batch size
        out = self.linear(out)
        return out

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

# Distributed training
# https://yangkky.github.io/2019/07/08/distributed-pytorch-tutorial.html
# TODO allreduce for r^2, (for loss this is handled internally by pytorch)
def train(gpu, epoch, train_dataloader, model, dist, amp, opt, criterion):
    device = next(model.parameters()).device
    gen_train = tqdm(enumerate(train_dataloader), total=int(len(train_dataloader.dataset) / 
                                                            (dist.get_world_size()*train_dataloader.batch_size)),
                       desc='training')
    loss_acc=0
    iters=0
    model.train()
    for g in gen_train:
        i, (local_batch, local_labels) = g
        # Transfer to GPU
        local_batch, local_labels = local_batch.to(device), local_labels.to(device)

        #forward pass
        y_pred = model(local_batch)
        loss_train = criterion(y_pred.flatten(),local_labels.flatten())

        #backprop + update
        opt.zero_grad()
        with amp.scale_loss(loss_train, opt) as scaled_loss:
            scaled_loss.backward()
        opt.step()

        # log for rank=0
        if gpu ==0:   
            loss_acc+=loss_train.item()
        iters+=1

    return loss_acc/iters
        
def test(gpu, epoch, test_dataloader, model, dist, criterion):
    device = next(model.parameters()).device
    gen_test = tqdm(enumerate(test_dataloader), total=int(len(test_dataloader.dataset) / 
                                                           (dist.get_world_size()*test_dataloader.batch_size)),
                   desc='testing')
    # Testing
    iters=0
    loss_acc=0
    model.eval()
    y_pred_arr = []
    y_test_arr= []
    with torch.no_grad():
        for g in gen_test:
            i, (local_batch, local_labels) = g
            # Transfer to GPU
            local_batch, local_labels = local_batch.to(device), local_labels.to(device)
         
            y_pred = model(local_batch)
            loss_test = criterion(y_pred.flatten(),local_labels.flatten())
            
            y_pred_arr.append(y_pred)
            y_test_arr.append(local_labels)

            if gpu==0:
                loss_acc+=loss_test.item()
            iters+=1

    y_pred_arr = [item for sublist in y_pred_arr for item in sublist]    
    y_test_arr = [item for sublist in y_test_arr for item in sublist]    
    y_pred_arr = torch.FloatTensor(y_pred_arr).to(device)
    y_test_arr = torch.FloatTensor(y_test_arr).to(device)

    r2 = calc_r2(gpu, dist, y_test_arr, y_pred_arr)
    
    return loss_acc/iters, r2    

#https://github.com/pytorch/pytorch/issues/14536
def calc_r2(gpu, dist, y_true, y_pred):
    # TODO all_gather->gather for rank=0
    gather_true = [torch.ones_like(y_true) for _ in range(dist.get_world_size())]
    gather_pred = [torch.ones_like(y_pred)  for _ in range(dist.get_world_size())]
    dist.all_gather(gather_true, y_true)
    dist.all_gather(gather_pred, y_pred)
    
    y_pred_arr = [item for sublist in gather_pred for item in sublist]    
    y_test_arr = [item for sublist in gather_true for item in sublist]    
    return r2_score(y_test_arr, y_pred_arr)
    
def plot(train_loss, test_loss, test_r2):
    fig ,(ax1,ax2) = plt.subplots(2, figsize=(5,7))
    ax1.plot(train_loss,label="train loss")
    ax1.plot(test_loss, label="test loss")
    ax1.set_xlabel("Epochs")
    ax1.set_ylabel("MSE")
    ax1.legend()

    ax2.plot(test_r2,label="test R2")
    ax2.set_xlabel("Epochs")
    ax2.set_ylabel("R2")
    ax2.legend()

    plt.savefig("metrics.png")
    
def run(gpu, args):
    
    # set up processes
    rank = args.nr * args.gpus + gpu
    print(rank)
    dist.init_process_group(                                   
    	backend='nccl',                                         
   		init_method='env://',                                   
    	world_size=args.world_size,                              
    	rank=rank                                               
    )                                                          
    print("init processes")

    # load data
    x,y,scaler  = load_data()

    x_train, x_val, y_train, y_val = train_test_split(x,y,test_size=0.2, shuffle=True)
    train_dataset = ContactDataset(x_train,y_train)
    test_dataset = ContactDataset(x_val,y_val)
    print("loaded data")

    # create model
    torch.manual_seed(0)
    model = ContactModel()
    torch.cuda.set_device(gpu)
    model.cuda(gpu)
    
    # create loss function and optimizer
    criterion = nn.MSELoss().cuda(gpu)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-5)
    model, opt = amp.initialize(model, opt, opt_level='O2')
    print("apex init")
    
    model = DDP(model) #torch.nn.parallel.DistributedDataParallel(model, device_ids=[gpu])
    print("Model created")

    # distributed data loading
    train_sampler = torch.utils.data.distributed.DistributedSampler(
    	train_dataset,
    	num_replicas=args.world_size,
    	rank=rank
    )
    test_sampler = torch.utils.data.distributed.DistributedSampler(
    	test_dataset,
    	num_replicas=args.world_size,
    	rank=rank
    )
    
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=64, num_workers=0, shuffle=False,
                                                   sampler=train_sampler)
    test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=64, num_workers=0, shuffle=False,
                                                  sampler=test_sampler)
    
    # train loop
    plot_train = []
    plot_test = []
    plot_r2=[]
    for epoch in range(args.epochs):
        train_loss = train(gpu, epoch, train_dataloader, model, dist, amp, opt, criterion)
        test_loss, test_r2 = test(gpu, epoch, test_dataloader, model, dist, criterion)
        plot_train.append(train_loss)
        plot_test.append(test_loss)
        plot_r2.append(test_r2)
        if gpu==0:
            print("Epoch {0} train loss {1}, test loss {2}, test r2 {3}".format(epoch, train_loss, test_loss,test_r2))

    if gpu == 0:
        plot(plot_train, plot_test, plot_r2)
        state = model.state_dict()
        torch.save({'model_state': state,
                    'opt_state': opt.state_dict(),
                    'args': args,
                    'amp': amp.state_dict()}, "model.pt")
    
    
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-n', '--nodes', default=1,
                        type=int, metavar='N')
    parser.add_argument('-g', '--gpus', default=1, type=int,
                        help='number of gpus per node')
    parser.add_argument('-nr', '--nr', default=0, type=int,
                        help='ranking within the nodes')
    parser.add_argument('--epochs', default=2, type=int, 
                        metavar='N',
                        help='number of total epochs to run')
    args = parser.parse_args()
    #########################################################
    args.world_size = args.gpus * args.nodes                #
    os.environ['MASTER_ADDR'] = 'localhost'                 #
    os.environ['MASTER_PORT'] = '8899'                      #
    mp.spawn(run, nprocs=args.gpus, args=(args,))         #
    #########################################################
    
if __name__ == '__main__':
    main()
