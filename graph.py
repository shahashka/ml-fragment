import torch.nn as nn
import torch
import pandas as pd
import pickle
import matplotlib.pyplot as plt
from apex import amp
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import glob
from scipy.stats import pearsonr
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from torch.utils.tensorboard import SummaryWriter
import time
from sklearn.preprocessing import OneHotEncoder

import dgl
from dgl.nn.pytorch import GraphConv
from torch.optim.lr_scheduler import ReduceLROnPlateau
import torch.nn.functional as F
import networkx as nx
from dgl import function as fn

from scipy import stats
from sklearn.metrics import mean_absolute_error, r2_score, confusion_matrix, accuracy_score, precision_score, recall_score, roc_curve, auc
from regression_enrichment_surface import regression_enrichment_surface as rds
from torch.utils.data import DataLoader
import argparse

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-n', type=int,
                    help='number of samples')
    return parser.parse_args()
class GraphDistConv(GraphConv):
    def __init__(self,
                 in_feats,
                 out_feats,
                 norm='both',
                 weight=True,
                 bias=True,
                 activation=None):
        super(GraphDistConv, self).__init__(in_feats,out_feats,norm,weight,bias,activation)
        
    def forward(self, graph, feat, dist, weight=None):
        graph = graph.local_var()

        if self._norm == 'both':
            degs = graph.out_degrees().to(feat.device).float().clamp(min=1)
            norm = torch.pow(degs, -0.5)
            shp = norm.shape + (1,) * (feat.dim() - 1)
            norm = torch.reshape(norm, shp)
            feat = feat * norm

        if weight is not None:
            if self.weight is not None:
                raise DGLError('External weight is provided while at the same time the'
                               ' module has defined its own weight parameter. Please'
                               ' create the module with flag weight=False.')
        else:
            weight = self.weight

        if self._in_feats > self._out_feats:
            # mult W first to reduce the feature size for aggregation.
            if weight is not None:
                feat = torch.matmul(feat, weight)
            graph.srcdata['h'] = feat
            graph.edata['w'] = dist
            graph.update_all(fn.u_mul_e('h', 'w', out='m'), fn.sum('m', 'h'))
            rst = graph.dstdata['h']
        else:
            # aggregate first then mult W
            graph.srcdata['h'] = feat
            graph.edata['w'] = dist
            graph.update_all(fn.u_mul_e('h', 'w', out='m'), fn.sum('m', 'h'))
            rst = graph.dstdata['h']

            if weight is not None:
                rst = torch.matmul(rst, weight)

        if self._norm != 'none':
            degs = graph.in_degrees().to(feat.device).float().clamp(min=1)
            if self._norm == 'both':
                norm = torch.pow(degs, -0.5)
            else:
                norm = 1.0 / degs
            shp = norm.shape + (1,) * (feat.dim() - 1)
            norm = torch.reshape(norm, shp)
            rst = rst * norm

        if self.bias is not None:
            rst = rst + self.bias

        if self._activation is not None:
            rst = self._activation(rst)

        return rst
    
class GCN(nn.Module):
    def __init__(self, g, in_feats, h1, h2):
        super(GCN, self).__init__()
        self.g = g
        self.conv1 = GraphDistConv(in_feats, h1)
        self.bn1 = nn.BatchNorm1d(h1)
        self.conv2 = GraphDistConv(h1, h2)
        self.linear = nn.Linear(h2,1)

    def forward(self, inputs, dist):
        h = self.conv1(self.g, inputs, dist)
        h = torch.relu(h)
      #  h = self.bn1(h)
        h = self.conv2(self.g, h, dist)
        h = torch.relu(h)
        fp = torch.sum(h, dim=0)
        out = self.linear(fp)
        return out
    
class GraphDataset(torch.utils.data.Dataset):
    def __init__(self, graph):
        'Initialization'
        self.graph = graph

    def __len__(self):
        'Denotes the total number of samples'
        return len(self.graph)

    def __getitem__(self, index):
        'Generates one sample of data'
        return self.graph[index]
    
atom_residues = [
            "ALA",
            "ARG",
            "ASN",
            "ASP",
            "CYS",
            "GLN",
            "GLU",
            "GLY",
            "HIS",
            "ILE",
            "LEU",
            "LYS",
            "MET",
            "PHE",
            "PRO",
            "SER",
            "THR",
            "TRP",
            "TYR",
            "VAL",
           "H", 
           "I",
           "N",
           "P",
           "C",
           "O",
           "F",
           "S",
           "Li",
           "Cl",
           "Br"]
encoder = OneHotEncoder(sparse=False).fit(np.array(atom_residues).reshape(-1,1))

# graph dataset is the same as image except no interpolation and only one "channel"
#graphs,docks = pickle.load(open("dset_jak2_8a_80x80_NEAREST_graph.pkl", "rb"))

 
def cutoff(val):
    return 1 if val < 0.5 else 0 #only within 5 angstrom considered connection

def build_graph(graph,feat, dock):
    # read matrix, cutoff at 5 angstroms to create adjacency matrix
    adj_matrix = np.array([[cutoff(y) for y in x] for x in graph])

    # use networkx to create a graph from adjacency matrix
    # visualize
    vis = nx.from_numpy_matrix(adj_matrix)
    # Kamada-Kawaii layout usually looks pretty for arbitrary graphs
    #pos = nx.kamada_kawai_layout(vis)
#     protein = '#00FFFF'
#     ligand = '#FF00FF'
    #nx.draw(vis, pos, with_labels=True, node_color=[ligand if i[0] == 1 else protein for i in feat])
    
    
    # convert to DGL graph
    g = dgl.DGLGraph()
    g = dgl.from_networkx(vis)
    edges = graph*adj_matrix
    for i in range(edges.shape[0]):
        edges[i][i] = 1
    edges=edges[edges!=0]
    edges = 1/edges
    edges[edges==1]=0
    
#     # create node features
    feat = encoder.transform(np.array(feat).reshape(-1,1))
    g.ndata['feat'] = torch.FloatTensor(feat)
    #g.ndata['feat'] = torch.nn.Embedding(adj_matrix.shape[0], 32).weight
    g.ndata['label'] = torch.FloatTensor(dock*np.ones(adj_matrix.shape[0]))
    g.edata['inv_dist'] = torch.FloatTensor(edges)

    return g


# function to concatenate/batch graphs
# creates one large graph per batch that has disjoint subgraphs
def collate(data):
    graph = dgl.batch(data)
    return graph
    
def train(n):
    data=pickle.load(open("graphs_10k.pkl","rb"))[0:n]
    x_train, x_val, _, _ = train_test_split(data,data,test_size=0.2, shuffle=True)
    train_dataset = GraphDataset(x_train)
    test_dataset = GraphDataset(x_val)


    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda:0" if use_cuda else "cpu")

    batch_size=1
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, collate_fn=collate)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, collate_fn=collate)
    g = train_dataset[0]
    n_classes = 1
    num_feats = g.ndata['feat'].shape[1]
    g = g.int().to(device)
    # define the model
    model = GCN(g,
                num_feats,
                64,
                64)
    # define the optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=5e-4)
    lr_red = ReduceLROnPlateau(optimizer, patience=10, threshold=0.01)
    model = model.to(device)
    loss_fcn = nn.MSELoss()
    num_epochs=50
    loss_train_store=[]
    loss_test_store=[]

    r2_train_store=[]
    r2_test_store=[]
    for epoch in range(num_epochs):
        model.train()
        loss_acc=0
        iters=0
        y_pred_values=[]
        y_test_values=[]
        for batch, subgraph in enumerate(train_dataloader):
            subgraph = subgraph.to(device)

            model.g = subgraph
            y_pred = model(subgraph.ndata['feat'].float(), subgraph.edata['inv_dist'].float())
            loss = loss_fcn(y_pred[0], subgraph.ndata['label'].float()[0])

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            loss_acc+=loss.item()
            iters+=1
            y_pred_values.append(y_pred.cpu()[0])
            y_test_values.append(subgraph.ndata['label'].float().cpu()[0])  
        r2_train = r2_score( y_test_values, y_pred_values)  
        r2_train_store.append(r2_train)


        loss_train_store.append(loss_acc/iters)
        lr_red.step(loss_train_store[-1])
        with torch.no_grad():
            model.eval()
            loss_acc=0
            iters=0
            y_pred_values=[]
            y_test_values=[]
            for batch, subgraph in enumerate(test_dataloader):
                subgraph = subgraph.to(device)
                model.g = subgraph
                y_pred = model(subgraph.ndata['feat'].float(), subgraph.edata['inv_dist'].float())
                loss = loss_fcn(y_pred[0],subgraph.ndata['label'].float()[0])        

                loss_acc+=loss.item()
                iters+=1

                y_pred_values.append(y_pred.cpu()[0])
                y_test_values.append(subgraph.ndata['label'].float().cpu()[0])  

            r2_test = r2_score( y_test_values, y_pred_values)    
            loss_test_store.append(loss_acc/iters)
            r2_test_store.append(r2_test)

        print(epoch, loss_train_store[-1], loss_test_store[-1], r2_train_store[-1], r2_test_store[-1])
        
        
    plt.plot(loss_train_store, label='train loss')
    plt.plot(loss_test_store, label='test loss')
    plt.legend()
    plt.savefig("loss.png")
    plt.clf()
    plt.plot(r2_train_store, label='train r2')
    plt.plot(r2_test_store, label='test r2')
    plt.legend()
    plt.savefig("r2.png")
    plt.clf()
    
    plt.hist(y_test_values, label="test", alpha=0.5)
    plt.hist(y_pred_values, label="pred", alpha=0.5)
    plt.legend()
    plt.savefig("hist.png")
    plt.clf()


    rds_model = rds.RegressionEnrichmentSurface(percent_min=-3)
    rds_model.compute(np.array(y_test_values).flatten(), np.array(y_pred_values).flatten(), samples=30)
    rds_model.plot(save_file="rds_on_cell.png",
                       title='Regression Enrichment Surface (Avg over Unique Cells)')
    plt.savefig("res.png")
    plt.clf()
    
    
if __name__ == "__main__":
    args = get_args()   
    train(args.n)
