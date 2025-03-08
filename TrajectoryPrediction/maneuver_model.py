import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
from tqdm import trange

batch_size = 64  
device = 'cuda'
nf = 18
ip_l = 16
enc_s = 128
dec_s = 128
op_l = 24
ip_emb_s = 64
class maneuver(nn.Module):
    def __init__(self):
        super(maneuver,self).__init__()
        self.device = device
        self.nf = nf
        self.ip_l = ip_l
        self.op_l = op_l
        self.enc_s = enc_s
        self.dec_s = dec_s
        self.ip_emb_s = ip_emb_s
        self.numlat = 3 # lm, rm, sm
        self.numlon = 2 # br, nm
        self.ip_emb = torch.nn.Linear(self.nf, self.ip_emb_s)
        self.man_lstm = torch.nn.LSTM(self.ip_emb_s,self.enc_s,1)
        self.op_lat = torch.nn.Linear(self.enc_s, self.numlat)
        self.op_lon = torch.nn.Linear(self.enc_s, self.numlon)

        self.leaky_relu = torch.nn.LeakyReLU(0.1)
        self.relu = torch.nn.ReLU()
        self.softmax = torch.nn.Softmax(dim=1)

    def forward(self, x):
        h0 = torch.zeros(1, x.size(1), self.enc_s).to(self.device)
        c0 = torch.zeros(1, x.size(1), self.enc_s).to(self.device)
        x = self.ip_emb(x)
        x = self.leaky_relu(x)
        manout, (h, c) = self.man_lstm(x, (h0, c0))
        manout = manout[-1, :, :]
        lat_man = self.op_lat(manout)
        lon_man = self.op_lon(manout)
        lat_pr = self.softmax(lat_man)
        lon_pr = self.softmax(lon_man)
        return h, lat_pr, lon_pr


class trainpredict(nn.Module):
    def __init__(self,device):
        super(trainpredict, self).__init__()
        self.device = device
        self.maneuverpred = maneuver().to(device)


    def train_model(self, input_tensor, manc, learning_rate, n_epochs, batch_size):
        losses = np.full(n_epochs, np.nan)
        optimizer = optim.Adam(self.parameters(), lr=learning_rate)
        criterion = nn.CrossEntropyLoss()
        x = int(input_tensor.shape[1] / batch_size)
        total = x * batch_size

        n_trainbatches = int(total / batch_size)
        with trange(n_epochs) as tr:
            for it in tr:
                s = 0
                e = 0
                batch_loss = 0
                for b in range(n_trainbatches):
                    if b != n_trainbatches - 1:
                        e = e + batch_size
                    else:
                        e = total
                    input_batch = input_tensor[:, s: e, :]
                    man_batch = manc[s: e, :]
                    s = e
                    input_batch = input_batch.to(device)
                    man_batch = man_batch.to(device)
                    _, lat, lon = self.maneuverpred(input_batch)

                    loss = criterion(lat,man_batch[:,0:3]) + criterion(lon,man_batch[:,3:5])
                    losses[it] = loss.item()
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    batch_loss += loss.item()
                batch_loss /= n_trainbatches
                print(f'Epoch {it + 1}/{n_epochs}, Loss: {batch_loss}')
                tr.set_postfix(loss="{0:.3f}".format(batch_loss))
        return losses
    def predict(self,input_tensor):

        input_tensor = input_tensor.unsqueeze(1)
        _, lat,lon = self.maneuverpred(input_tensor)
        max_index_lat = torch.argmax(lat, dim=1)
        latclass = torch.zeros_like(lat)
        latclass[0, max_index_lat] = 1

        max_index_lon = torch.argmax(lon, dim=1)
        lonclass = torch.zeros_like(lon)
        lonclass[0, max_index_lon] = 1
        return latclass,lonclass