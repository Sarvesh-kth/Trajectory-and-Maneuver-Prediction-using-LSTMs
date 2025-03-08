import torch
import pandas as pd
import numpy as np
import trajencdec

device = 'cuda'

train_x = pd.read_csv("revtraindata.csv")
manc = train_x.iloc[:,-5:].values
train_x = train_x.iloc[:,:288].values

# datas = datas.head(1000)

x_tr = pd.read_csv("revtrain_x.csv")
y_tr = pd.read_csv("revtrain_y.csv")

# Labels (maneuver classes)

train_x = train_x.reshape(-1, 16, 18)
train_x = np.transpose(train_x, (1, 0, 2))

y_fut_tr = y_tr.iloc[:,:15].values
x_fut_tr = x_tr.iloc[:,:15].values
fut_tr = np.stack((x_fut_tr, y_fut_tr), axis=2)


fut_tr = fut_tr.reshape(-1, 15, 2)
fut_tr = np.transpose(fut_tr, (1, 0, 2))
totalnum = fut_tr.shape[0]

iw = 16
ow = 15
batch_size = 64
emb_s = 64
train_x = torch.tensor(train_x, dtype=torch.float32).to(device)
fut_tr = torch.tensor(fut_tr, dtype=torch.float32).to(device)
manc = torch.tensor(manc, dtype=torch.float32).to(device)

print(fut_tr.shape)
model = trajencdec.lstm_encdec(input_size = train_x.shape[2], hidden_size = 128).to(device)
learning_rate = 0.00005
loss = model.train_model(train_x, fut_tr, manc, learning_rate, n_epochs = 400, target_len = ow, batch_size = batch_size)
torch.save(model.state_dict(), 'rev44traj.pth')