import torch
import pandas as pd
import numpy as np
import trajencdec
import maneuver_model

device = 'cuda'

train_x = pd.read_csv("revtestdata.csv")
manc = train_x.iloc[:,-5:].values
train_x = train_x.iloc[:,:288].values


x_tr = pd.read_csv("revtest_x.csv")
y_tr = pd.read_csv("revtest_y.csv")

testrow = 10
# Labels (maneuver classes)

train_x = train_x.reshape(-1, 16, 18)
train_x = np.transpose(train_x, (1, 0, 2))

y_fut_tr = y_tr.iloc[:,:15].values  # Future trajectory
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

modelmaneuver = maneuver_model.trainpredict(device)
modelmaneuver.load_state_dict(torch.load('Maneuverpred.pth'))
lat,lon = modelmaneuver.predict(train_x[:,testrow,:])
maneuver = torch.cat([lat,lon],dim = 1)

modeltraj = trajencdec.lstm_encdec(input_size = train_x.shape[2], hidden_size = 128).to(device)
modeltraj.load_state_dict(torch.load('rev44traj.pth'))
modeltraj.eval()
predictions = modeltraj.predict(train_x[:,testrow,:], ow, maneuver)
print("Predicted")
print(predictions)
print("Actual")
print(fut_tr[:,testrow,:])
