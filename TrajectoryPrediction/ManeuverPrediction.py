import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
import maneuver_model
device = torch.device("cuda")
datas = pd.read_csv("revtestdata.csv")

train_x = datas.iloc[:, 0:288].values
y_train = datas.iloc[:, -5:].values
train_x = train_x.reshape(-1, 16, 18)
X_train = np.transpose(train_x, (1, 0, 2))

X_train = torch.tensor(X_train, dtype=torch.float32).to(device)
y_train = torch.tensor(y_train, dtype=torch.float32).to(device)

model = maneuver_model.trainpredict(device)
model.load_state_dict(torch.load('Maneuverpred.pth'))
lat,lon = model.predict(X_train[:,0,:])
print("Predicted Latitude classes ",lat)
print("Actual Latitude class ",y_train[0,:3])
print("Predicted Longitute classes ", lon)
print("Actual Longitude class ", y_train[0,3:])
