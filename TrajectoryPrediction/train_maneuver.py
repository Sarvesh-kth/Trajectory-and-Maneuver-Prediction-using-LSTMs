import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
import maneuver_model
device = torch.device("cuda")
datas = pd.read_csv("revtraindata.csv")

train_x = datas.iloc[:, 0:288].values
y_train = datas.iloc[:, -5:].values
train_x = train_x.reshape(-1, 16, 18)
X_train = np.transpose(train_x, (1, 0, 2))

X_train = torch.tensor(X_train, dtype=torch.float32).to(device)
y_train = torch.tensor(y_train, dtype=torch.float32).to(device)

model = maneuver_model.trainpredict(device)
learning_rate = 0.01
n_epochs = 10
batch_size = 64
loss = model.train_model(X_train, y_train, learning_rate, n_epochs, batch_size)

torch.save(model.state_dict(), 'Maneuverpred.pth')
