import numpy as np
import csv
import torch.nn.utils as torch_utils
from tqdm import trange
import torch
import torch.nn as nn
from torch import optim

device = 'cuda'
class lstm_encoder(nn.Module):

    def __init__(self, input_size, hidden_size, num_layers=1):
        super(lstm_encoder, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.emb = 64
        self.fc1 = nn.Linear(self.input_size, self.emb)
        self.leaky_relu = torch.nn.LeakyReLU(0.1)
        self.lstm = nn.LSTM(self.emb,self.hidden_size,self.num_layers)
        self.fc2 = nn.Linear(self.hidden_size, self.emb)

    def forward(self, x_input):
        x_input = self.fc1(x_input)
        x_input = self.leaky_relu(x_input)
        _, (hidden_state1,cell_state1) = self.lstm(x_input)
        hidden_state = self.leaky_relu(self.fc2(hidden_state1.view(hidden_state1.shape[1], hidden_state1.shape[2])))

        return hidden_state,(hidden_state1,cell_state1)


class lstm_decoder(nn.Module):

    def __init__(self, input_size, hidden_size, num_layers):

        super(lstm_decoder, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.emb = 64
        self.op = 2
        self.lstm = nn.LSTM(self.input_size, self.hidden_size,self.num_layers)
        self.fc1 = nn.Linear(self.hidden_size, self.emb)
        self.leaky_relu = torch.nn.LeakyReLU(0.1)
        self.fc2 = nn.Linear(self.emb, self.op)

    def forward(self, x_input, encoder_hidden_states):

        hidden_state, cell_state = encoder_hidden_states
        hidden_state = hidden_state.to(x_input.device)
        cell_state = cell_state.to(x_input.device)
        lstm_out, self.hidden = self.lstm(x_input.unsqueeze(0), (hidden_state, cell_state))
        output = self.fc1(lstm_out.squeeze(0))
        output = self.fc2(self.leaky_relu(output))

        return output, self.hidden


class lstm_encdec(nn.Module):

    def __init__(self, input_size, hidden_size):
        super(lstm_encdec, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size

        self.encoder = lstm_encoder(input_size=input_size, hidden_size=hidden_size,num_layers=1)
        self.decoder = lstm_decoder(input_size=69, hidden_size=hidden_size,num_layers=1)

    def train_model(self, input_tensor, target_tensor, manc, learning_rate, n_epochs, target_len, batch_size):
        losses = np.full(n_epochs, np.nan)
        optimizer = optim.Adam(self.parameters(), lr=learning_rate)
        criterion = nn.MSELoss()
        csvfile = 'losses.csv'

        x = int(input_tensor.shape[1] / batch_size)
        total = x * batch_size

        n_trainbatches = int(total / batch_size)

        with trange(n_epochs) as tr, open(csvfile, 'w', newline='') as csvf:
            csv_writer = csv.writer(csvf)
            csv_writer.writerow(['Epoch', 'Loss'])
            for it in tr:
                batch_loss = 0.
                s = 0
                e = 0
                for b in range(n_trainbatches):
                    # select data
                    if b != n_trainbatches - 1:
                        e = e + batch_size
                    else:
                        e = total

                    input_batch = input_tensor[:, s: e, :]
                    target_batch = target_tensor[:, s: e, :]
                    man_batch = manc[s: e, :]
                    s = e

                    input_batch = input_batch.to(device)
                    target_batch = target_batch.to(device)
                    man_batch = man_batch.to(device)

                    outputs = torch.zeros(target_len, batch_size, 2)

                    encoder_hidden,decoder_hidden = self.encoder(input_batch)

                    decoder_input = encoder_hidden
                    decoder_input = decoder_input.to(device)
                    decoder_input = torch.cat((decoder_input, man_batch), dim=1)

                    for t in range(target_len):
                        decoder_output, decoder_hidden = self.decoder(decoder_input, decoder_hidden)
                        outputs[t] = decoder_output

                    outputs = outputs.to(device)
                    loss = criterion(outputs, target_batch)
                    batch_loss += loss.item()

                    optimizer.zero_grad()
                    loss.backward()
                    torch_utils.clip_grad_norm_(self.parameters(), max_norm=1.0)
                    optimizer.step()

                batch_loss /= n_trainbatches
                losses[it] = batch_loss

                tr.set_postfix(loss="{0:.3f}".format(batch_loss))
                csv_writer.writerow([it, batch_loss])

        return losses

    def predict(self, input_tensor, target_len, manc):

        input_tensor = input_tensor.unsqueeze(1)
        encoder_hidden,decoder_hidden = self.encoder(input_tensor)
        outputs = torch.zeros(target_len, 2)

        decoder_input = encoder_hidden
        decoder_input = decoder_input.to(device)
        decoder_input = torch.cat((decoder_input, manc), dim=1)

        for t in range(target_len):
            decoder_output, decoder_hidden = self.decoder(decoder_input, decoder_hidden)
            outputs[t] = decoder_output.squeeze(0)

        np_outputs = outputs.detach().numpy()

        return np_outputs
