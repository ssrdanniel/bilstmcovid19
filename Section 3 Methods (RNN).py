from sklearn.metrics import mean_squared_error, r2_score
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from torch.utils.data import Dataset, DataLoader
import os
import matplotlib.pyplot as plt
# from sklearn.externals import joblib
import datetime as dt
import matplotlib as mpl


# Hyper-parameters
input_size = 2 # number of features
hidden_size = 64
num_layers = 4
batch_size = 16
num_epochs = 100
learning_rate = 0.0005
decay = 0.0001
pred_length = 30
num_classes = pred_length
sequence_length = 60


def create_increase(x):
    # y = np.transpose(x, (1,0))
    new_x = np.diff(x, axis=0)
    new_x = np.insert(new_x, 0, 0, axis=0)
    return new_x


def reading_data(fcsv='./daily2.csv'):
    df = pd.read_csv(fcsv)
    all_data = df[['positiveIncrease', 'recoveredIncrease']].values.astype(np.float32)
    df2 = pd.read_csv('./time-series-19-covid-combined.csv')
    russia_data = create_increase(df2[(df2['country'] == 'Russia')][['Confirmed', 'Recovered']].values.astype(np.float32))
    brazil_data = create_increase(df2[(df2['country'] == 'Brazil')][['Confirmed', 'Recovered']].values.astype(np.float32))
    india_data = create_increase(df2[(df2['country'] == 'India')][['Confirmed', 'Recovered']].values.astype(np.float32))
    peru_data = create_increase(df2[(df2['country'] == 'Peru')][['Confirmed', 'Recovered']].values.astype(np.float32))
    mexico_data = create_increase(df2[(df2['country'] == 'Mexico')][['Confirmed', 'Recovered']].values.astype(np.float32))
    sa_data = create_increase(df2[(df2['country'] == 'South Africa')][['Confirmed', 'Recovered']].values.astype(np.float32))
    lens = len(brazil_data)
    train_data = np.vstack((brazil_data, india_data, russia_data, sa_data, peru_data, mexico_data, all_data))
    return train_data, all_data, lens


def normalization(alld):
    print(np.max(alld), np.min(alld))
    scaler = MinMaxScaler(feature_range=(-1, 1))
    # scaler = MinMaxScaler()
    scaler.fit(alld)
    alld = scaler.transform(alld)
    return alld, scaler


def create_inout_sequences(input_data, tw=sequence_length, pred_length=30):
    inout_seq = []
    L = len(input_data)
    for i in range(L-tw-pred_length):
        train_seq = input_data[i:i+tw]
        train_label = input_data[i+tw:i+tw+pred_length, 0]
        inout_seq.append((train_seq, train_label))
    return inout_seq


def create_inout_sequences_1(all_data, tw=sequence_length, lens=157, pred_length=30):
    inout_seq = []
    n = int(len(all_data)/lens)
    new_input = []
    for i in range(n-1):
        new_input.append(all_data[i*lens:(i+1)*lens])
    i += 1
    new_input.append(all_data[i*lens:-30])
    for input_data in new_input:
        L = len(input_data)
        for i in range(L-tw-pred_length):
            train_seq = input_data[i:i+tw]
            train_label = input_data[i+tw:i+tw+pred_length, 0]
            inout_seq.append((train_seq, train_label))
    return inout_seq


class UsDataset(Dataset):
    def __init__(self, fcsv='./daily.csv', phase='train'):
        train, all_data, lens= reading_data(fcsv)
        self.train, self.scaler= normalization(train)
        # self.train= std_normalization(train)
        if phase=='train':
            self.seq = create_inout_sequences_1(self.train, lens=lens)

    def __len__(self):
        return len(self.seq)

    def __getitem__(self, idx):
        seq, label = self.seq[idx]
        return seq, label


usdataset = UsDataset(fcsv='./daily2.csv')
scale = usdataset.scaler
train_loader = DataLoader(usdataset, shuffle=True, batch_size=batch_size)
print(len(usdataset))


# Recurrent neural network (many-to-one)
class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        super(RNN, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.RNN(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_classes)  # 2 for bidirection

    def forward(self, x):
        # Set initial states
        # h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size)
        # c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size)

        tensor1 = torch.Tensor(self.num_layers * 2, x.size(0), self.hidden_size)
        h0 = torch.nn.init.uniform(tensor1, a=0, b=1)  # 2 for bidirection
        tensor2 = torch.Tensor(self.num_layers * 2, x.size(0), self.hidden_size)
        c0 = torch.nn.init.uniform(tensor2, a=0, b=1)

        # Forward propagate RNN
        out, _ = self.lstm(x)  # out: tensor of shape (batch_size, seq_length, hidden_size*2)

        # Decode the hidden state of the last time step
        out = self.fc(out[:, -1, :])
        return out



model = RNN(input_size, hidden_size, num_layers, num_classes)
# Loss and optimizer
criterion = torch.nn.MSELoss()  # reduction='sum' created huge loss value
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=decay)


def train(save_dir='logs'):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # Train the model
    for epoch in range(num_epochs):
        avg_loss = []
        for i, (images, labels) in enumerate(train_loader):
            images = images.reshape(-1, sequence_length, input_size)
            images = torch.tensor(images, dtype=torch.float32)
            labels =  torch.tensor(labels, dtype=torch.float32)

            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, labels)

            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            avg_loss.append(loss.data.numpy())
        print('Epoch [{}/{}], Loss: {:.4f}'.format(epoch + 1, num_epochs, np.mean(avg_loss)))

        if (epoch + 1) % 5 == 0 or np.mean(avg_loss) < 0.12:
            state_dict = model.state_dict()
            save_ = os.path.join(save_dir, '{}_lstm.pth'.format(str(epoch)))
            torch.save({'state_dict': state_dict, 'epoch': epoch}, save_)


def prediction(model_path='./logs/999_lstm.pth'):

    checkpoint = torch.load(model_path)
    model.load_state_dict(checkpoint['state_dict'])
    model.eval()

    df = pd.read_csv('./daily2.csv')
    train_label = df[['positiveIncrease', 'recoveredIncrease']].values.astype(np.float32)
    train_label = scale.transform(train_label)
    test_input = train_label[-90:-60]
    test_input = torch.tensor([test_input], dtype=torch.float32)
    output = model(test_input).cpu().data.numpy()
    output = output.reshape(1, pred_length, 1)
    new_output = np.tile(output, (1,1,2))[0]
    actual_pre = scale.inverse_transform(new_output)[:, 0] # prediction result

    positiveIncrease = df[['positiveIncrease']].values.astype(np.float32)[-30:]
    tmp = actual_pre.reshape(pred_length, 1)
    print(np.max(tmp), np.min(tmp))

    date2_1 = dt.datetime(2020, 6, 8)
    date2_2 = dt.datetime(2020, 7, 8)
    delta2 = dt.timedelta(days=1)
    dates2 = mpl.dates.drange(date2_1, date2_2, delta2)
    print(len(dates2))
    return dates2, tmp, positiveIncrease
    fig, ax = plt.subplots()
    plt.title('Days vs Positive Cases Increased Everyday')
    plt.ylabel('Confirmed')
    plt.xlabel('Date')
    ax = plt.gca()
    plt.plot(dates2, np.array(tmp), label='Prediction')
    plt.plot(dates2, np.array(positiveIncrease), label='Truth')
    dateFmt = mpl.dates.DateFormatter('%Y-%m-%d')
    ax.xaxis.set_major_formatter(dateFmt)
    plt.xticks(rotation=90)
    fig.autofmt_xdate()

    plt.legend(loc=2)
    plt.show()
    # plt.savefig('./figure4.jpg', dpi=1500)


if __name__ == "__main__":
    save_dir = './logs_rnn2_{}_{}_{}_{}'.format(num_layers, hidden_size, sequence_length, pred_length)
    # train(save_dir=save_dir)
    print(save_dir)
    prediction(model_path='./logs_rnn2_{}_{}_{}_{}/32_lstm.pth'.format(num_layers, hidden_size, sequence_length, pred_length))

