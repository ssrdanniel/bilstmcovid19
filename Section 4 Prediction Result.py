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
import datetime as predict_truth_line
plt.rcParams["font.family"] = "Times New Roman"
plt.autoscale(enable=True, axis='both', tight=None)


# Hyper-parameters
input_size = 2 # number of features
hidden_size = 64
num_layers = 4
batch_size = 16
num_epochs = 200
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


class GRU(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        super(GRU, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.gru = nn.GRU(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        # Set initial hidden and cell states
        tensor1 = torch.Tensor(self.num_layers, x.size(0), self.hidden_size)
        h0 = torch.nn.init.uniform(tensor1, a=0, b=1)  # 2 for bidirection
        # c0 = torch.zeros(self.num_layers * 2, x.size(0), self.hidden_size)
        tensor2 = torch.Tensor(self.num_layers, x.size(0), self.hidden_size)
        c0 = torch.nn.init.uniform(tensor2, a=0, b=1)

        # Forward propagate gru
        out, _ = self.gru(x)  # out: tensor of shape (batch_size, seq_length, hidden_size)

        # Decode the hidden state of the last time step
        out = self.fc(out[:, -1, :])
        return out


usdataset = UsDataset(fcsv='./daily2.csv')
scale = usdataset.scaler
train_loader = DataLoader(usdataset, shuffle=True, batch_size=batch_size)
print(len(usdataset))


# Recurrent neural network (many-to-one)
class BiRNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        super(BiRNN, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(hidden_size * 2, num_classes)  # 2 for bidirection
        # self.fc1 = nn.Linear(hidden_size*2, 64)
        # self.fc2 = nn.Linear(64, num_classes)

    def forward(self, x):
        # Set initial states
        # h0 = torch.zeros(self.num_layers * 2, x.size(0), self.hidden_size) # 2 for bidirection
        # c0 = torch.zeros(self.num_layers * 2, x.size(0), self.hidden_size)

        tensor1 = torch.Tensor(self.num_layers * 2, x.size(0), self.hidden_size)
        h0 = torch.nn.init.uniform(tensor1, a=0, b=1)  # 2 for bidirection
        tensor2 = torch.Tensor(self.num_layers * 2, x.size(0), self.hidden_size)
        c0 = torch.nn.init.uniform(tensor2, a=0, b=1)
        # Forward propagate LSTM
        out, _ = self.lstm(x, (h0, c0))  # out: tensor of shape (batch_size, seq_length, hidden_size*2)

        # Decode the hidden state of the last time step
        out = self.fc(out[:, -1, :])
        # out = self.fc1(out[:, -1, :])
        # out = self.fc2(out)
        return out


model = BiRNN(input_size, hidden_size, num_layers, num_classes)
# model = GRU(input_size, hidden_size, num_layers, num_classes)
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
    test_input = train_label[-60:]
    test_input = torch.tensor([test_input], dtype=torch.float32)
    output = model(test_input).cpu().data.numpy()
    output = output.reshape(1, pred_length, 1)
    new_output = np.tile(output, (1, 1, 2))[0]
    actual_pre = scale.inverse_transform(new_output)[:, 0]  # prediction result

    positiveIncrease = df[['positiveIncrease']].values.astype(np.float32)
    tmp = actual_pre.reshape(pred_length, 1)
    print(np.max(tmp), np.min(tmp))
    total_increase = np.vstack((positiveIncrease, tmp))
    # fig = plt.figure()
    plt.figure(figsize=(9,4))
    plt.ylabel('Daily Increase Cases', size=15, weight='bold')
    plt.xlabel('Date', size=15, weight='bold')
    delta3 = dt.timedelta(days=1)
    date3_1 = dt.datetime(2020, 1, 22)
    date3_2 = dt.datetime(2020, 8, 8)
    dates3 = mpl.dates.drange(date3_1, date3_2, delta3)
    ax = plt.gca()
    line1, = plt.plot(dates3, total_increase, color='#388004', label='Forecast', linestyle='-', marker='o', markersize='2',
                      antialiased=True, lw=0.5)
    line2, = plt.plot(dates3[:-30], positiveIncrease, color='#3065f2', label='Observed', linestyle='-', marker='s',
                      markersize='2', antialiased=True, lw=0.5)
    dateFmt = mpl.dates.DateFormatter('%Y-%m-%d')
    plt.legend(handles=[line2, line1], loc='upper left', bbox_to_anchor=(0, 0.98), frameon=False)
    plt.axvline(predict_truth_line.datetime(2020, 7, 8), ls=':', c='g', lw=1)
    plt.title('Daily Increase in COVID-19 Cases in US', size=15, weight='bold')
    ax.xaxis.set_major_formatter(dateFmt)
    # fig.autofmt_xdate(bottom=0.18)
    # fig.subplots_adjust(left=0.2)
    plt.savefig('./figure4-1.jpg', dpi=1500)
    plt.show()

    # ax = plt.gca()
    # Train and test dataset visualization
    # plt.rcParams["font.family"] = "Times New Roman"
    # plt.rcParams['font.size'] = 12
    # # plt.title('Train and Test Data Used',size=15, weight='bold')
    # plt.ylabel('Daily Increase Cases', size=15)
    # plt.xlabel('Date', size=15)
    # # plt.figure(figsize=(9,3))
    # delta3 = dt.timedelta(days=1)
    # date3_1 = dt.datetime(2020, 1, 22)
    # date3_2 = dt.datetime(2020, 8, 8)
    # dates3 = mpl.dates.drange(date3_1, date3_2, delta3)
    #
    # # plt.plot(dates3, np.array(total_increase), color='r', label='Test', linestyle='-', marker='o', markersize='2',
    # #          antialiased=True, lw=0.5)
    # plt.plot(dates3[:-29], np.array(total_increase[:-29]), color='#3065f2', label='Train', linestyle='-', marker='s',
    #          markersize='2', antialiased=True, lw=0.5)
    # plt.axvline(predict_truth_line.datetime(2020, 7, 8), ls=':', c='g', lw=1)
    # dateFmt = mpl.dates.DateFormatter('%Y-%m-%d')
    # ax.xaxis.set_major_formatter(dateFmt)
    #
    # plt.legend(loc=2)
    #
    # plt.show()

    # plt.rcParams["font.family"] = "Times New Roman"
    # plt.rcParams['font.size'] = 12
    # plt.title('Daily Increase in COVID-19 Cases in US', size=15, weight='bold')
    # plt.ylabel('Daily Increase Cases', size=15, weight='bold')
    # plt.xlabel('Time Index', size=15, weight='bold')
    # plt.plot(dates3, np.array(total_increase), color='#388004', label='Forecast', linestyle='-', marker='^', markersize='3.7',
    #          antialiased=True, lw=0.5)
    # plt.plot(dates3[:-30], np.array(total_increase[:-30]), color='#a6212c', label='Observed', linestyle='-', marker='s',
    #          markersize='3.7', antialiased=True, lw=0.5)
    # dateFmt = mpl.dates.DateFormatter('%Y-%m-%d')
    # ax.xaxis.set_major_formatter(dateFmt)
    # plt.legend(loc=1)
    # plt.axvline(predict_truth_line.datetime(2020, 7, 8), ls=':', c='#ad7b7b', lw=1)
    # plt.savefig('./newStyle_final.jpg', dpi=1500)
    # plt.show()


if __name__ == "__main__":
    save_dir = 'logs_final5_{}_{}_{}_{}'.format(num_layers, hidden_size, sequence_length, pred_length)
    # train(save_dir=save_dir)
    # prediction(model_path='./logs_gru2_{}_{}_{}_{}/133_lstm.pth'.format(num_layers, hidden_size, sequence_length, pred_length))
    prediction(model_path='./logs_bilstm2_{}_{}_{}_{}/13_lstm.pth'.format(num_layers, hidden_size, sequence_length, pred_length))


