
# dataset @ https://finance.yahoo.com/quote/MSFT/history/
import time
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from copy import deepcopy as dc
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import MinMaxScaler


df = pd.read_csv('MSFT.csv')
df = df[['Date', 'Close']]
df['Date'] = pd.to_datetime(df['Date'])
plt.plot(df['Date'], df['Close'])

def prepare_dataframe_for_lstm(df, n_steps):
    df = dc(df)

    df.set_index('Date', inplace=True)

    for i in range(1, n_steps+1):
        df[f'Close(t-{i})'] = df['Close'].shift(i)

    df.dropna(inplace=True)

    return df

lookback = 7
shifted_df = prepare_dataframe_for_lstm(df, lookback)
shifted_df_as_np = shifted_df.to_numpy()
scaler = MinMaxScaler(feature_range=(-1, 1))
shifted_df_as_np = scaler.fit_transform(shifted_df_as_np)

X = shifted_df_as_np[:, 1:]
y = shifted_df_as_np[:, 0]

X = dc(np.flip(X, axis=1))

split_index = int(len(X) * 0.95)

X_train = X[:split_index]
X_test = X[split_index:]

y_train = y[:split_index]
y_test = y[split_index:]

X_train.shape, X_test.shape, y_train.shape, y_test.shape

X_train = X_train.reshape((-1, lookback, 1))
X_test = X_test.reshape((-1, lookback, 1))

y_train = y_train.reshape((-1, 1))
y_test = y_test.reshape((-1, 1))

X_train.shape, X_test.shape, y_train.shape, y_test.shape

X_train = torch.tensor(X_train).float()
y_train = torch.tensor(y_train).float()
X_test = torch.tensor(X_test).float()
y_test = torch.tensor(y_test).float()

X_train.shape, X_test.shape, y_train.shape, y_test.shape


class TimeSeriesDataset(Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, i):
        return self.X[i], self.y[i]

train_dataset = TimeSeriesDataset(X_train, y_train)
test_dataset = TimeSeriesDataset(X_test, y_test)

batch_size = 32

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# Define the PyTorch model
class StockPredModel(nn.Module):
    def __init__(self, name, input_size, hidden_size, num_layers):
        super(StockPredModel, self).__init__()
        self.name = name if name else None
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers, batch_first=True)
        self.fc1 = nn.Linear(hidden_size, 32)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(32, 32)
        self.fc3 = nn.Linear(32, 1)

    def forward(self, x):
        batch_size = x.size(0)  
        h0 = torch.zeros(self.num_layers, batch_size, self.hidden_size)
        c0 = torch.zeros(self.num_layers, batch_size, self.hidden_size)
        lstm_out, _ = self.lstm(x, (h0, c0))
        x = self.fc1(lstm_out[:, -1, :])  # Take the last LSTM output
        x = self.relu(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.fc3(x)
        return x

def train(data_loader):
    torch.manual_seed(42)

    start = time.time()

    model = StockPredModel('StockPrediction', 1, 64, 1)

    # Define loss function (MSE loss) and optimizer
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    num_epochs = 10
    for epoch in range(num_epochs):
        for data in data_loader:
            inputs, _labels = data
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, _labels)
            loss.backward()
            optimizer.step()
        print(f"Epoch [{epoch + 1}/{num_epochs}] Loss: {loss.item()}")

        print('Saving Model')
        torch.save({'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'loss': loss}, f'checkpoints/{model.name}_epoch_{epoch + 1}.pt')
        print('Model Saved')

    print('Finished Training')
    end = time.time()
    print('training time ', end-start)


def test(data_loader, *args, **kwargs):
    torch.manual_seed(42)

    model = StockPredModel('StockPrediction', 1 , 64, 1)

    # Define loss function (MSE loss) and optimizer
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    checkpoint = torch.load(f'checkpoints/{model.name}_epoch_10.pt')


    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    model.eval()

    total_loss = 0.0
    num_samples = 0

    with torch.no_grad():
        for data in data_loader:
            inputs, targets = data
            outputs = model(inputs)
            test_predictions = outputs.numpy().flatten()
            loss = criterion(outputs, targets)
            total_loss += loss.item() * inputs.size(0)
            num_samples += inputs.size(0)

            dummies = np.zeros((inputs.shape[0], lookback+1))
            dummies[:, 0] = test_predictions
            dummies = scaler.inverse_transform(dummies)

            test_predictions = dc(dummies[:, 0])

            dummies = np.zeros((inputs.shape[0], lookback+1))
            dummies[:, 0] = targets.flatten()
            dummies = scaler.inverse_transform(dummies)

            new_y_test = dc(dummies[:, 0])

            plt.plot(new_y_test, label='Actual Close')
            plt.plot(test_predictions, label='Predicted Close')
            plt.xlabel('Day')
            plt.ylabel('Close')
            plt.ylim(0, 1000)
            plt.legend()
            plt.show()

    # Calculate the mean squared error
    mse = total_loss / num_samples

    print(f"Mean Squared Error (MSE): {mse}")


train(train_loader)
test(test_loader)