import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
from tqdm import tqdm


class WeatherDataset(Dataset):
    def __init__(self, data):
        self.x = data[:, :-1]
        self.y = data[:, -1]

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return torch.tensor(self.x[idx], dtype=torch.float32), torch.tensor(self.y[idx], dtype=torch.float32)


# Улучшенная модель
class ImprovedPerceptron(nn.Module):
    def __init__(self, input_size):
        super(ImprovedPerceptron, self).__init__()
        self.fc1 = nn.Linear(input_size, 128)
        self.bn1 = nn.BatchNorm1d(128)
        self.fc2 = nn.Linear(128, 64)
        self.dropout = nn.Dropout(0.5)
        self.fc3 = nn.Linear(64, 32)
        self.fc4 = nn.Linear(32, 1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.bn1(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = F.relu(self.fc3(x))
        return self.fc4(x)


# Функция для тренировки модели
def fit(num_epochs, model, criterion, optimizer, device, trainloader, testloader):
    train_losses = []
    test_losses = []

    for epoch in range(num_epochs):
        train_loss = 0  # Суммарная ошибка на обучении
        model.train()
        for x_batch, y_batch in tqdm(trainloader, desc=f"Epoch {epoch + 1}/{num_epochs} - Training"):
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)

            optimizer.zero_grad()
            predictions = model(x_batch).squeeze()
            loss = criterion(predictions, y_batch)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        train_losses.append(train_loss / len(trainloader))

        # Оценка на тестовой выборке
        test_loss = 0
        model.eval()
        with torch.no_grad():
            for x_batch, y_batch in tqdm(testloader, desc=f"Epoch {epoch + 1}/{num_epochs} - Testing"):
                x_batch = x_batch.to(device)
                y_batch = y_batch.to(device)

                predictions = model(x_batch).squeeze()
                loss = criterion(predictions, y_batch)
                test_loss += loss.item()

        test_losses.append(test_loss / len(testloader))

        print(f"Epoch {epoch + 1}/{num_epochs} | Train Loss: {train_losses[-1]:.4f} | Test Loss: {test_losses[-1]:.4f}")

    return train_losses, test_losses


# Функция для построения графиков
def plot_training(train_losses, test_losses):
    plt.figure(figsize=(12, 6))
    plt.plot(train_losses, label="Train Loss", color="blue")
    plt.plot(test_losses, label="Test Loss", color="orange")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title("Training and Testing Loss")
    plt.legend()
    plt.grid()
    plt.show()


# Загрузка и обработка данных
def load_data(file):
    df = pd.read_csv(file, header=None, names=["datetime", "weather"])
    df["weather"] = df["weather"].interpolate(method='linear')
    df["year"] = df["datetime"].str[:4].astype(int)
    df["month"] = df["datetime"].str[4:6].astype(int)
    df["day"] = df["datetime"].str[6:8].astype(int)
    df["hour"] = df["datetime"].str[9:11].astype(int)
    df = df.drop(columns=["datetime"])
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(df)
    return scaled_data


# Основной скрипт
if __name__ == "__main__":
    filepath = "weather.csv"
    batch_size = 32
    epochs = 10
    learning_rate = 0.0001

    dataset = load_data(filepath)
    end_train_index = round(dataset.shape[0] * 0.8)
    train_set = dataset[:end_train_index]
    test_set = dataset[end_train_index:]

    print(f'Количество данных во всём наборе данных: {dataset.shape}')
    print(f'Количество данных в обучающей выборке: {train_set.shape}')
    print(f'Количество данных в тестовой выборке: {test_set.shape}')

    train_dataset = WeatherDataset(train_set)
    test_dataset = WeatherDataset(test_set)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    input_size = train_set.shape[1] - 1
    model = ImprovedPerceptron(input_size)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    criterion = nn.MSELoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.01)

    train_losses, test_losses = fit(epochs, model, criterion, optimizer, device, train_loader, test_loader)
    plot_training(train_losses, test_losses)
