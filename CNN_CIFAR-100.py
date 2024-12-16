import numpy as np
import torch.nn as nn
import torch.utils.data as Data
import torchvision
from torchvision import transforms
import torcheval.metrics
import torch
from tqdm.notebook import tqdm

from Basel_weather_neural_network import plot_training


class MyCifar(Data.Dataset):
    def __init__(self, images, labels, transform=None):
        self.images = images
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        image = self.images[index]

        if (self.transform):
            image = self.transform(image)

        label = self.labels[index]
        return image, label


def transform_to_image(vector_list):
    return [np.array(x).reshape((3, 32, 32)).transpose((1, 2, 0)) for x in vector_list]


def fit(num_epochs, model, criterion, optimizer, device, trainloader, testloader, sched):
    # Массив суммарных ошибок сети при обучении на каждой эпохе
    train_losses = []

    # Массив суммарных ошибок сети при тестировании после каждой эпохи
    test_losses = []

    # Массив доли правильных ответов сети при тестировании после каждой эпохи
    test_accuracies = []
    for epoch in tqdm(range(num_epochs)):

        loss_sum = 0  # суммарная ошибка сети

        model = model.train()  # включаем режим обучения модели
        for _, (image, label) in enumerate(trainloader):
            image = image.to(device)
            label = label.type(torch.LongTensor)
            label = label.to(device)

            # после каждой эпохи градиенты необходимо обнулять
            optimizer.zero_grad()

            logits = model(image)  # итоговые вероятности классов
            loss = criterion(logits, label)  # подсчёт функции ошибки
            loss.backward()  # пересчёт градиентов

            optimizer.step()  # градиентный шаг
            sched.step()  # меняем темп обучения

            loss_sum += loss.item()
            del label
            del image

        if epoch % 5 == 0:
            print(f'Train Epoch: {epoch + 1} | Loss: {loss_sum / len(trainloader)}')

        train_losses.append(loss_sum / len(trainloader))

        loss_sum = 0
        precision = 0
        accuracy = 0
        recall = 0

        model.eval()  # включаем режим тестирования модели
        with torch.no_grad():  # отключение градиентов
            for _, (image, label) in enumerate(testloader):
                image = image.to(device)
                label = label.type(torch.LongTensor)
                label = label.to(device)

                outputs = model(image)
                loss_sum += criterion(outputs, label).item()

                # извлечение из полученных вероятностей наиболее вероятного класса
                _, preds = torch.max(outputs, axis=-1)

                accuracy += torcheval.metrics.functional.multiclass_accuracy(preds, label, num_classes=10,
                                                                             average='macro')
                precision += torcheval.metrics.functional.multiclass_precision(preds, label, num_classes=10,
                                                                               average='macro')
                recall += torcheval.metrics.functional.multiclass_recall(preds, label, num_classes=10, average='macro')
                del label
                del image

        if epoch % 5 == 0:
            print(
                f'Test Epoch: {epoch + 1} | Loss: {loss_sum / len(testloader)}  | Accuracy: {accuracy / len(testloader)} | Precision: {precision / len(testloader)}')
            print(
                f'Recall: {recall / len(testloader)} | F-Score: {2 * recall * precision / (recall + precision) / len(testloader)}')

        test_losses.append(loss_sum / len(testloader))
        test_accuracies.append(accuracy / len(testloader))

    return train_losses, test_losses, test_accuracies


class AlexNet(nn.Module):
    def __init__(self):
        super(AlexNet, self).__init__()

        self.conv1 = nn.Conv2d(3, 96, kernel_size=11, stride=4, padding=2)
        self.pool1 = nn.MaxPool2d(kernel_size=3, stride=2)

        self.conv2 = nn.Conv2d(96, 256, kernel_size=5, padding=2)
        self.pool2 = nn.MaxPool2d(kernel_size=3, stride=2)

        self.conv3 = nn.Conv2d(256, 384, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(384, 384, kernel_size=3, padding=1)
        self.conv5 = nn.Conv2d(384, 256, kernel_size=3, padding=1)
        self.pool3 = nn.MaxPool2d(kernel_size=3, stride=2)

        self.flat = nn.Flatten()
        self.fc1 = nn.Linear(256 * 6 * 6, 4096)
        self.fc2 = nn.Linear(4096, 4096)
        self.fc3 = nn.Linear(4096, 10)

        self.drop = nn.Dropout(p=0.5)

    def forward(self, x):
        x = self.pool1(nn.ReLU()(self.conv1(x)))
        x = self.pool2(nn.ReLU()(self.conv2(x)))
        x = nn.ReLU()(self.conv3(x))
        x = nn.ReLU()(self.conv4(x))
        x = self.pool3(nn.ReLU()(self.conv5(x)))

        x = self.flat(x)
        x = self.drop(nn.ReLU()(self.fc1(x)))
        x = self.drop(nn.ReLU()(self.fc2(x)))
        x = self.fc3(x)

        return x


# размер пакета
BATCH_SIZE = 5000

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

trainset = torchvision.datasets.CIFAR100(root='./data', train=True,
                                         download=True, transform=transform)

testset = torchvision.datasets.CIFAR100(root='./data', train=False,
                                        download=True, transform=transform)

trainloader = torch.utils.data.DataLoader(trainset, batch_size=4,
                                          shuffle=True, num_workers=2)
testloader = torch.utils.data.DataLoader(testset, batch_size=4,
                                         shuffle=False, num_workers=2)

device = torch.device("cpu")
model = AlexNet()
model = model.to(device)

num_epochs = 30
learning_rate = 0.01

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.AdamW(model.parameters(), weight_decay=0.0001, lr=learning_rate)
sched = torch.optim.lr_scheduler.OneCycleLR(optimizer, learning_rate, epochs=num_epochs,
                                            steps_per_epoch=len(trainloader))

info = fit(num_epochs, model, criterion, optimizer, device, trainloader, testloader, sched)
train_losses, test_losses, _ = info
plot_training(train_losses, test_losses)
