import pickle
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as Data
from torchvision import transforms
import torcheval.metrics
import torch
from tqdm.notebook import tqdm

from Basel_weather_neural_network import plot_training


def unpickle(file):
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict


def load_dataset():
    image_dic1 = unpickle(directory_path + 'data_batch_1')
    image_dic2 = unpickle(directory_path + 'data_batch_2')
    image_dic3 = unpickle(directory_path + 'data_batch_3')
    image_dic4 = unpickle(directory_path + 'data_batch_4')
    image_dic5 = unpickle(directory_path + 'data_batch_5')

    return np.concatenate(
        [image_dic1[b'data'], image_dic2[b'data'], image_dic3[b'data'], image_dic4[b'data'], image_dic5[b'data']],
        axis=0), np.concatenate(
        [image_dic1[b'labels'], image_dic2[b'labels'], image_dic3[b'labels'], image_dic4[b'labels'],
         image_dic5[b'labels']], axis=0)


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

        if (epoch % 9 == 0):
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

        if (epoch % 9 == 0):
            print(
                f'Test Epoch: {epoch + 1} | Loss: {loss_sum / len(testloader)}  | Accuracy: {accuracy / len(testloader)} | Precision: {precision / len(testloader)}')
            print(
                f'Recall: {recall / len(testloader)} | F-Score: {2 * recall * precision / (recall + precision) / len(testloader)}')

        test_losses.append(loss_sum / len(testloader))
        test_accuracies.append(accuracy / len(testloader))

    return train_losses, test_losses, test_accuracies


class ResNet(nn.Module):
    def __init__(self):
        super(ResNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(64)

        self.conv2 = nn.Conv2d(64, 128, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(128)
        self.pool1 = nn.MaxPool2d(2)

        self.conv3 = nn.Conv2d(128, 128, 3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.conv4 = nn.Conv2d(128, 128, 3, padding=1)
        self.bn4 = nn.BatchNorm2d(128)

        self.conv5 = nn.Conv2d(128, 256, 3, padding=1)
        self.bn5 = nn.BatchNorm2d(256)
        self.pool2 = nn.MaxPool2d(2)

        self.conv6 = nn.Conv2d(256, 512, 3, padding=1)
        self.bn6 = nn.BatchNorm2d(512)
        self.pool3 = nn.MaxPool2d(2)

        self.conv7 = nn.Conv2d(512, 512, 3, padding=1)
        self.bn7 = nn.BatchNorm2d(512)
        self.conv8 = nn.Conv2d(512, 512, 3, padding=1)
        self.bn8 = nn.BatchNorm2d(512)
        self.pool4 = nn.MaxPool2d(2)

        self.flat = nn.Flatten()
        self.dense = nn.Linear(512, 10)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.pool1(x)

        res = x
        x = F.relu(self.bn3(self.conv3(x)))
        x = F.relu(self.bn4(self.conv4(x))) + res

        x = F.relu(self.bn5(self.conv5(x)))
        x = self.pool2(x)

        x = F.relu(self.bn6(self.conv6(x)))
        x = self.pool3(x)

        res = x
        x = F.relu(self.bn7(self.conv7(x)))
        x = F.relu(self.bn8(self.conv8(x))) + res
        x = self.pool4(x)

        x = self.flat(x)
        out = F.softmax(self.dense(x), dim=1)
        return out


directory_path = './cifar-10-batches-py/'
# размер пакета
BATCH_SIZE = 50000
images, labels = load_dataset()
label_names = unpickle('./cifar-10-batches-py/batches.meta')[b'label_names']
test_dic = unpickle('./cifar-10-batches-py/test_batch')

transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

# обучающая и тестовая выборка
trainset = MyCifar(transform_to_image(images), labels, transform)
testset = MyCifar(transform_to_image(test_dic[b'data']), test_dic[b'labels'], transform)

# загрузчики данных
trainloader = Data.DataLoader(trainset, batch_size=BATCH_SIZE, shuffle=True)
testloader = Data.DataLoader(testset, batch_size=BATCH_SIZE, shuffle=True)

device = torch.device("cpu")

model = ResNet()
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