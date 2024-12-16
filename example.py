import pickle
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as Data
import os


class Perceptron(nn.Module):
    def __init__(self):
        super(Perceptron, self).__init__()

        self.dense1 = nn.Linear(in_features=3072, out_features=2048)
        self.bench_normalize1 = nn.BatchNorm1d(2048)
        self.dense2 = nn.Linear(in_features=2048, out_features=1024)
        self.drop1 = nn.Dropout(0.5)
        self.dense3 = nn.Linear(in_features=1024, out_features=256)
        self.drop2 = nn.Dropout(0.5)
        self.dense4 = nn.Linear(in_features=256, out_features=10)

    def forward(self, x):
        x = F.leaky_relu(self.dense1(x))
        x = self.bench_normalize1(x)
        x = F.leaky_relu(self.dense2(x))
        x = self.drop1(x)
        x = F.leaky_relu(self.dense3(x))
        x = self.drop2(x)
        out = F.softmax(self.dense4(x), dim=1)
        return out


class MyCifar(Data.Dataset):
    def __init__(self, images, labels):
        self.images = images
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        image = F.normalize(torch.from_numpy(np.array(self.images[index])).float(), dim=0)
        label = self.labels[index]
        return image, label


def load_dataset():
    image_dic1 = unpickle('./cifar-10-batches-py/data_batch_1')
    image_dic2 = unpickle('./cifar-10-batches-py/data_batch_2')
    image_dic3 = unpickle('./cifar-10-batches-py/data_batch_3')
    image_dic4 = unpickle('./cifar-10-batches-py/data_batch_4')
    image_dic5 = unpickle('./cifar-10-batches-py/data_batch_5')

    return np.concatenate([image_dic1[b'data'], image_dic2[b'data'], image_dic3[b'data'],
                           image_dic4[b'data'], image_dic5[b'data']], axis=0), \
        np.concatenate([image_dic1[b'labels'], image_dic2[b'labels'], image_dic3[b'labels'],
                        image_dic4[b'labels'], image_dic5[b'labels']], axis=0)


def unpickle(file):
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict


def show_image(image_data, label, label_names):
    image = np.array(image_data).reshape((3, 32, 32)).transpose((1, 2, 0))
    plt.imshow(image)
    plt.title(label_names[label].decode('utf-8'))
    plt.show()


def save_data(file, data):
    with open(file, 'a') as f:
        f.write(data + '\n')


def fit(num_epochs, model, criterion, optimizer, device, trainloader, testloader, log_file):
    for epoch in range(num_epochs):
        loss_sum = 0
        correct = 0
        num = 0

        model.train()
        for _, (image, label) in enumerate(trainloader):
            image = image.to(device)
            label = label.to(device)

            optimizer.zero_grad()
            logits = model(image)
            loss = criterion(logits, label)
            loss.backward()
            optimizer.step()

            _, preds = torch.max(logits, axis=-1)
            correct += (preds == label).sum().item()
            num += len(image)
            loss_sum += loss.item()

        train_accuracy = correct / num * 100
        epoch_log = f"Epoch: {epoch + 1} | Loss: {loss_sum / len(trainloader):.4f} | Train Accuracy: {train_accuracy:.2f}%"
        save_data(log_file, epoch_log)

        model.eval()
        correct = 0
        num = 0
        with torch.no_grad():
            for _, (image, label) in enumerate(testloader):
                image = image.to(device)
                label = label.to(device)
                outputs = model(image)

                _, preds = torch.max(outputs, axis=-1)
                correct += (preds == label).sum().item()
                num += len(image)

            test_accuracy = correct / num * 100
            test_log = f"Test Accuracy: {test_accuracy:.2f}%"
            save_data(log_file, test_log)


if __name__ == '__main__':
    BATCH_SIZE = 50000
    label_names = unpickle('./cifar-10-batches-py/batches.meta')[b'label_names']
    test_dic = unpickle('./cifar-10-batches-py/test_batch')
    images, labels = load_dataset()

    trainset = MyCifar(images, labels)
    testset = MyCifar(test_dic[b'data'], test_dic[b'labels'])

    trainloader = Data.DataLoader(trainset, batch_size=BATCH_SIZE, shuffle=True)
    testloader = Data.DataLoader(testset, batch_size=BATCH_SIZE, shuffle=True)

    learning_rate = 0.001
    num_epochs = 100
    log_file = 'training_log2.txt'

    if os.path.exists(log_file):
        os.remove(log_file)

    device = torch.device("cpu")
    model = Perceptron().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), weight_decay=0.1, lr=learning_rate)

    fit(num_epochs, model, criterion, optimizer, device, trainloader, testloader, log_file)
