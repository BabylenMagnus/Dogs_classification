import torch
import torch.nn as nn
from torchvision.models import resnet101

from config import *
import os

import matplotlib.pyplot as plt


def train_classification(
        epochs, model, train_loader, valid_loader, loss_func, optimizer, noti_rate=20
):
    model.train()
    train_losses = []
    test_accuracies = []
    train_accuracies = []

    for epoch in range(epochs):
        total_loss = 0
        total_accuracy = 0

        for batch_idx, (data, classes) in enumerate(train_loader, start=1):

            data = data.cuda()
            data = data.to(torch.float32)
            data /= 255
            input_data = data

            classes = classes.cuda()
            output = model(input_data)
            loss = loss_func(output, classes)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            total_accuracy += sum(output.argmax(1) == classes).item() / len(classes)

            if not batch_idx % noti_rate:
                print(
                    f'Train batch: {batch_idx} train loss: {round(total_loss / (batch_idx + 1), 4)}, '
                    f'train accuracy: {round(total_accuracy / (batch_idx + 1), 4)}'
                )

        avg_loss = total_loss / len(train_loader)
        train_losses.append(avg_loss)

        avg_acc = total_accuracy / len(train_loader)
        train_accuracies.append(avg_acc)

        test_accuracy = 0

        for data, classes in valid_loader:
            with torch.no_grad():
                data = data.cuda()
                data = data.to(torch.float32)
                data /= 255
                classes = classes.cuda()

                output = model(data)

                test_accuracy += sum(output.argmax(1) == classes).item() / len(classes)

        test_accuracy /= len(valid_loader)
        test_accuracies.append(test_accuracy)

        print(
            f'Train Epoch: {epoch} train loss: {round(avg_loss, 4)}, train accuracy: {round(avg_acc, 4)} '
            f' test accuracy: {round(test_accuracy, 4)}'
        )

    return train_losses, train_accuracies, test_accuracies


def plot_result(train_losses, train_accuracies, test_accuracies):
    x = list(range(len(train_losses)))

    f, polts = plt.subplots(1, 2)
    f.set_figheight(5)
    f.set_figwidth(15)

    polts[0].plot(x, train_accuracies)
    polts[0].plot(x, test_accuracies)
    polts[0].legend(["train accuracies", "test accuracies"])

    polts[1].plot(x, train_losses)
    polts[1].legend(["train loss"])

    plt.show()


def get_model(device='cuda'):
    weights = sorted(os.listdir(WEIGHT_PATH))

    if weights:
        return torch.load(WEIGHT_PATH + weights[-1], map_location=device)

    model = resnet101(pretrained=True)
    model.fc = nn.Sequential(
        nn.Linear(2048, 128),
        nn.ReLU(inplace=True),
        nn.Linear(128, 10)
    )

    return model.to(device)
