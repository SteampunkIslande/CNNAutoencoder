import imghdr
import os
from random import shuffle

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from PIL import Image
from torch.utils.data import DataLoader
from torch.utils.data import Dataset

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

from datetime import datetime
import pytz  # So that this program's timezone is always french !



class UNet(nn.Module):

    def __init__(self, **kwargs):
        super().__init__()
        if "kernel_size" in kwargs:
            kernel_size = kwargs["kernel_size"]
        else:
            kernel_size = 3
        self.conv1 = nn.Sequential(nn.Conv2d(4, 32, (kernel_size, kernel_size), stride=1, padding=1),
                                   nn.LeakyReLU(0.2),
                                   nn.Conv2d(32, 32, (kernel_size, kernel_size), stride=1, padding=1),
                                   nn.LeakyReLU(0.2))
        self.pool1 = nn.MaxPool2d((2, 2))

        self.conv2 = nn.Sequential(nn.Conv2d(32, 64, (kernel_size, kernel_size), stride=1, padding=1),
                                   nn.LeakyReLU(0.2),
                                   nn.Conv2d(64, 64, (kernel_size, kernel_size), stride=1, padding=1),
                                   nn.LeakyReLU(0.2))

        self.pool2 = nn.MaxPool2d((2, 2))

        self.conv3 = nn.Sequential(nn.Conv2d(64, 128, (kernel_size, kernel_size), stride=1, padding=1),
                                   nn.LeakyReLU(0.2),
                                   nn.Conv2d(128, 128, (kernel_size, kernel_size), stride=1, padding=1),
                                   nn.LeakyReLU(0.2))

        self.pool3 = nn.MaxPool2d((2, 2))

        self.conv4 = nn.Sequential(nn.Conv2d(128, 256, (kernel_size, kernel_size), stride=1, padding=1),
                                   nn.LeakyReLU(0.2),
                                   nn.Conv2d(256, 256, (kernel_size, kernel_size), stride=1, padding=1),
                                   nn.LeakyReLU(0.2))

        self.pool4 = nn.MaxPool2d((2, 2))

        self.conv5 = nn.Sequential(nn.Conv2d(256, 512, (kernel_size, kernel_size), stride=1, padding=1),
                                   nn.LeakyReLU(0.2),
                                   nn.Conv2d(512, 512, (kernel_size, kernel_size), stride=1, padding=1),
                                   nn.LeakyReLU(0.2))

        self.up6 = nn.ConvTranspose2d(512, 256, 2, stride=2)

        self.conv6 = nn.Sequential(nn.Conv2d(512, 256, (kernel_size, kernel_size), stride=1, padding=1),
                                   nn.LeakyReLU(0.2),
                                   nn.Conv2d(256, 256, (kernel_size, kernel_size), stride=1, padding=1),
                                   nn.LeakyReLU(0.2))

        self.up7 = nn.ConvTranspose2d(256, 128, 2, stride=2)

        self.conv7 = nn.Sequential(nn.Conv2d(256, 128, (kernel_size, kernel_size), stride=1, padding=1),
                                   nn.LeakyReLU(0.2),
                                   nn.Conv2d(128, 128, (kernel_size, kernel_size), stride=1, padding=1),
                                   nn.LeakyReLU(0.2))

        self.up8 = nn.ConvTranspose2d(128, 64, 2, stride=2)

        self.conv8 = nn.Sequential(nn.Conv2d(128, 64, (kernel_size, kernel_size), stride=1, padding=1),
                                   nn.LeakyReLU(0.2),
                                   nn.Conv2d(64, 64, (kernel_size, kernel_size), stride=1, padding=1),
                                   nn.LeakyReLU(0.2))

        self.up9 = nn.ConvTranspose2d(64, 32, 2, stride=2)

        self.conv9 = nn.Sequential(nn.Conv2d(64, 32, (kernel_size, kernel_size), stride=1, padding=1),
                                   nn.LeakyReLU(0.2),
                                   nn.Conv2d(32, 32, (kernel_size, kernel_size), stride=1, padding=1),
                                   nn.LeakyReLU(0.2))

        self.conv10 = nn.Conv2d(32, 12, (1, 1), stride=1)

    def forward(self, x):
        conv1 = self.conv1(x)
        pool1 = self.pool1(conv1)
        conv2 = self.conv2(pool1)
        pool2 = self.pool2(conv2)
        conv3 = self.conv3(pool2)
        pool3 = self.pool3(conv3)
        conv4 = self.conv4(pool3)
        pool4 = self.pool4(conv4)
        conv5 = self.conv5(pool4)
        up6 = self.up6(conv5)
        up6 = torch.cat([up6, conv4], 1)
        conv6 = self.conv6(up6)
        up7 = self.up7(conv6)
        up7 = torch.cat([up7, conv3], 1)
        conv7 = self.conv7(up7)
        up8 = self.up8(conv7)
        up8 = torch.cat([up8, conv2], 1)
        conv8 = self.conv8(up8)
        up9 = self.up9(conv8)
        up9 = torch.cat([up9, conv1], 1)
        conv9 = self.conv9(up9)
        conv10 = self.conv10(conv9)
        return F.pixel_shuffle(conv10, 2)


def createAndTrainModel(**kwargs):
    """
    :param dataset: The dataset you want the autoencoder to be trained on.
    Dataset's __getitem__ function should return a C,H,W-shaped tensor
    :param batch_size: The size of the batches (defaults to 4)
    :param learning_rate: The learning rate (defaults to 1e-3)
    :param num_epochs: The number of epochs (defaults to 50)
    :param criterion: The Loss function (defaults to L1Loss)
    :param log_file_name: (very optional) Path to a file where to store the history of the training
    :param optimizer: (optional) Callable that returns an actual optimizer (i.e. a proper optimizer factory)
    :param save_frequency: Save model's state every save_frequency epoch
    :param run_name: (optional) The name of the run, so that you can find it in the mess of directories this program will create
    :return: a fully trained model AND the latest loss computed
    """

    assert "dataset" in kwargs, "No dataset provided (use dataset keyword argument)"
    aeDataset = kwargs["dataset"]

    if not isinstance(aeDataset, Dataset):
        raise ValueError("Dataset provided is not a valid torch dataset")

    if "batch_size" in kwargs:
        batch_size = kwargs["batch_size"]
    else:
        batch_size = 4

    if "run_name" in kwargs:
        run_name = kwargs["run_name"]
    else:
        run_name = datetime.now(pytz.timezone("CET")).strftime("Run-%b-%d-%Hh%M")

    if "learning_rate" in kwargs:
        learning_rate = kwargs["learning_rate"]
    else:
        learning_rate = 1e-4

    if "num_epochs" in kwargs:
        num_epochs = kwargs["num_epochs"]
    else:
        num_epochs = 50

    if "save_frequency" in kwargs:
        save_frequency = kwargs["save_frequency"]
    else:
        save_frequency = 10

    if save_frequency <= 0:
        save_frequency = num_epochs  # If invalid save frequency (0 or less), we only save once at the end

    aeDataloader = DataLoader(aeDataset, batch_size=batch_size, pin_memory=True)

    model = UNet()
    model.to(device)

    if "criterion" in kwargs:
        criterion = kwargs["criterion"]
    else:
        criterion = nn.L1Loss()

    if "optimizer" in kwargs:
        optimizer = kwargs["optimizer"](
            model.parameters())  # In case an optimizer factory is specified, let's give it this model's parameters
    else:
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    lossLog = []

    for in_images, gt_images in aeDataloader:
        for epoch in range(num_epochs):
            loss = 0
            in_images = in_images.to(device)
            gt_images = gt_images.to(device)
            optimizer.zero_grad()

            output = model(in_images)

            train_loss = criterion(output,
                                   gt_images)  # The goal is to denoise, so making this look like a noisy image is maybe not the best. To be continued...

            train_loss.backward()
            optimizer.step()

            loss += train_loss.item()

        loss = loss / (len(aeDataloader))

        lossLog.append((epoch, loss))

        print(f"epoch [{epoch + 1}/{num_epochs}], loss:{loss:.4f}")

        # We save the progress every save_frequency epoch, but don't want to save first as it is useless
        if (epoch + 1) % save_frequency == 0 and epoch > 0:
            # Get French time stamp even if Colab's GPUs are far away
            timeString = datetime.now(pytz.timezone("CET")).strftime("%b-%d-%Hh%M")
            check_point_dict = {
                "model_state_dict": model.state_dict(),
                "epoch": epoch,
                "optimizer_state_dict": optimizer.state_dict()
            }
            torch.save(check_point_dict, f"results/{run_name}_check_point_{timeString}_EPOCH_{epoch + 1}")

    if "log_file_name" in kwargs:
        f = open(kwargs["log_file_name"], "w")
        f.write(f"Epoch,"
                f"Loss,"
                f"Criterion,"
                f"LearningRate\n")
        for epochX, lossY in lossLog:
            f.write(f"{epochX}"
                    f",{lossY},"
                    f"{criterion.__class__.__name__},"
                    f"{learning_rate}"
                    f"\n")
        f.close()

    return model  # Return the trained model, and the latest training loss it yielded
