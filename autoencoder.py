
import os
from random import shuffle

import torch.nn as nn
import torch
import torchvision
from PIL import Image
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

import imghdr

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def listImagesInDir(directory, ignoreDirs=[]):
    res = []
    ignoreDirs = [d.split("/")[-1] for d in ignoreDirs]
    for root, dirs, files in os.walk(directory, topdown=True):
        dirs[:] = [d for d in set(dirs) - set(ignoreDirs)]
        for fileName in files:
            if imghdr.what(os.path.join(root, fileName)) in ['rgb', 'gif', 'pbm', 'pgm', 'ppm', 'tiff', 'xbm', 'jpeg', 'bmp', 'png', 'webp']:
                res.append(os.path.join(root, fileName))
    return res

class AutoEncoderLocalFileDataset(Dataset):
    def __init__(self, **kwargs):
        if "dirPath" in kwargs:
            if "ignoreDirs" in kwargs:
                self.imagePaths = listImagesInDir(kwargs["dirPath"], ignoreDirs=kwargs["ignoreDirs"])
            else:
                self.imagePaths = listImagesInDir(kwargs["dirPath"])

        if "csvFilesList" in kwargs and os.path.isfile(kwargs["csvFilesList"]):
            self.imagePaths = [line.strip() for line in open(kwargs["csvFilesList"], "r")]  # Just load every line from the list of files

        if "fileNames" in kwargs:
            self.imagePaths = kwargs["fileNames"]

        self.indices = list(range(len(self.imagePaths)))
        shuffle(self.indices)

    def __len__(self):
        return len(self.imagePaths)

    def __getitem__(self, index):
        path = self.imagePaths[self.indices[index]]  # We shuffled the indices ourselves, so we need to use our indices
        img = Image.open(path)
        if img.size[0] / img.size[1] < 1:
            img = img.transpose(Image.ROTATE_90)
        img = img.resize((292, 196))
        return torchvision.transforms.ToTensor()(img)


class AutoEncoder(nn.Module):

    def __init__(self, **kwargs):
        super().__init__()
        self.conv1 = nn.Sequential(nn.Conv2d(3,32,(3,3)),
                                   nn.LeakyReLU(0.2),
                                   nn.Conv2d(32,32,(3,3)),
                                   nn.LeakyReLU(0.2))
        self.pool1 = nn.MaxPool2d((2,2))

        self.conv2 = nn.Sequential(nn.Conv2d(32,64,(3,3)),
                                   nn.LeakyReLU(0.2),
                                   nn.Conv2d(64,64,(3,3)),
                                   nn.LeakyReLU(0.2))

        self.pool2 = nn.MaxPool2d((2,2))

        self.conv3 = nn.Sequential(nn.Conv2d(64, 128, (3, 3)),
                                   nn.LeakyReLU(0.2),
                                   nn.Conv2d(128, 128, (3, 3)),
                                   nn.LeakyReLU(0.2))

        self.pool3 = nn.MaxPool2d((2,2))

        self.conv4 = nn.Sequential(nn.Conv2d(128, 256, (3, 3)),
                                   nn.LeakyReLU(0.2),
                                   nn.Conv2d(256, 256, (3, 3)),
                                   nn.LeakyReLU(0.2))

        self.pool4 = nn.MaxPool2d((2,2))

        self.conv5 = nn.Sequential(nn.Conv2d(256, 512, (3, 3)),
                                   nn.LeakyReLU(0.2),
                                   nn.Conv2d(512, 512, (3, 3)),
                                   nn.LeakyReLU(0.2))

        self.conv6 = nn.Sequential(nn.Conv2d(512,256,(3,3),),
                                   nn.LeakyReLU(0.2),
                                   nn.Conv2d(256,256,(3,3)),
                                   nn.LeakyReLU(0.2))

        self.conv7 = nn.Sequential(nn.Conv2d(256,128,(3,3)),
                                   nn.LeakyReLU(0.2),
                                   nn.Conv2d(128,128,(3,3)),
                                   nn.LeakyReLU(0.2))

        self.conv8 = nn.Sequential(nn.Conv2d(128,64,(3,3)),
                                   nn.LeakyReLU(0.2),
                                   nn.Conv2d(64,64,(3,3)),
                                   nn.LeakyReLU(0.2))

        self.conv9 = nn.Sequential(nn.Conv2d(64,32,(3,3)),
                                   nn.LeakyReLU(0.2),
                                   nn.Conv2d(32,32,(3,3)),
                                   nn.LeakyReLU(0.2))

        self.conv10 = nn.Conv2d(32,12,(1,1))


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
        up6 = self.upsampleAndConcat(conv5,conv4,256,512)


    def upsampleAndConcat(self,x1,x2,output_channels,input_channels):
        poolsize=2
        deconv = nn.ConvTranspose2d(input_channels,output_channels,(poolsize,poolsize))


def createAndTrainModel(**kwargs):
    """
    :param dataset: The dataset you want the autoencoder to be trained on.
    Dataset's __getitem__ function should return a C,H,W-shaped tensor
    :param batch_size: The size of the batches (defaults to 4)
    :param learning_rate: The learning rate (defaults to 1e-3)
    :param num_epochs: The number of epochs (defaults to 50)
    :param criterion: The Loss function (defaults to MSELoss)
    :param logFileName: (very optional) Path to a file where to store the history of the training
    :return: a fully trained model AND the latest loss computed
    """

    if "dataset" in kwargs:
        aeDataset = kwargs["dataset"]
    else:
        raise ValueError("No dataset provided (use dataset keyword argument)")

    if not isinstance(aeDataset,Dataset):
        raise ValueError("Dataset provided is not a valid torch dataset")

    if "batch_size" in kwargs:
        batch_size=kwargs["batch_size"]
    else:
        batch_size=4

    if "learning_rate" in kwargs:
        learning_rate=kwargs["learning_rate"]
    else:
        learning_rate=1e-3

    if "num_epochs" in kwargs:
        num_epochs=kwargs["num_epochs"]
    else:
        num_epochs=50

    aeDataloader = DataLoader(aeDataset, batch_size=batch_size,pin_memory=True)

    model=AutoEncoder()
    model.to(device)

    if "criterion" in kwargs:
        criterion=kwargs["criterion"]
    else:
        criterion=nn.MSELoss()

    optimizer=torch.optim.Adam(model.parameters(),lr=learning_rate)

    lossLog=[]

    for epoch in range(num_epochs):
        loss=0
        for batch in aeDataloader:
            batch=batch.to(device)
            optimizer.zero_grad()

            output = model(batch)

            train_loss=criterion(output,batch)# The goal is to denoise, so making this look like a noisy image is maybe not the best. To be continued...

            train_loss.backward()
            optimizer.step()

            loss+=train_loss.item()

        loss=loss/(len(aeDataloader))

        lossLog.append((epoch,loss))

        print(f"epoch [{epoch+1}/{num_epochs}], loss:{loss:.4f}")
        # if epoch%10==0 and "directory" in kwargs:
        #     pic=torchvision.transforms.ToPILImage()(output[0].cpu())
        #     pic.save(f"{kwargs['directory']}/SomeImprobableName__Epoch{epoch+1}.jpg")

    if "logFileName" in kwargs:
        f=open(kwargs["logFileName"],"w")
        f.write(f"Epoch,"
                f"Loss,"
                f"Criterion,"
                f"LearningRate\n")
        for epochX,lossY in lossLog:
            f.write(f"{epochX}"
                    f",{lossY},"
                    f"{criterion.__class__.__name__},"
                    f"{learning_rate}"
                    f"\n")
        f.close()

    return model,loss # Return the trained model, and the latest training loss it yielded

