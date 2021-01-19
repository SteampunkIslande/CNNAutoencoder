import argparse
import json
from datetime import datetime

import pytz
import torch

from supported_optimizers import optimizers
from unet_dataset import LocalFilesUnetDataset

import torch.nn as nn

from torch.utils.data import DataLoader
from torch.utils.data import Dataset

from unet import UNet
from load_model import loadModel, loadCheckpoint

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

timeString = lambda: datetime.now(pytz.timezone("CET")).strftime("%b-%d-%Hh%M")

crits = {
    "L1Loss": torch.nn.L1Loss(),
    "MSELoss": torch.nn.MSELoss(),
    "BCELoss": torch.nn.BCELoss()
}

def trainModel(dataset, optimizer_factory, batch_size=4, num_epochs=50,
               criterion=nn.L1Loss, log_file_name="vsc.csv",
               save_frequency=-1, run_name="blade_run", pretrained_path=None,
               checkpoint_path=None, validation_set=None):
    """
    :param dataset: The dataset you want the autoencoder to be trained on.
    Dataset's __getitem__ function should return two C,H,W-shaped tensors
    :param optimizer_factory: Callable that returns an actual optimizer (i.e. a proper optimizer factory)
    :param batch_size: The size of the batches (defaults to 4)
    :param num_epochs: The number of epochs (defaults to 50)
    :param criterion: The Loss function (defaults to L1Loss)
    :param log_file_name: (very optional) Path to a file where to store the history of the training
    :param save_frequency: Save model's state every save_frequency epoch (default to 10)
    :param run_name: (optional) The name of the run, so that you can find it in the mess of directories this program will create
    :param pretrained_path: (optional) If specified, the path to a model's saved state_dict (only the model's parameters)
    :param checkpoint_path: (optional) If specified, the path to a saved checkpoint
    :param validation_set: (optional) If specified, the training will be validated using this dataset
    :return: a fully trained model
    """

    if not isinstance(dataset, Dataset):
        raise ValueError("Dataset provided is not a valid torch dataset")

    if save_frequency <= 0:
        save_frequency = num_epochs  # If invalid save frequency (0 or less), we only save once at the end

    assert log_file_name.endswith(".csv")
    assert save_frequency > 0, "Save frequency must be a positive number (or negative to set default value)"

    dataloader = DataLoader(dataset, batch_size=batch_size, pin_memory=True)
    if validation_set is not None:
        validationLoader = DataLoader(validation_set,batch_size=batch_size,pin_memory=True)
    else:
        validationLoader = None



    if pretrained_path is not None:
        model = loadModel(pretrained_path)
    else:
        model = UNet()

    optimizer = optimizer_factory(model.parameters())

    if checkpoint_path is not None:
        model,startingEpoch,optimizer = loadCheckpoint(pretrained_path,model,optimizer)
    else:
        startingEpoch = 0

    model.to(device)

    lossLog = []

    for epoch in range(startingEpoch,startingEpoch+num_epochs):
        print(f"Starting epoch {epoch + 1}...")
        trainLoss = 0
        for index, (in_images, gt_images) in enumerate(dataloader):
            in_images = in_images.to(device)
            gt_images = gt_images.to(device)
            optimizer.zero_grad()

            output = model(in_images)

            loss = criterion(output,gt_images)

            loss.backward()
            optimizer.step()

            trainLoss += loss.item()

        trainLoss = trainLoss / (len(dataloader))

        valLoss = 0
        if validationLoader is not None:
            for index, (in_images, gt_images) in enumerate(validationLoader):
                in_images = in_images.to(device)
                gt_images = gt_images.to(device)

                output = model(in_images)

                loss = criterion(output, gt_images)

                valLoss += loss.item()

            valLoss = valLoss / (len(validationLoader))

        lossLog.append((epoch, trainLoss, valLoss))

        print(f"Epoch {epoch + 1}/{num_epochs} finished, loss:{loss:.4f}")

        # We save the progress every save_frequency epoch
        if (epoch + 1) % save_frequency == 0:
            # Get French time stamp even if Colab's GPUs don't have consistent timezones
            timeString = datetime.now(pytz.timezone("CET")).strftime("%b-%d-%Hh%M")
            check_point_dict = {
                "model_state_dict": model.state_dict(),
                "epoch": epoch,
                "optimizer_state_dict": optimizer.state_dict()
            }
            torch.save(check_point_dict, f"checkpoints/{run_name}_check_point_{timeString}_EPOCH_{epoch + 1}")

    if len(log_file_name) != 0:
        runInfo = {}
        runInfo["CSV_filename"] = log_file_name
        runInfo["Optimizer"] = {optimizer.__class__.__name__
                               : [{k:v for k,v in param.items() if k!="params"} for param in optimizer.param_groups]}
        runInfo["Criterion"] = [criterion.__class__.__name__]

        f = open(log_file_name, "w")
        f.write(f"Epoch,"
                f"Training Loss,"
                f"Validation Loss\n")
        for epochX, trainLossY, valLossY in lossLog:
            f.write(f"{epochX},{trainLossY},{valLossY}\n")
        f.close()

        json.dump(runInfo,open(log_file_name.replace(".csv",".json"),"w"))

    return model  # Return the trained model



if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # Dataset related
    parser.add_argument("files_list",
                        help="Creates and train a noise reducing UNet Convolutionnal Neural Network, provided a list of reference images in any text file")
    parser.add_argument("--patch_size", help="How big the sample patch", type=int, default=256)

    # Training related
    parser.add_argument("--batch_size", help="Number of examples per batch", type=int, default=4)
    parser.add_argument("--epochs", help="Number of training epochs", type=int, default=50)
    parser.add_argument("--log_file_name", help="Path to a new file to log the results of training")
    parser.add_argument("--save_frequency", help="How many epochs before saving a checkpoint", type=int, default=10)
    parser.add_argument("--run_name", help="Just to give a name to the run, so you can quickly identify it")
    parser.add_argument("--model_name", help="The path where to save the model")

    parser.add_argument("--validation_set",help="The path to the validation set file")

    parser.add_argument("--checkpoint_path",help="If you have a checkpoint, specify its path")
    parser.add_argument("--trained_model_path",help="If you have already trained model, specify its path")

    # Loss function related
    parser.add_argument("--criterion", help="Loss function to minimize for training", choices=crits.keys(),
                        default="L1Loss")

    # Optimizer related
    parser.add_argument("--optimizer_params", help="Path to a file with optimizer's parameters")

    args = parser.parse_args()

    dataset = LocalFilesUnetDataset("dataset",args.files_list)
    print(f"Dataset loaded !")
    trainModelArgs = {
        "dataset": dataset,
        "batch_size": args.batch_size,
        "num_epochs": args.epochs,
        "save_frequency": args.save_frequency
    }

    if args.validation_set:
        trainModelArgs["validation_set"]=LocalFilesUnetDataset("dataset",args.validation_set)

    if args.log_file_name:
        trainModelArgs["log_file_name"] = args.log_file_name

    if args.optimizer_params:
        optInfo = json.load(open(
            args.optimizer_params))  # Get the optimizer's parameters (dict with optimizer name and parameters in param field)
        optimizerFactory = optimizers[optInfo["name"]](**optInfo[
            "params"])  # optimizers is a dict of optimizer factories. These factories are initialized with params
        trainModelArgs["optimizer_factory"] = optimizerFactory
    else:
        trainModelArgs["optimizer_factory"] = lambda params: torch.optim.Adam(params, lr=1e-4)

    if args.run_name:
        trainModelArgs["run_name"] = args.run_name

    model = trainModel(**trainModelArgs)

    model_name = args.model_name if args.model_name else f"Model_{args.batch_size}_{args.epochs}_{timeString()}"

    model.cpu()  # Before ever saving model to file, make sure it has host memory mapping (won't depend on harware)
    torch.save(model.state_dict(), model_name)
