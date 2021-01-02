import argparse
import json
from datetime import datetime

import pytz
import torch

from supported_optimizers import optimizers
from unet import createAndTrainModel
from unet_dataset import LocalFilesUnetDataset

# ------------How to save a torch model depending on the actual goal
#            https://stackoverflow.com/questions/42703500/best-way-to-save-a-trained-model-in-pytorch

timeString = lambda: datetime.now(pytz.timezone("CET")).strftime("%b-%d-%Hh%M")

crits = {
    "L1Loss": torch.nn.L1Loss(),
    "MSELoss": torch.nn.MSELoss(),
    "BCELoss": torch.nn.BCELoss()
}

parser = argparse.ArgumentParser()

# Dataset related
parser.add_argument("files_list",
                    help="Creates and train a noise reducing UNet Convolutionnal Neural Network, provided a list of reference images in any text file")
parser.add_argument("--patch_size", help="How big the sample patch",type=int, default=256)

# Training related
parser.add_argument("--batch_size", help="Number of examples per batch", type=int, default=4)
parser.add_argument("--epochs", help="Number of training epochs", type=int, default=50)
parser.add_argument("--log_file_name", help="Path to a new file to log the results of training")
parser.add_argument("--save_frequency", help="How many epochs before saving a checkpoint", type=int, default=10)
parser.add_argument("--run_name", help="Just to give a name to the run, so you can quickly identify it")
parser.add_argument("--model_name", help="The path where to save the model")

# Loss function related
parser.add_argument("--criterion", help="Loss function to minimize for training", choices=crits.keys(),
                    default="L1Loss")

# Optimizer related
parser.add_argument("--optimizer_params", help="Path to a file with optimizer's parameters")
parser.add_argument("--learning_rate", help="Optimizer's learning rate", type=float, default=1e-3)

args = parser.parse_args()

if __name__ == "__main__":
    dataset = LocalFilesUnetDataset(args.files_list,"dataset/Sony/long","dataset/Sony/short")
    print(f"Dataset loaded !")
    createAndTrainModelArgs = {
        "dataset": dataset,
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "num_epochs": args.epochs,
        "criterion": crits[args.criterion],
        "save_frequency": args.save_frequency
    }
    if args.log_file_name:
        createAndTrainModelArgs["log_file_name"] = args.log_file_name

    if args.optimizer_params:
        optInfo = json.load(open(args.optimizer_params)) # Get the optimizer's parameters (dict with optimizer name and parameters in param field)
        optimizerFactory = optimizers[optInfo["name"]](**optInfo["params"]) # optimizers is a dict of optimizer factories. These factories are initialized with params
        print(f"Created optimizer with parameters {optInfo['params']}")
        createAndTrainModelArgs["optimizer"] = optimizerFactory

    if args.run_name:
        createAndTrainModelArgs["run_name"] = args.run_name

    model = createAndTrainModel(**createAndTrainModelArgs)

    model_name = args.model_name if args.model_name else f"Model_{args.batch_size}_{args.epochs}_{timeString()}"

    model.cpu()  # Before ever saving model to file, make sure it has host memory mapping (won't depend on harware)
    torch.save(model.state_dict(),model_name)

