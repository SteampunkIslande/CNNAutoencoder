from unet import UNet
import torch


def loadModel(path, eval_mode=False):
    """
    :param path: The path to the saved state dictionnary
    :param eval_mode: Set it to True if you want to use the model for inference
    :return: The pre-trained model from the specified state dictionnary
    """
    model = UNet()
    model.load_state_dict(
        torch.load(path, map_location=lambda storage, loc: storage)
        )  # Make sure that we always load the model into CPU
    if eval_mode:
        model.eval()
    return model

def loadCheckpoint(path, model: torch.nn.Module, optimizer: torch.optim.Optimizer):
    """
    :param path: Path to the checkpoint to load
    :param model: The model you want to revert the state to
    :param optimizer: The optimizer you want to revert the state to
    :return: A tuple with model, epoch, and the reloaded optimizer
    """
    checkpoint = torch.load(path, map_location=lambda storage, loc: storage)
    model.load_state_dict(checkpoint["model_state_dict"])  # Make sure that we always load the model into CPU
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    epoch = checkpoint["epoch"]
    return model, epoch, optimizer
