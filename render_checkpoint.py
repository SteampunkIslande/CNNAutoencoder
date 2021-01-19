import argparse

import torch
import torchvision

from test_model import renderImage
from unet import UNet

toPIL = torchvision.transforms.ToPILImage()

def renderCheckpoint(checkpoint_path,input_image,exposure_correction,tile_size):
    model = UNet()
    checkpoint = torch.load(checkpoint_path,map_location=lambda storage, loc: storage)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    npImg = renderImage(model,input_image,exposure_correction,tile_size)
    return npImg


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint_file_name", help="Path to the checkpoint to test")
    parser.add_argument("image",help="Path to the image to render using checkpoint's model")
    parser.add_argument("exposure_correction",help="Exposure correction ratio to apply while rendering this image",type=int)
    parser.add_argument("-o","--output",help="The path to save the render to",default="result.jpg")
    parser.add_argument("--tile_size",help="How big the tile for rendering",type=int,default=256)
    args = parser.parse_args()

    image = toPIL(renderCheckpoint(args.checkpoint_file_name,args.image,args.exposure_correction,args.tile_size))
    image.save(args.output)


