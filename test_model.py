import torch
from unet import UNet
from torchvision.transforms import ToPILImage, ToTensor

import numpy as np
from unet_dataset import pack_raw

import argparse

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class ModelRenderer():
    def __init__(self, trained_model,exposure_correction):
        self.trained_model = trained_model
        self.trained_model.to(device)
        self.exposure_correction = exposure_correction

    def __call__(self, input_raw):
        inTensor = ToTensor()(input_raw * self.exposure_correction).to(device).unsqueeze(0)
        outTensor = self.trained_model(inTensor)[0]
        return outTensor.permute((1,2,0)).detach().cpu().numpy()

def renderTiles(input_array,tile_size,render_function):
    result = np.array([[]])
    for y in range(0, input_array.shape[0], tile_size):
        growing_line = np.array([])
        for x in range(0,input_array.shape[1],tile_size):
            renderedTile = render_function(input_array[y:y+tile_size,x:x+tile_size,:])
            growing_line = np.hstack([growing_line, renderedTile]) if growing_line.size else renderedTile

        result = np.vstack([result,growing_line]) if result.size else growing_line

    return ((result+1)*127.5).astype(np.uint8)


def loadModel(path):
    ae = UNet()
    ae.load_state_dict(torch.load(path))
    ae.eval()
    return ae

parser = argparse.ArgumentParser()

parser.add_argument("model_file_name", help="Path to the trained model to test")

parser.add_argument("image", help="Path to the image to denoise")

parser.add_argument("exposure_correction", help="Exposure correction ratio for rendering", type=float)

parser.add_argument("-o","--output",help="Where to save the result")
parser.add_argument("--tile_size",default=256, type=int)

args = parser.parse_args()

if __name__ == "__main__":
    model = loadModel(args.model_file_name)
    exposure_correction = args.exposure_correction
    renderer = ModelRenderer(model,exposure_correction)
    input_array = pack_raw(args.image)
    output_array = renderTiles(input_array,args.tile_size,renderer)
    image = ToPILImage()(output_array)
    output_fn = args.output if args.output else "result.jpg"
    image.save(output_fn)


