import os
from random import shuffle

import numpy as np
import rawpy
import torchvision
from torch.utils.data import Dataset


def pack_raw(file_name, bps=14):
    raw = rawpy.imread(file_name)
    # pack Bayer image to 4 channels
    im = raw.raw_image_visible.astype(np.float32)
    im = np.maximum(im - 512, 0) / ((2 ** bps - 1) - 512)  # subtract the black level

    im = np.expand_dims(im, axis=2)
    img_shape = im.shape
    H = img_shape[0]
    W = img_shape[1]

    out = np.concatenate((im[0:H:2, 0:W:2, :],
                          im[0:H:2, 1:W:2, :],
                          im[1:H:2, 1:W:2, :],
                          im[1:H:2, 0:W:2, :]), axis=2)
    return out


class LocalFilesUnetDataset(Dataset):
    def __init__(self,images_list_file_name, gt_path, in_path, count=100, bps=14, patch_size=256):
        f = open(images_list_file_name)
        self.in_images = []
        self.gt_images = []
        self.patch_size = patch_size
        i = 0
        for line in f:
            if i >= count: # Only take the count first images of the list
                break
            in_name, gt_name, _, _ = line.split(" ")

            in_name = os.path.basename(in_name)
            gt_name = os.path.basename(gt_name)
            ratio = float(gt_name[9:-5]) / float(in_name[9:-5])

            in_name = os.path.join(in_path, in_name)
            gt_name = os.path.join(gt_path, gt_name)

            self.in_images.append(pack_raw(in_name, bps) * ratio)

            gt_raw = rawpy.imread(gt_name)
            im = gt_raw.postprocess(use_camera_wb=True, half_size=False, no_auto_bright=True, output_bps=16)
            self.gt_images.append(np.float32(im / 65535.0))

            i += 1

        f.close()

        assert len(self.gt_images) == len(self.in_images), f"There must be as many ground truth images as inputs"

        self.npArrToTensor = torchvision.transforms.ToTensor()

        self.indices = list(range(len(self.gt_images)))
        shuffle(self.indices)

        self.in_patches=[] # Define in_patches now to avoid referenced before assignement error
        self.gt_patches=[]

        self.generatePatches()

    def __len__(self):
        return len(self.indices)

    def generatePatches(self):
        # self.in_patches and self.gt_patches are numpy array's slices that refer to self.in_images and self.gt_images respectively
        self.in_patches=[] # Clear the patches without risking to damage in_images the potential previous ones might refer to
        self.gt_patches=[]
        ps = self.patch_size

        for index, (in_image, gt_image) in enumerate(zip(self.in_images,self.gt_images)):
            print(f"Generating patch {index+1} / {len(self.in_images)}...",end="")
            H = in_image.shape[0]
            W = in_image.shape[1]

            xx = np.random.randint(0, W - ps)
            yy = np.random.randint(0, H - ps)
            input_patch = in_image[yy:yy + ps, xx:xx + ps, :]
            gt_patch = gt_image[yy * 2:yy * 2 + ps * 2, xx * 2:xx * 2 + ps * 2, :]

            if np.random.randint(2, size=1)[0] == 1:  # random flip
                input_patch = np.flip(input_patch, axis=1)
                gt_patch = np.flip(gt_patch, axis=1)
            if np.random.randint(2, size=1)[0] == 1:
                input_patch = np.flip(input_patch, axis=0)
                gt_patch = np.flip(gt_patch, axis=0)
            if np.random.randint(2, size=1)[0] == 1:  # random transpose
                input_patch = np.transpose(input_patch, (1, 0, 2))
                gt_patch = np.transpose(gt_patch, (1, 0, 2))
            self.in_patches.append(input_patch)
            self.gt_patches.append(gt_patch)
            print("done.")

    def __getitem__(self, index):
        if self.in_patches[self.indices[index]].data.contiguous:
            in_tensor = self.npArrToTensor(self.in_patches[self.indices[index]]) # No need to copy, it's contiguous
        else:
            in_tensor = self.npArrToTensor(self.in_patches[self.indices[index]].copy()) # Not contiguous, and can't create tensor without actually copying

        if self.gt_patches[self.indices[index]].data.contiguous:
            out_tensor = self.npArrToTensor(self.gt_patches[self.indices[index]]) # No need to copy, it's contiguous
        else:
            out_tensor = self.npArrToTensor(self.gt_patches[self.indices[index]].copy()) # Not contiguous, and can't create tensor without actually copying
        return in_tensor,out_tensor
