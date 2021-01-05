import numpy as np


def bayer2D_to_4_channels(im):
    im = np.maximum(im - 512, 0) / (16383 - 512)  # subtract the black level

    im = np.expand_dims(im, axis=2)
    img_shape = im.shape
    H = img_shape[0]
    W = img_shape[1]

    out = np.concatenate((im[0:H:2, 0:W:2, :],
                          im[0:H:2, 1:W:2, :],
                          im[1:H:2, 1:W:2, :],
                          im[1:H:2, 0:W:2, :]), axis=2)
    return out


def rgbToBayer(image):
    result = np.zeros((image.shape[0], image.shape[1]), dtype=np.float32)
    h, w, c = image.shape
    for row in range(h):
        for col in range(w):
            if row % 2 == 0:
                channel = 0 if col % 2 == 0 else 1
            else:
                channel = 1 if col % 2 == 0 else 2
            result[row, col] = image[row, col, channel] * 64 + 512

    return bayer2D_to_4_channels(result)
