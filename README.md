# CNNAutoencoder

This repository is a PyTorch implementation of the See In The Dark paper (see [Citation](#Citation) for more info).
This project is part of our Master's degree in Digital Sciences track at the Centre de Recherche Interdisciplinaire, Paris, France.

The goal of this project is to learn an AI how to turn low light, noisy images into their correctly exposed counterpart, thus reducing digital noise.
To do so, the authors of the paper this model is based on trained a U-net with under-exposed images. The expected output being images with longer exposure time.

You may find several implementations on the Internet, but I didn't find any that uses torch's Dataset idiomatic way of training.

# Citation

Chen Chen, Qifeng Chen, Jia Xu, and Vladlen Koltun, "Learning to See in the Dark", in CVPR, 2018.
