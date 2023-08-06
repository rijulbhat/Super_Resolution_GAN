import argparse
import os
import numpy as np
import math
import itertools
import sys

import torchvision.transforms as transforms
from torchvision.utils import save_image, make_grid

from torch.utils.data import DataLoader
from torch.autograd import Variable

from models import *
from datasets import *

import torch.nn as nn
import torch.nn.functional as F
import torch

if __name__ == "__main__":

    generator = GeneratorResNet()
    discriminator = Discriminator(input_shape=(3, 64, 64))
    feature_extractor = FeatureExtractor()
    epoch = sys.argv[1]
    path_gen = "saved_models/generator_" + epoch + ".pth"
    path_disc = "saved_models/discriminator_" + epoch + ".pth"
    generator.load_state_dict(torch.load(path_gen))
    discriminator.load_state_dict(torch.load(path_disc))

    dataloader = DataLoader(
            ImageDataset("../../data/img_align_celeba_test", hr_shape=(64,64)),
            batch_size=1,
            shuffle=True,
            num_workers=4,
        )
    Tensor = torch.Tensor
    for i, imgs in enumerate(dataloader):
        # Configure model input
        imgs_lr = Variable(imgs["lr"].type(Tensor))
        imgs_hr = Variable(imgs["hr"].type(Tensor))

        # Generate a high resolution image from low resolution input
        gen_hr = generator(imgs_lr)

        # Save image grid with upsampled inputs and SRGAN outputs
        
        gen_hr = make_grid(gen_hr, nrow=1, normalize=True)
        image_name = "srgan_image_"+epoch+".png"
        save_image(gen_hr, image_name, normalize=False)

        imgs_lr = nn.functional.interpolate(imgs_lr, scale_factor=4)

        save_image(imgs_lr, "lr_upscaled_image.png", normalize=False)

        save_image(imgs_hr, "hr_image.png")
