################################################################################
# MIT License
#
# Copyright (c) 2020 Phillip Lippe
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to conditions.
#
# Author: Deep Learning Course | Fall 2020
# Date Created: 2020-11-22
################################################################################

import argparse
import os
import datetime
import statistics
import random

from tqdm import tqdm, trange
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from torchvision.utils import make_grid, save_image

from mnist import mnist
from models import GeneratorMLP, DiscriminatorMLP
from utils import TensorBoardLogger


class GAN(nn.Module):

    def __init__(self, hidden_dims_gen, hidden_dims_disc, dp_rate_gen,
                 dp_rate_disc, z_dim, *args, **kwargs):
        """
        PyTorch Lightning module that summarizes all components to train a GAN.

        Inputs:
            hidden_dims_gen  - List of hidden dimensionalities to use in the
                              layers of the generator
            hidden_dims_disc - List of hidden dimensionalities to use in the
                               layers of the discriminator
            dp_rate_gen      - Dropout probability to use in the generator
            dp_rate_disc     - Dropout probability to use in the discriminator
            z_dim            - Dimensionality of latent space
        """
        super().__init__()
        self.z_dim = z_dim

        self.generator = GeneratorMLP(z_dim=z_dim,
                                      hidden_dims=hidden_dims_gen,
                                      dp_rate=dp_rate_gen)
        self.discriminator = DiscriminatorMLP(hidden_dims=hidden_dims_disc,
                                              dp_rate=dp_rate_disc)

    @torch.no_grad()
    def sample(self, batch_size):
        """
        Function for sampling a new batch of random images from the generator.

        Inputs:
            batch_size - Number of images to generate
        Outputs:
            x - Generated images of shape [B,C,H,W]
        """
        z = torch.randn(size=(batch_size, self.z_dim), device=self.device)
        x = self.generator(z)

        return x

    @torch.no_grad()
    def interpolate(self, batch_size=2, interpolation_steps=3):
        """
        Function for interpolating between a batch of pairs of randomly sampled
        images. The interpolation is performed on the latent input space of the
        generator.

        Inputs:
            batch_size          - Number of image pairs to generate
            interpolation_steps - Number of intermediate interpolation points
                                  that should be generated.
        Outputs:
            x - Generated images of shape [B,interpolation_steps+2,C,H,W]
        """

        z_1 = torch.randn(size=(batch_size, self.z_dim), device=self.device)
        z_2 = torch.randn(size=(batch_size, self.z_dim), device=self.device)

        inter_steps = torch.linspace(0, 1, steps=interpolation_steps+2, device=self.device)

        interpolate_images = []
        for i in range(interpolation_steps+2):
            inter_z_step = z_2 * inter_steps[i] + z_1 * (1 - inter_steps[i])
            inter_x = self.generator(inter_z_step)
            interpolate_images.append(inter_x)
            # print(inter_z_step)
        
        x = torch.stack(interpolate_images).permute(1, 0, 2, 3, 4)# .view(-1, self.z_dim) 
        # print(inter_vectors.shape) 

        # x = torch.reshape(x, shape=(batch_size, interpolation_steps+2, 1,28,28))
    
        return x

    def generator_step(self, x_real):
        """
        Training step for the generator. Note that you do *not* need to take
        any special care of the discriminator in terms of stopping the
        gradients to its parameters, as this is handled by having two different
        optimizers. Before the discriminator's gradients in its own step are
        calculated, the previous ones have to be set to zero.

        Inputs:
            x_real - Batch of images from the dataset
        Outputs:
            loss - The loss for the generator to optimize
            logging_dict - Dictionary of string to Tensor that should be added
                           to our TensorBoard logger
        """
        batch_size = x_real.shape[0]

        # x_sample = self.sample(batch_size)
        x_sample = torch.randn(size=(batch_size, self.z_dim), device=self.device, requires_grad=True)
        x_sample = self.generator(x_sample)

        x_output = self.discriminator(x_sample)
        
        x_target = torch.ones_like(x_output)

        loss = F.binary_cross_entropy_with_logits(x_output, x_target)

        acc_gen = calculate_accuracy(torch.sigmoid(x_output), x_target)
        
        logging_dict = {"loss": loss}

        acc_gen_dict = {"acc": acc_gen}

        return loss, logging_dict, acc_gen_dict

    def discriminator_step(self, x_real):
        """
        Training step for the discriminator. Note that you do not have to use
        the same generated images as in the generator_step. It is simpler to
        sample a new batch of "fake" images, and use those for training the
        discriminator. It has also been shown to stabilize the training.
        Remember to log the training loss, and other potentially interesting 
        metrics.

        Inputs:
            x_real - Batch of images from the dataset
        Outputs:
            loss - The loss for the discriminator to optimize
            logging_dict - Dictionary of string to Tensor that should be added
                           to our TensorBoard logger
        """

        # Remark: there are more metrics that you can add. 
        # For instance, how about the accuracy of the discriminator?
        batch_size = x_real.shape[0]

        # training on real data
        output_real = self.discriminator(x_real)

        # generate targets for real data
        target_real = torch.ones_like(output_real)

        # copute real loss
        loss_real = F.binary_cross_entropy_with_logits(output_real, target_real)

        # calculate accuracy on real images, need to apply sigmoid 
        acc_real = calculate_accuracy(torch.sigmoid(output_real), target_real)

        # sample batch of fake images
        # x_fake = self.sample(batch_size)
        x_fake = torch.randn(size=(batch_size, self.z_dim), device=self.device, requires_grad=True)

        x_fake = self.generator(x_fake)

        # training on fake images
        output_fake = self.discriminator(x_fake)

        # generate targets for fake data
        target_fake = torch.zeros_like(output_fake)

        # compute fake loss
        loss_fake = F.binary_cross_entropy_with_logits(output_fake, target_fake)

        # accuracy fake
        acc_fake = calculate_accuracy(torch.sigmoid(output_fake), target_fake)

        loss = loss_real + loss_fake

        logging_dict = {"loss": loss}
        # raise NotImplementedError

        acc_log_disc = {"acc_real": acc_real, "acc_fake":acc_fake}

        return loss, logging_dict, acc_log_disc

    @property
    def device(self):
        """
        Property function to get the device on which the model is.
        """
        return self.generator.device


def generate_and_save(model, epoch, summary_writer, batch_size=64):
    """
    Function that generates and save samples from the GAN.
    The generated samples images should be added to TensorBoard and,
    eventually saved inside the logging directory.

    Inputs:
        model - The GAN model that is currently being trained.
        epoch - The epoch number to use for TensorBoard logging and saving of the files.
        summary_writer - A TensorBoard summary writer to log the image samples.
        batch_size - Number of images to generate/sample
    """
    # Hints:
    # - You can access the logging directory via summary_writer.log_dir
    # - Use torchvision function "make_grid" to create a grid of multiple images
    # - Use torchvision function "save_image" to save an image grid to disk

    x_samples = model.sample(batch_size)

    grid_img = make_grid(x_samples)

    # save image and store to tensorboard
    save_image(grid_img, 'samples_GAN/samples_epoch'+str(epoch)+'.png')

    summary_writer.add_image("samples_epoch"+str(epoch), grid_img, epoch)

    # raise NotImplementedError


def interpolate_and_save(model, epoch, summary_writer, batch_size=10,
                         interpolation_steps=5):
    """
    Function that generates and save the interpolations from the GAN.
    The generated samples and mean images should be added to TensorBoard and,
    if self.save_to_disk is True, saved inside the logging directory.

    Inputs:
        model - The GAN model that is currently being trained.
        epoch - The epoch number to use for TensorBoard logging and saving of the files.
        summary_writer - A TensorBoard summary writer to log the image samples.
        batch_size - Number of images to generate/sample
        interpolation_steps - Number of interpolation steps to perform
                              between the two random images.
    """
    # Hints:
    # - You can access the logging directory path via summary_writer.log_dir
    # - Use the torchvision function "make_grid" to create a grid of multiple images
    # - Use the torchvision function "save_image" to save an image grid to disk
    
    # You also have to implement this function in a later question of the assignemnt. 
    # By default it is skipped to allow you to test your other code so far. 
    # print("WARNING: Interpolation function has not been implemented yet.")
    # output interpolate [B,interpolation_steps+2,C,H,W]
    images = model.interpolate(batch_size, interpolation_steps)
  
    B, steps, c, h, w = images.shape
    
    images = torch.reshape(images, shape=(B*steps, c, h, w))

    img_grid = make_grid(images, nrow=interpolation_steps+2)

    save_image(img_grid, 'samples_GAN/interpolation'+str(epoch)+'.png', normalize=True)

    summary_writer.add_image("interpolation_epoch"+str(epoch), img_grid, epoch)



def train_gan(model, train_loader,
              logger_gen, logger_disc,
              optimizer_gen, optimizer_disc):
    """
    Function for training a GAN model on a dataset for a single epoch.

    Inputs:
        model - GAN model to train
        train_loader - Data Loader for the dataset you want to train on
        logger_gen - Logger object for the generator (see utils.py)
        logger_disc - Logger object for the discriminator (see utils.py) 
        optimizer - The optimizer used to update the parameters
    """
    model.train()

    for imgs, _ in train_loader:
        imgs = imgs.to(model.device)

        # Remark: add the logging dictionaries via 
        # "logger_gen.add_values(logging_dict)" for the generator, and
        # "logger_disc.add_values(logging_dict)" for discriminator
        # (both loggers should get different dictionaries, the outputs 
        #  of the respective step functions)

        # Generator update 
        loss_gen, log_dict_gen, log_acc_gen = model.generator_step(imgs)

        # zero gradients and backpropagate
        optimizer_gen.zero_grad()
        loss_gen.backward()
        optimizer_gen.step()

        # log gen loss and accuracy
        logger_gen.add_values(log_dict_gen)
        logger_gen.add_values(log_acc_gen)

        # Discriminator update
        loss_disc, log_dict_disc, log_acc_disc = model.discriminator_step(imgs)

        # backpropagate
        optimizer_disc.zero_grad()
        loss_disc.backward()
        optimizer_disc.step()

        # log disc loss and accuracy
        logger_disc.add_values(log_dict_disc)
        logger_disc.add_values(log_acc_disc)


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def calculate_accuracy(predictions, targets):
    """
    Calculate accuracy between predicitons and targets
    """
    batch_size, _ = predictions.shape
    
    predictions = np.where(predictions.cpu().detach().numpy() > 0.5, 1.0, 0.0)
    # predictions = torch.where(predictions > 0.5, 1.0, 0.0)
    
    correct = np.sum(predictions == targets.cpu().detach().numpy())
    
    accuracy = correct / float(batch_size)

    return torch.tensor(accuracy)


def main(args):
    """
    Main Function for the full training loop of a GAN model.
    Makes use of a separate train function for a single epoch.
    Remember to implement the optimizers, everything else is provided.

    Inputs:
        args - Namespace object from the argument parser
    """
    if args.seed is not None:
        seed_everything(args.seed)

    # Preparation of logging directories
    experiment_dir = os.path.join(
        args.log_dir, datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S"))
    checkpoint_dir = os.path.join(
        experiment_dir, 'checkpoints')
    os.makedirs(experiment_dir, exist_ok=True)
    os.makedirs(checkpoint_dir, exist_ok=True)

    train_loader = mnist(batch_size=args.batch_size,
                         num_workers=args.num_workers)

    # Create model
    model = GAN(hidden_dims_gen=args.hidden_dims_gen,
                hidden_dims_disc=args.hidden_dims_disc,
                dp_rate_gen=args.dp_rate_gen,
                dp_rate_disc=args.dp_rate_disc,
                z_dim=args.z_dim)

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    model = model.to(device)

    # Create two separate optimizers for generator and discriminator
    # You can use the Adam optimizer for both models.
    # It is recommended to reduce the momentum (beta1) to e.g. 0.5
    optimizer_gen = torch.optim.Adam(model.generator.parameters(), lr=args.lr, betas=(0.5, 0.999))
    optimizer_disc = torch.optim.Adam(model.discriminator.parameters(), lr=args.lr, betas=(0.5, 0.999))
    # raise NotImplementedError

    # TensorBoard logger
    # See utils.py for details on "TensorBoardLogger" class
    summary_writer = SummaryWriter(experiment_dir)
    logger_gen = TensorBoardLogger(summary_writer, name="generator")
    logger_disc = TensorBoardLogger(summary_writer, name="discriminator")

    # Initial generation before training
    generate_and_save(model, 0, summary_writer, 64)

    # Training loop
    print(f"Using device {device}")
    epoch_iterator = (trange(args.epochs, desc="GAN")
                      if args.progress_bar else range(args.epochs))
    for epoch in epoch_iterator:
        # Training epoch
        train_iterator = (tqdm(train_loader, desc="Training", leave=False)
                          if args.progress_bar else train_loader)
        train_gan(model, train_iterator,
                  logger_gen, logger_disc,
                  optimizer_gen, optimizer_disc)

        # Logging images
        if (epoch + 1) % 10 == 0:
            print("Epoch {:d}".format(epoch+1))
            generate_and_save(model, epoch+1, summary_writer)
            interpolate_and_save(model, epoch+1, summary_writer)

        # Saving last model (only every 10 epochs to reduce IO traffic)
        # As we do not have a validation step, we cannot determine the "best"
        # checkpoint during training except looking at the samples.
        # import pdb
        # pdb.set_trace()
        if (epoch + 1) % 10 == 0:
            torch.save(model.state_dict(),
                       os.path.join(checkpoint_dir, "model_checkpoint.pt"))

    print("Main completed")


if __name__ == '__main__':
    # Feel free to add more argument parameters
    parser = argparse.ArgumentParser()

    # Model hyperparameters
    parser.add_argument('--z_dim', default=32, type=int,
                        help='Dimensionality of latent space')
    parser.add_argument('--hidden_dims_gen', default=[128, 256, 512],
                        type=int, nargs='+',
                        help='Hidden dimensionalities to use inside the ' +
                             'generator. To specify multiple, use " " to ' +
                             'separate them. Example: \"128 256 512\"')
    parser.add_argument('--hidden_dims_disc', default=[512, 256],
                        type=int, nargs='+',
                        help='Hidden dimensionalities to use inside the ' +
                             'discriminator. To specify multiple, use " " to ' +
                             'separate them. Example: \"512 256\"')
    parser.add_argument('--dp_rate_gen', default=0.1, type=float,
                        help='Dropout rate in the discriminator')
    parser.add_argument('--dp_rate_disc', default=0.3, type=float,
                        help='Dropout rate in the discriminator')

    # Optimizer hyperparameters
    parser.add_argument('--lr', default=2e-4, type=float,
                        help='Learning rate to use')
    parser.add_argument('--batch_size', default=128, type=int,
                        help='Batch size to use for training')

    # Other hyperparameters
    parser.add_argument('--epochs', default=250, type=int,
                        help='Number of epochs to train.')
    parser.add_argument('--seed', default=42, type=int,
                        help='Seed to use for reproducing results')
    parser.add_argument('--num_workers', default=4, type=int,
                        help='Number of workers to use in the data loaders.' +
                             'To have a truly deterministic run, this has to be 0.')
    parser.add_argument('--log_dir', default='GAN_logs/', type=str,
                        help='Directory where the PyTorch Lightning logs ' +
                             'should be created.')
    parser.add_argument('--progress_bar', action='store_true',
                        help='Use a progress bar indicator for interactive experimentation. ' +
                             'Not to be used in conjuction with SLURM jobs.')

    args = parser.parse_args()

    main(args)
