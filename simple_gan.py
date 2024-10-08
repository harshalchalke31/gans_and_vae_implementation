import torch.nn as nn


class Discriminator(nn.Module):
    def __init__(self,image_dimension):  # image_dimension = 784, 1x28x28 image of MNIST
        super().__init__()
        self.disc = nn.Sequential(
            nn.Linear(in_features=image_dimension,out_features=128),
            nn.LeakyReLU(negative_slope=0.1), # why leaky relu is preferred in GANs?
            nn.Linear(in_features=128,out_features=1),
            nn.Sigmoid(),  # to ensure outputs are between 0 and 1
        )

    def forward(self,x):
        return self.disc(x)

class Generator(nn.Module):
    def __init__(self, z_dimension, image_dimension):
        super().__init__()
        self.gen = nn.Sequential(
            nn.Linear(in_features=z_dimension,out_features=256),
            nn.LeakyReLU(negative_slope=0.1),
            nn.Linear(in_features=256,out_features=image_dimension),
            nn.Tanh(), # we are going to normalize input from MNIST to be -1 to 1, therefore, our output should also be from -1 to 1 
        )

    def forward(self,x):
        return self.gen(x)

