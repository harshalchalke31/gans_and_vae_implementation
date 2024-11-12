import torch.nn as nn
import torch
import torch.nn.functional as F


class Discriminator(nn.Module):
    def __init__(self,image_dimension):  # image_dimension = 784, 1x28x28 image of MNIST
        super().__init__()
        self.disc = nn.Sequential(
            nn.Linear(in_features=image_dimension,out_features=256),
            nn.LeakyReLU(0.01), # why leaky relu is preferred in GANs?
            nn.Dropout(0.3),

            nn.Linear(in_features=256,out_features=128),
            nn.LeakyReLU(0.01),
            nn.Dropout(0.3),

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
            nn.LeakyReLU(0.01),
            
            nn.Linear(in_features=256,out_features=128),
            nn.LeakyReLU(0.01),
            
            nn.Linear(in_features=128,out_features=image_dimension),
            nn.Tanh(), # we are going to normalize input from MNIST to be -1 to 1, therefore, our output should also be from -1 to 1 
        )

    def forward(self,x):
        return self.gen(x)


class DCDiscriminator(nn.Module):
    def __init__(self, img_channels, features_d):
        super(DCDiscriminator, self).__init__()
        self.disc = nn.Sequential(
            nn.Conv2d(in_channels=img_channels,
                      out_channels=features_d,
                      kernel_size=4,
                      stride=2,
                      padding=1),
            nn.LeakyReLU(negative_slope=0.02),
            self.conv_block(inchannels=features_d,
                            outchannels=features_d * 2,
                            kernel_size=4,
                            stride=2,
                            padding=1),
            self.conv_block(inchannels=features_d * 2,
                            outchannels=features_d * 4,
                            kernel_size=4,
                            stride=2,
                            padding=1),
            nn.Conv2d(in_channels=features_d * 4,
                      out_channels=1,
                      kernel_size=4,
                      stride=1,
                      padding=0,
                      bias=False),
            nn.Sigmoid(),
        )

    def conv_block(self, inchannels, outchannels, kernel_size, stride, padding):
        return nn.Sequential(
            nn.Conv2d(in_channels=inchannels,
                      out_channels=outchannels,
                      kernel_size=kernel_size,
                      stride=stride,
                      padding=padding,
                      bias=False),
            nn.BatchNorm2d(outchannels),
            nn.LeakyReLU(negative_slope=0.2,inplace=True)
        )

    def forward(self, x):
        return self.disc(x)


class DCGenerator(nn.Module):
    def __init__(self, z_dimension, img_channels, features_g):
        super(DCGenerator, self).__init__()
        self.gen = nn.Sequential(
            # Input is N x z_dimension x 1 x 1
            self.conv_T_block(
                inchannels=z_dimension,
                outchannels=features_g * 8,
                kernel_size=4,
                stride=1,
                padding=0
            ),
            # N x f*8 x 4 x 4
            self.conv_T_block(
                inchannels=features_g * 8,
                outchannels=features_g * 4,
                kernel_size=4,
                stride=2,
                padding=1
            ),
            # N x f*4 x 8 x 8
            self.conv_T_block(
                inchannels=features_g * 4,
                outchannels=features_g * 2,
                kernel_size=4,
                stride=2,
                padding=1
            ),
            # N x f x 32 x 32
            nn.ConvTranspose2d(
                in_channels=features_g*2,
                out_channels=img_channels,
                kernel_size=4,
                stride=2,
                padding=1,
                bias=False
            ),
            nn.Tanh()  # [-1,1]
        )

    def conv_T_block(self, inchannels, outchannels, kernel_size, stride, padding):
        return nn.Sequential(
            nn.ConvTranspose2d(
                in_channels=inchannels,
                out_channels=outchannels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                bias=False
            ),
            nn.BatchNorm2d(outchannels),
            nn.ReLU(True)
        )

    def forward(self, x):
        return self.gen(x)


def initialize_weights(model):
    for m in model.modules():
        if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d, nn.BatchNorm2d)):
            nn.init.normal_(m.weight.data, 0.0, 0.02)


class VAE(nn.Module):
    def __init__(self, input_dim=784, hidden_dim=512, z_dim=50): 
        super(VAE, self).__init__()
        
        # encoder network
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc_mu = nn.Linear(hidden_dim, z_dim)
        self.fc_logvar = nn.Linear(hidden_dim, z_dim)
        
        # decoder network
        self.fc3 = nn.Linear(z_dim, hidden_dim)
        self.fc4 = nn.Linear(hidden_dim, input_dim)
    
    def encoder(self, x):
        h = F.relu(self.fc1(x))
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar
    
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def decoder(self, z):
        h = F.relu(self.fc3(z))
        return torch.sigmoid(self.fc4(h))
    
    def forward(self, x):
        mu, logvar = self.encoder(x.view(-1, 784))
        z = self.reparameterize(mu, logvar)
        x_recon = self.decoder(z)
        return x_recon, mu, logvar
