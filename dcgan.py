import torch.nn as nn
import torch

class Discriminator(nn.Module):
    def __init__(self,img_channels,features_d):
        super().__init__(Discriminator,self).__init__()
        self.disc = nn.Sequential(
            nn.Conv2d(in_channels=img_channels,
                      out_channels=features_d,
                      kernel_size=4,
                      stride=2,
                      padding=1),
            nn.LeakyReLU(negative_slope=0.2),
            self.conv_block(inchannels=features_d,
                            outchannels=features_d*2,
                            kernel_size=4,
                            stride=2,
                            padding=1),
            self.conv_block(inchannels=features_d*2,
                            outchannels=features_d*4,
                            kernel_size=4,
                            stride=2,
                            padding=1),
            self.conv_block(inchannels=features_d*4,
                            outchannels=features_d*8,
                            kernel_size=4,
                            stride=2,
                            padding=1),
            nn.Conv2d(in_channels=features_d*8,
                      out_channels=1,
                      kernel_size=4,
                      stride=2,
                      padding=0),
            nn.Sigmoid(),

        )
    
    def conv_block(self,inchannels,outchannels,kernel_size,stride,padding):
        return nn.Sequential(
            nn.Conv2d(in_channels=inchannels,
                      out_channels=outchannels,
                      kernel_size=kernel_size,
                      stride=stride,
                      bias=False),
            nn.BatchNorm2d(outchannels),
            nn.LeakyReLU(negative_slope=0.2)
        )
    
    def forward(self,x):
        return self.disc(x)
    
class Generator(nn.Module):
    def __init__(self,z_dimension,img_channels,features_g):
        super(Generator,self).__init__()
    def conv_T_block(inchannels,outchannels,kernel_size,stride,)