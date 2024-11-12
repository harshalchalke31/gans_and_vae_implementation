import torch.nn as nn
import torch

class DCDiscriminator(nn.Module):
    def __init__(self,img_channels,features_d):
        super().__init__(DCDiscriminator,self).__init__()
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
    
class DCGenerator(nn.Module):
    def __init__(self,z_dimension,img_channels,features_g):
        super(DCGenerator,self).__init__()
        self.gen = nn.Sequential(
            # N x 1 x 1
            self.conv_T_block(
                inchannels=z_dimension,
                outchannels=features_g*16,
                kernel_size=4,
                stride=2,
                padding=0
            ),
            # N x f*16 x 4 x 4
            self.conv_T_block(
                inchannels=features_g*16,
                outchannels=features_g*8,
                kernel_size=4,
                stride=2,
                padding=1
            ),
            # N x f*8 x 8 x 8
            self.conv_T_block(
                inchannels=features_g*8,
                outchannels=features_g*4,
                kernel_size=4,
                stride=2,
                padding=1
            ),
            # N x f*4 x 16 x 16
            self.conv_T_block(
                inchannels=features_g*4,
                outchannels=features_g*2,
                kernel_size=4,
                stride=2,
                padding=1
            ),
            # N x f*2 x 32 x 32
            nn.ConvTranspose2d(
                in_channels=features_g*2,
                out_channels=img_channels,
                kernel_size=4,
                stride=2,
                padding=1
            ),
            nn.Tanh() # [-1,1]

        )
    def conv_T_block(inchannels,outchannels,kernel_size,stride,padding):
        return nn.Sequential(
            nn.ConvTranspose2d(
                in_channels=inchannels,
                out_channels=outchannels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding
            ),
            nn.BatchNorm2d(outchannels),
            nn.ReLU()
        )
    def forward(self,x):
        return self.gen(x)
    
def initialize_weights(model):
    for m in model.modules():
        if isinstance(m,(nn.Conv2d,nn.ConvTranspose2d,nn.BatchNorm2d)):
            nn.init.normal_(m.weight.data, 0.0,0.02)
