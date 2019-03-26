#-*- coding:utf-8 -*-

import torch
import torch.nn as nn
import torch.nn.functional as F


class ResNet(nn.Module):
      def __init__(self, input_channel, plain=64):
          super(ResNet, self).__init__()
          self.layers = nn.Sequential()
          self.layers.add_module('bn1', nn.BatchNorm2d(input_channel))
          self.layers.add_module('relu1', nn.ReLU())
          self.layers.add_module('conv1', nn.Conv2d(input_channel, plain, kernel_size=3, padding=1))
          self.layers.add_module('bn2', nn.BatchNorm2d(plain))
          self.layers.add_module('relu2', nn.ReLU())
          self.layers.add_module('dropout', nn.Dropout2d(0.3))
          self.layers.add_module('conv2', nn.Conv2d(plain, input_channel, kernel_size=3, padding=1))

      def forward(self, x):
          return self.layers(x) + x


class Generator(nn.Module):
      def __init__(self, input_channel=2, down=3, up=2, resnum=3, ndf=32):
          super(Generator, self).__init__()
          self.encoder = nn.Sequential()
          self.resnet = nn.Sequential()
          self.decoder = nn.Sequential()
          self.ndown = down
          self.nup = up
          for i in range(down):
              if i == 0:
                 self.encoder.add_module('conv'+str(i+1), nn.Conv2d(input_channel, (i+1)*ndf, kernel_size=3, padding=1, stride=2))
              else:
                 self.encoder.add_module('conv'+str(i+1), nn.Conv2d(i*ndf, (i+1)*ndf, kernel_size=3, padding=1, stride=2))
              self.encoder.add_module('bn'+str(i+1), nn.BatchNorm2d( (i+1)*ndf) ) 
              self.encoder.add_module('relu'+str(i+1), nn.ReLU())

          for i in range(resnum):
              self.resnet.add_module('resnet'+str(i+1), ResNet(down*ndf))
 
          for i in range(up):
              if i == up-1:
                 out = nn.ConvTranspose2d( (up+1-i)*ndf, 1, kernel_size=4, padding=1, stride=2)
                 self.decoder.add_module('convt'+str(i+1), out)
                 self.decoder.add_module('tanh', nn.Tanh())
              else:
                 self.decoder.add_module('convt'+str(i+1), nn.ConvTranspose2d( (up+1-i)*ndf, (up-i)*ndf, kernel_size=4, padding=1, stride=2))
                 self.decoder.add_module('bn'+str(i+1+down), nn.BatchNorm2d( (up-i)*ndf ))
                 self.decoder.add_module('relu'+str(i+1+down), nn.ReLU())

          torch.nn.init.normal(out.weight, mean=0.0, std=0.02)
          torch.nn.init.constant(out.bias, 0.0)
 
      def forward(self, x):
          encoded = self.encoder(x)
          x = self.resnet(encoded)
          decoded = self.decoder(x)
          return decoded, encoded



class Discriminator(nn.Module):
      def __init__(self, input_channel=1, down=5, ndf=1024):
          super(Discriminator, self).__init__()
          self.CBL = nn.Sequential()
          for i in range(down):
              if i == 0:
                 self.CBL.add_module('conv'+str(i+1), nn.Conv2d(input_channel, ndf//2**i, kernel_size=3, padding=1, stride=2))
              else:
                 self.CBL.add_module('conv'+str(i+1), nn.Conv2d(ndf//2**(i-1), ndf//2**i, kernel_size=3, padding=1, stride=2))
              self.CBL.add_module('bn'+str(i+1), nn.BatchNorm2d(ndf//2**i))
              self.CBL.add_module('leakyrelu'+str(i+1), nn.LeakyReLU(0.2))

          out = nn.Conv2d(ndf//2**(down-1), 1, kernel_size=3, padding=1, stride=2)
          self.output = nn.Sequential()
          self.output.add_module('lastconv', out)
          self.output.add_module('generalavgpool', nn.AvgPool2d(kernel_size=2))
          self.output.add_module('sigmoid', nn.Sigmoid())
          torch.nn.init.normal(out.weight, mean=0.0, std=0.02)
          torch.nn.init.constant(out.bias, 0.0)
 
      def forward(self, x):
          x = self.CBL(x)
          x = self.output(x)
          return x.view(-1, 1)

if __name__ == '__main__':
   G = Generator()
   print(G)
   D = Discriminator()
   x = torch.randn(1, 1, 100, 100)
   o = D(x)
   print(o.shape)
