#-*- coding:utf-8 -*-

import torch
import torch.nn as nn
import torch.optim as optim

from torch.autograd import Variable
from torch.utils.data import DataLoader, TensorDataset
from network import *
from glob import glob
import matplotlib.pyplot as plt
import numpy as np
import cv2
import os

def get_data(batch_size):
    pathes1 = np.sort(glob('./data/image1/*'))
    pathes2 = np.sort(glob('./data/image2/*'))
    images1 = []
    images2 = []
    for path1, path2 in zip(pathes1, pathes2):
        img1 = cv2.imread(path1, 0)
        img2 = cv2.imread(path2, 0)
        z = np.random.randn(200, 200)
        input1 = np.array([z, img1 / 255.0])
        images1.append(input1)
        images2.append(img2 / 255.0)
    data = TensorDataset(torch.Tensor(images1).type(torch.FloatTensor), torch.Tensor(images2).type(torch.FloatTensor))
    data_loader = DataLoader(data, batch_size=batch_size, shuffle=True)
    return data_loader

def plot_loss(d_losses, g_losses, num_epoch, num_epochs, save=False, save_dir='./results/loss', show=False):
    fig, ax = plt.subplots()
    ax.set_xlim(0, num_epochs)
    ax.set_ylim(0, max(np.max(g_losses), np.max(d_losses))*1.1)
    plt.xlabel('Epoch {0}'.format(num_epoch + 1))
    plt.ylabel('Loss Values')
    plt.plot(d_losses, label='Discriminator')
    plt.plot(g_losses, label='Generator')
    plt.legend()

    if save:
        if not os.path.exists(save_dir):
            os.mkdir(save_dir)
        save_fn = save_dir + 'MpixGAN_losses_epoch_{:d}'.format(num_epoch + 1) + '.png'
        plt.savefig(save_fn)

    if show:
        plt.show()
    else:
        plt.close()

def train():
    EPOCH = 5
    learning_rate = 0.00001
    betas = (0.5, 0.999)
    batch_size = 16

    G = Generator()
    D = Discriminator()

    train_loader = get_data(batch_size)  
    criterion = torch.nn.BCELoss()
    
    G_optimizer = torch.optim.Adam(G.parameters(), lr=learning_rate, betas=betas)
    D_optimizer = torch.optim.Adam(D.parameters(), lr=learning_rate, betas=betas)

    D_avg_losses = []
    G_avg_losses = []

    for epoch in range(EPOCH):
        D_losses = []
        G_losses = []

        for i, (I1, I2) in enumerate(train_loader):
            mini_batch = I1.size()[0]
            I2 = I2.unsqueeze(1)
            x_ = Variable(I2)

            y_real_ = Variable(torch.ones(mini_batch))
            y_fake_ = Variable(torch.zeros(mini_batch))

            D_real_decision = D(x_).squeeze()
            D_real_loss = criterion(D_real_decision, y_real_)


            gen_image, _ = G(I1)

            D_fake_decision = D(gen_image).squeeze()
            D_fake_loss = criterion(D_fake_decision, y_fake_)

            D_loss = D_real_loss + D_fake_loss
            D.zero_grad()
            D_loss.backward()
            D_optimizer.step()   

            x = Variable(I1)
            y_real__ = Variable(torch.ones(mini_batch))
            gen_image, _ = G(x)

            D_fake_decision = D(gen_image).squeeze()
            G_loss = criterion(D_fake_decision, y_real__)

            D.zero_grad()
            G.zero_grad()
            G_loss.backward()
            G_optimizer.step()


            D_losses.append(D_loss.data[0])
            G_losses.append(G_loss.data[0])
             
            print('Epoch [%d/%d], Step [%d/%d], D_loss: %.4f, G_loss: %.4f'
                  % (epoch+1, EPOCH, i+1, len(train_loader), D_loss.data[0], G_loss.data[0]))
            if i > 100:
               break
       

        D_avg_loss = torch.mean(torch.FloatTensor(D_losses))
        G_avg_loss = torch.mean(torch.FloatTensor(G_losses))
        D_avg_losses.append(D_avg_loss)
        G_avg_losses.append(G_avg_loss)

        plot_loss(D_avg_losses, G_avg_losses, epoch, EPOCH, save=True)

    torch.save(G.state_dict(), 'mpixG')
    torch.save(D.state_dict(), 'mpixD')


if __name__ == '__main__':
   train()
