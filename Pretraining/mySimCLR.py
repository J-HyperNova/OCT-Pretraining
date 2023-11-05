from simclr import SimCLR
from simclr.modules.transformations import TransformsSimCLR
import torch


from torchvision import models, transforms, datasets
import wandb 
import os
import numpy as np
import skimage.io as io
from skimage.transform import resize
resnet = models.resnet50(weights="ResNet50_Weights.DEFAULT")
import matplotlib.pyplot as plt
from tqdm import tqdm
import torch.nn as nn
projection_dim = 256
n_feautres=resnet.fc.in_features
model=SimCLR(resnet,projection_dim,n_feautres)
model=nn.DataParallel(model)
model=model.cuda()
epochs=50
batch_size=64
learning_rate=3e-4
opt = torch.optim.Adam(model.parameters(), lr=learning_rate)
run= wandb.init(
    # set the wandb project where this run will be logged
    project="Simclr",
    config={
    "architecture": "Simclr",
    "backbone": "resnet50",
    "dataset": "California_OCT Training Set",
    "epochs": epochs,
    "projection_dim": projection_dim,
    "learning_rate": learning_rate,
    "optimizer": "Adam",
    "batch_size": batch_size
    })


def createDataloader():
    # create a dataloader
    
    transform = TransformsSimCLR(size=256)
    dataset = datasets.ImageFolder("/home-mscluster/jknopfmacher/Research/Datasets/CaliOCT/train", transform=transform)
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

    return loader

def train():
    loader = createDataloader()
    losses = []
    for epoch in tqdm(range(epochs), colour="green"):
        this_epoch_loss = 0
        for (x_i, x_j),_ in tqdm(loader, colour="blue"):
            #print type
          
            x_i = x_i.cuda()
            x_j = x_j.cuda()

            loss= model(x_i, x_j)[0]
            #get single loss value
            loss = torch.mean(loss)
            this_epoch_loss += loss.item()
            opt.zero_grad()
            loss.backward()
            opt.step()
        wandb.log({"loss": this_epoch_loss/len(loader)})
        losses.append(this_epoch_loss/len(loader))
        #every 5 epochs, save the model checkpoint with the name being the config
        if epoch % 5 == 0:
            torch.save(model.state_dict(), "/home-mscluster/jknopfmacher/Research/SIMCLR/Model/256_simclr_{}_{}_{}_{}.pth".format(batch_size, epoch, projection_dim, learning_rate))
            np.save("/home-mscluster/jknopfmacher/Research/SIMCLR/loss/256_losses_{}_{}_{}_{}.npy".format(batch_size, epoch, projection_dim, learning_rate), np.array(losses))
            #save optimiser state
            torch.save(opt.state_dict(), "/home-mscluster/jknopfmacher/Research/SIMCLR/optim/256_opt_{}_{}_{}_{}.pth".format(batch_size, epoch, projection_dim, learning_rate))
            #save resnet state
            torch.save(resnet.state_dict(), "/home-mscluster/jknopfmacher/Research/SIMCLR/Resnet/256_resnet_{}_{}_{}_{}.pth".format(batch_size, epoch, projection_dim, learning_rate))
    return losses
losses = train()
#now save 
#save the model
torch.save(model.state_dict(), "/home-mscluster/jknopfmacher/Research/SIMCLR/Model/256_simclr_{}_{}_{}_{}.pth".format(batch_size, epochs, projection_dim, learning_rate))
#save the losses
np.save("/home-mscluster/jknopfmacher/Research/SIMCLR/loss/256_losses_{}_{}_{}_{}.npy".format(batch_size, epochs, projection_dim, learning_rate), np.array(losses))
#save optimiser state
torch.save(opt.state_dict(), "/home-mscluster/jknopfmacher/Research/SIMCLR/optim/256_opt_{}_{}_{}_{}.pth".format(batch_size, epochs, projection_dim, learning_rate))
#save resnet state
torch.save(resnet.state_dict(), "/home-mscluster/jknopfmacher/Research/SIMCLR/Resnet/256_resnet_{}_{}_{}_{}.pth".format(batch_size, epochs, projection_dim, learning_rate))

    

