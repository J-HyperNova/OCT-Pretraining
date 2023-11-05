
import torch

from pixel_level_contrastive_learning import PixelCL
from torchvision import models, transforms, datasets
import wandb 
import os
import numpy as np
import skimage.io as io
from skimage.transform import resize
resnet = models.resnet50(pretrained=True)

from tqdm import tqdm
import torch.nn as nn


projection_dim = 256
similarity_temperature = 0.3

model= PixelCL(
    resnet,
    image_size = 128,
    hidden_layer_pixel = 'layer4',  # leads to output of 8x8 feature map for pixel-level learning
    hidden_layer_instance = -2,     # leads to output for instance-level learning
    projection_size = projection_dim,          # size of projection output, 256 was used in the paper
    projection_hidden_size = 2048,  # size of projection hidden dimension, paper used 2048
    moving_average_decay = 0.99,    # exponential moving average decay of target encoder
    ppm_num_layers = 1,             # number of layers for transform function in the pixel propagation module, 1 was optimal
    ppm_gamma = 2,                  # sharpness of the similarity in the pixel propagation module, already at optimal value of 2
    distance_thres = 0.7,           # ideal value is 0.7, as indicated in the paper, which makes the assumption of each feature map's pixel diagonal distance to be 1 (still unclear)
    similarity_temperature = similarity_temperature,   # temperature for the cosine similarity for the pixel contrastive loss
    alpha = 1.,                      # weight of the pixel propagation loss (pixpro) vs pixel CL loss
    use_pixpro = True,               # do pixel pro instead of pixel contrast loss, defaults to pixpro, since it is the best one
    cutout_ratio_range = (0.6, 0.8)  # a random ratio is selected from this range for the random cutout
)

#model=nn.DataParallel(model)
model=model.cuda()
epochs=50
batch_size=256
learning_rate=1e-4
opt = torch.optim.Adam(model.parameters(), lr=learning_rate)
run= wandb.init(
    # set the wandb project where this run will be logged
    project="PixelCL",
    config={
    "architecture": "PixelCL",
    "backbone": "resnet50",
    "dataset": "California_OCT Training Set",
    "epochs": epochs,
    "projection_dim": projection_dim,
    "learning_rate": learning_rate,
    "optimizer": "Adam",
    "batch_size": batch_size,
    "similarity_temperature": similarity_temperature
    })

def createDataloader():
    # create a dataloader
    transform=transforms.Compose([
        transforms.Resize((128,128),  antialias=None),
        transforms.ToTensor(),
        
    ])
    dataset = datasets.ImageFolder("/home-mscluster/jknopfmacher/Research/Datasets/CaliOCT/train", transform=transform)
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
    print("😀Created dataloader")
    return loader

def train():
    loader = createDataloader()
    losses = []
    for epoch in tqdm(range(epochs), colour="green"):
        this_epoch_loss = 0
        this_epoch_positve_pairs = 0
        for images, _ in tqdm(loader):
            images=images.cuda()
            loss, positvie_pairs = model(images, return_positive_pairs = True)
            #log loss
            this_epoch_loss += loss.item()
            opt.zero_grad()
            loss.backward()
            opt.step()
            this_epoch_positve_pairs += positvie_pairs
        #make sure to log the loss only once per epoch
        
        wandb.log({"loss": this_epoch_loss/len(loader), "positive_pairs": this_epoch_positve_pairs/len(loader)})
        losses.append(this_epoch_loss/len(loader))
        #every 5 epochs, save the model checkpoint with the name being the config
        if epoch % 5 == 0:
            torch.save(model.state_dict(), "/home-mscluster/jknopfmacher/Research/PixelLevelContrast/Model/256_simclr_{}_{}_{}_{}.pth".format(batch_size, epoch, projection_dim, learning_rate))
            np.save("/home-mscluster/jknopfmacher/Research/PixelLevelContrast/loss/256_losses_{}_{}_{}_{}.npy".format(batch_size, epoch, projection_dim, learning_rate), np.array(losses))
            #save optimiser state
            torch.save(opt.state_dict(), "/home-mscluster/jknopfmacher/Research/PixelLevelContrast/Optim/256_opt_{}_{}_{}_{}.pth".format(batch_size, epoch, projection_dim, learning_rate))
            #save resnet state
            torch.save(resnet.state_dict(), "/home-mscluster/jknopfmacher/Research/PixelLevelContrast/Resnet/256_resnet_{}_{}_{}_{}.pth".format(batch_size, epoch, projection_dim, learning_rate))
    return losses
losses = train()
#now save 
#save the model
torch.save(model.state_dict(), "/home-mscluster/jknopfmacher/Research/PixelLevelContrast/Model/256_simclr_{}_{}_{}_{}.pth".format(batch_size, epochs, projection_dim, learning_rate))
#save the losses
np.save("/home-mscluster/jknopfmacher/Research/PixelLevelContrast/loss/256_losses_{}_{}_{}_{}.npy".format(batch_size, epochs, projection_dim, learning_rate), np.array(losses))
#save optimiser state
torch.save(opt.state_dict(), "/home-mscluster/jknopfmacher/Research/PixelLevelContrast/Optim/256_opt_{}_{}_{}_{}.pth".format(batch_size, epochs, projection_dim, learning_rate))
#save resnet state
torch.save(resnet.state_dict(), "/home-mscluster/jknopfmacher/Research/PixelLevelContrast/Resnet/256_resnet_{}_{}_{}_{}.pth".format(batch_size, epochs, projection_dim, learning_rate))

    

