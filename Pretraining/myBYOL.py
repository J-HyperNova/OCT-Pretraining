import torch
from byol_pytorch import BYOL
from torchvision import models, transforms, datasets
import wandb 
import os
import numpy as np
import skimage.io as io
from skimage.transform import resize
resnet = models.resnet50(weights="ResNet50_Weights.DEFAULT") # load your in-built resnet model
import matplotlib.pyplot as plt
from tqdm import tqdm

image_size = 128 # size of your images
learner = BYOL( 
    resnet,
    image_size = image_size,
    hidden_layer = 'avgpool'
).cuda() # create your own BYOL model using resnet50 as the encoder
'''SET HYPERPARAMETERS'''
epochs=50
batch_size=320
learning_rate=3e-4
run= wandb.init(
    # set the wandb project where this run will be logged 
    project="BYOL",
    
    # track hyperparameters  
    config={
   
    "architecture": "BYOL",
    "backbone": "resnet50",
    "dataset": "California_OCT Training Set",
    "epochs": epochs,
    "hidden_layer": "avgpool",
    "learning_rate":learning_rate,
    "optimizer": "Adam",
    "batch_size": batch_size,
    "image_resolution": image_size
    }
)
opt = torch.optim.Adam(learner.parameters(), lr=learning_rate)

def createDataloader():
    # create a dataloader
    transform=transforms.Compose([
        transforms.Resize((image_size,image_size),  antialias=None),
        transforms.ToTensor(),
        
    ]) # resize the images to the desired size and convert them to tensors
    dataset = datasets.ImageFolder("/home-mscluster/jknopfmacher/Research/Datasets/CaliOCT/train", transform=transform) # create your own dataset
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True) # create your own dataloader and set shuffle to True
    print("😀Created dataloader")
    return loader


losses = [] # keep appending the losses to plot a graph later
loader = createDataloader() 
for i in tqdm(range(0,epochs), colour="green"):
    #loop over the batches of images
    #
    
    print("Starting epoch ", i)
    this_loss = 0
    for images, _ in tqdm(loader):
        images=images.cuda()
        loss = learner(images)
        #log loss
        this_loss += loss.item()
        opt.zero_grad()
        loss.backward()
        opt.step()
    #make sure to log the loss only once per epoch
    wandb.log({"loss": this_loss/len(loader)})
    losses.append(this_loss/len(loader))
    
    learner.update_moving_average() # update moving average of target encoder
    #save checkpoint every 10 epochs
    if i % 5 == 0:
        #save the model
        torch.save(learner.state_dict(), "/home-mscluster/jknopfmacher/Research/BYOL/checkpoint/{}_{}_{}_{}_BYOL.pth".format(batch_size, i, "resnet50", "avgpool"))
        torch.save(resnet.state_dict(), "/home-mscluster/jknopfmacher/Research/BYOL/Models/{}_{}_{}_{}_BYOL.pth".format(batch_size, i, "resnet50", "avgpool"))
        #save optimiser state
        torch.save(opt.state_dict(), "/home-mscluster/jknopfmacher/Research/BYOL/Optim/opt_{}_{}_{}_{}_BYOL.pth".format(batch_size, i, "resnet50", "avgpool"))

        print("Saved checkpoint")
        #save the loss including the epoch
        np.save("/home-mscluster/jknopfmacher/Research/BYOL/Losses/losses_{}_{}_{}_{}.npy".format(batch_size, i, "resnet50", "avgpool"), np.array(losses))

    

# save your improved network using the Batch size and Epochs in the filename
torch.save(learner.state_dict(), "/home-mscluster/jknopfmacher/Research/BYOL/checkpoint/{}_{}_{}_{}_BYOL.pth".format(batch_size, epochs, "resnet50", "avgpool"))
torch.save(resnet.state_dict(), "/home-mscluster/jknopfmacher/Research/BYOL/Models/{}_{}_{}_{}_BYOL.pth".format(batch_size, epochs, "resnet50", "avgpool"))
        #save optimiser state so that it can be used as the optimiser for encoder of a segmentation network

torch.save(opt.state_dict(), "/home-mscluster/jknopfmacher/Research/BYOL/Optim/opt_{}_{}_{}_{}_BYOL.pth".format(batch_size, epochs, "resnet50", "avgpool"))
#save the loss including the epoch
np.save("/home-mscluster/jknopfmacher/Research/BYOL/Losses/losses_{}_{}_{}_{}.npy".format(batch_size, epochs, "resnet50", "avgpool"), np.array(losses))
