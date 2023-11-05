
import torch
import sys
sys.path.append('/home-mscluster/jknopfmacher/Research/pyssl') #location of the pyssl library
from builders.simclrv2 import SimCLRv2
import warnings
warnings.filterwarnings("ignore")



from torchvision import models, transforms, datasets
import wandb 
import os
import numpy as np
import skimage.io as io
from skimage.transform import resize
resnet = models.resnet50(pretrained=False)

from tqdm import tqdm
import torch.nn as nn
feature_size = resnet.fc.in_features
resnet.fc = torch.nn.Identity()

projection_dim = 128
similarity_temperature = 0.3
kwargs = {
    'image_size': 128,
    'mean': [0.485, 0.456, 0.406],
    'std': [0.229, 0.224, 0.225]
}
model= SimCLRv2(resnet, feature_size, projection_dim=projection_dim, **kwargs)

print ("😀Created model")

#model=nn.DataParallel(model)
model=model.cuda()
epochs=50
batch_size=256
learning_rate=0.075
opt = torch.optim.Adam(model.parameters(), lr=learning_rate)
run = wandb.init(   project="mySIMCLRv2",
                 config={"epochs": epochs, 
                         "batch_size": batch_size, 
                         "learning_rate": learning_rate,
                    
                           "projection_dim": projection_dim})

def createDataloader():
    # create a dataloader
    transform=transforms.Compose([
        transforms.Resize((128,128),  antialias=None),
        transforms.ToTensor(),
        
    ])
    dataset = datasets.ImageFolder("/home-mscluster/jknopfmacher/Research/Datasets/CaliOCT/train", transform=transform)
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
    print("😀Created dataloader")
    for images, labels in loader: # iterate over batches
        print(images.dtype) # print the data type of the tensor elements
        break # break after one batch
    return loader

def train():
    loader = createDataloader()
    model.train()
    losses = []
    for epoch in tqdm(range(epochs), colour="green"):
        this_epoch_loss = 0
        
        for images, _ in tqdm(loader):
            images=images.cuda()
            loss = model(images)
            #log loss
            this_epoch_loss += loss.item()
            opt.zero_grad()
            loss.backward()
            opt.step()
           
        #make sure to log the loss only once per epoch
        
        wandb.log({"loss": this_epoch_loss/len(loader)})
        losses.append(this_epoch_loss/len(loader))
        #every 5 epochs, save the model checkpoint with the name being the config
        if epoch % 5 == 0:
            torch.save(model.state_dict(), "/home-mscluster/jknopfmacher/Research/SIMCLRv2/Model/v2_{}_{}_{}_{}.pth".format(batch_size, epoch, projection_dim, learning_rate))
            np.save("/home-mscluster/jknopfmacher/Research/SIMCLRv2/loss/losses_{}_{}_{}_{}.npy".format(batch_size, epoch, projection_dim, learning_rate), np.array(losses))
            #save optimiser state
            torch.save(opt.state_dict(), "/home-mscluster/jknopfmacher/Research/SIMCLRv2/Optim/opt_{}_{}_{}_{}.pth".format(batch_size, epoch, projection_dim, learning_rate))
            #save resnet state
            torch.save(resnet.state_dict(), "/home-mscluster/jknopfmacher/Research/SIMCLRv2/Resnet/resnet_{}_{}_{}_{}.pth".format(batch_size, epoch, projection_dim, learning_rate))
    return losses
losses = train()
#now save 
#save the model
torch.save(model.state_dict(), "/home-mscluster/jknopfmacher/Research/SIMCLRv2/Model/v2_{}_{}_{}_{}.pth".format(batch_size, epochs, projection_dim, learning_rate))
#save the losses
np.save("/home-mscluster/jknopfmacher/Research/SIMCLRv2/loss/losses_{}_{}_{}_{}.npy".format(batch_size, epochs, projection_dim, learning_rate), np.array(losses))
#save optimiser state
torch.save(opt.state_dict(), "/home-mscluster/jknopfmacher/Research/SIMCLRv2/Optim/opt_{}_{}_{}_{}.pth".format(batch_size, epochs, projection_dim, learning_rate))
#save resnet state
torch.save(resnet.state_dict(), "/home-mscluster/jknopfmacher/Research/SIMCLRv2/Resnet/resnet_{}_{}_{}_{}.pth".format(batch_size, epochs, projection_dim, learning_rate))

    

