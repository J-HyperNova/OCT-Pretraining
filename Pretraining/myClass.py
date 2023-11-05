import torch
from torchvision import models, transforms, datasets

import os
import numpy as np
import skimage.io as io
from skimage.transform import resize
resnet = models.resnet50(pretrained=True) #use resnet50 pretrained on imagenet
from tqdm import tqdm
import torch.nn as nn
import matplotlib.pyplot as plt
#read in training data as dataloader
import random
num_images=10000
print("done imports")
def read_data(type):
    transform=transforms.Compose([
        transforms.Resize((128,128),  antialias=None),
        transforms.ToTensor(),
        
    ])
    dataset= datasets.ImageFolder("/home-mscluster/jknopfmacher/Research/Datasets/FullCaliOCT/Images/"+type, transform=transform)
    if type=="train":
        indices = list(range(len(dataset)))
        random.shuffle(indices) #shuffle the indices
        indices = indices[:num_images] 
        dataset = torch.utils.data.Subset(dataset, indices)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=True)
    return dataloader
#get training data
train_dataloader = read_data("train")
print("Read in traindata")
val_dataloader = read_data("val")
print("Read in valdata")
test_dataloader = read_data("test")
print("Read in testdata")

#set up model
#use the resnet to classify the images with 4 classes
resnet.fc = nn.Linear(resnet.fc.in_features, 4)
#use gpu
resnet=nn.DataParallel(resnet)
resnet=resnet.cuda()
epochs=100
batch_size=32
learning_rate=0.0001
#use cross entropy loss
criterion = nn.CrossEntropyLoss()
#use adam optimizer
optimizer = torch.optim.Adam(resnet.parameters(), lr=learning_rate)
#use learning rate scheduler
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)
print("Set up model with {epochs} epochs, {batch_size} batch size, and {learning_rate} learning rate with {img} train images".format(epochs=epochs, batch_size=batch_size, learning_rate=learning_rate, img=num_images))
#set up wandb
# wandb.init(project="OCT_Classification",
#            config={
#                 "learning_rate": learning_rate,
#                 "epochs": epochs,
#                 "batch_size": batch_size,
#                 "architecture": "resnet50",
#                 "dataset": "CaliOCT",
#               }
#            )
#train the model
all_train_loss=[]
all_val_loss=[]
val_acc=[]
for epoch in tqdm(range(epochs)):
    #train
    resnet.train()
    train_loss=0
    for batch_idx, (data, target) in enumerate(train_dataloader):
        data, target = data.cuda(), target.cuda()
        optimizer.zero_grad()
        output = resnet(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
    train_loss /= len(train_dataloader)
    #validate
    resnet.eval()
    correct = 0
    total = 0
    val_loss=0
   
    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(val_dataloader):
            data, target = data.cuda(), target.cuda()
            output = resnet(data)
            _, predicted = torch.max(output.data, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()
            loss = criterion(output, target)
            val_loss += loss.item()
    all_train_loss.append(train_loss)
    all_val_loss.append(val_loss/len(val_dataloader))
    val_acc.append(correct/total)
    
    #wandb.log({"Train Loss": train_loss, "Validation Loss": val_loss/len(val_dataloader), "Validation Accuracy": correct/total})
    scheduler.step()
    #save checkpoints every 5 epochs
    if epoch % 5 == 0:
        #print the loss
        print("Epoch: {} Train Loss: {} Validation Loss: {} Validation Accuracy: {}".format(epoch, train_loss, val_loss/len(val_dataloader), correct/total))
        #plot the loss
        plt.figure()
        plt.plot(all_train_loss, label="Train Loss")
        plt.plot(all_val_loss, label="Validation Loss")
        plt.plot(val_acc, label="Validation Accuracy")
        plt.legend()
        plt.savefig("/home-mscluster/jknopfmacher/Research/Classification/graphs/{}_{}_{}_{}.png".format(epoch, batch_size, learning_rate, "class"))

        torch.save(resnet.state_dict(), "/home-mscluster/jknopfmacher/Research/Classification/Models/Model_{}_{}_{}_{}.pt".format(epoch, batch_size, learning_rate, "class"))
       # torch.save(resnet.encoder.state_dict(), "/home-mscluster/jknopfmacher/Research/Classification/Models/Enc_{}_{}_{}_{}.pt".format(epoch, batch_size, learning_rate, "class"))
        #optimizer
        torch.save(optimizer.state_dict(), "/home-mscluster/jknopfmacher/Research/Classification/Models/{}_{}_{}_{}_optimizer.pt".format(epoch, batch_size, learning_rate, "class"))
        #scheduler
        torch.save(scheduler.state_dict(), "/home-mscluster/jknopfmacher/Research/Classification/Models/{}_{}_{}_{}_scheduler.pt".format(epoch, batch_size, learning_rate, "class"))

#test the model
resnet.eval()
correct = 0
total = 0
confusion_matrix=np.zeros((4,4))
with torch.no_grad():
    for batch_idx, (data, target) in enumerate(test_dataloader):
        data, target = data.cuda(), target.cuda()
        output = resnet(data)
        _, predicted = torch.max(output.data, 1)
        total += target.size(0)
        correct += (predicted == target).sum().item()
        for i in range(len(target)):
            confusion_matrix[target[i]][predicted[i]]+=1
print(confusion_matrix)
#log the confusion matrix
plt.figure(figsize=(10,10))
plt.imshow(confusion_matrix, cmap='jet', interpolation='nearest')
#set axis so that the labels are the numbers from 0 to 10
plt.xticks(np.arange(4))
plt.yticks(np.arange(4))
#wandb.log({ "test_accuracy": correct/total, "confusion_matrix": [wandb.Image(plt, caption="Confusion Matrix")]})
print ("Test Accuracy: {}".format(correct/total))
#get f1 score
f1_score=[]
for i in range(4):
    precision=confusion_matrix[i][i]/np.sum(confusion_matrix[i])
    recall=confusion_matrix[i][i]/np.sum(confusion_matrix[:,i])
    f1_score.append(2*precision*recall/(precision+recall))
print("F1 Score: {}".format(np.mean(f1_score)))

#plot the loss
plt.figure()
plt.plot(all_train_loss, label="Train Loss")
plt.plot(all_val_loss, label="Validation Loss")
plt.plot(val_acc, label="Validation Accuracy")
plt.legend()
plt.savefig("/home-mscluster/jknopfmacher/Research/Classification/graphs/{}_{}_{}_{}.png".format(epochs, batch_size, learning_rate, "class"))
#save the loss
np.save("/home-mscluster/jknopfmacher/Research/Classification/Models/{}_{}_{}_{}_loss.npy".format(epochs, batch_size, learning_rate, "class"), [all_train_loss, all_val_loss, val_acc])


resnet=resnet.module
resnet.fc=torch.nn.Linear(resnet.fc.in_features, 1000)

#now save the model
torch.save(resnet.state_dict(), '/home-mscluster/jknopfmacher/Research/Segmentation/Pretrained/Class.pth')