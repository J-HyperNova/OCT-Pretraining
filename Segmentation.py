import segmentation_models_pytorch as smp
import os
import numpy as np
from torch import nn
from tqdm import tqdm
import matplotlib.pyplot as plt
from skimage import img_as_float
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix as cm
import segmentation_models_pytorch.utils as utils
from torch.utils.data import Dataset as Dataset
from PIL import Image 
import torch 
import matplotlib.colors as mcolors
import wandb 
import warnings
warnings.filterwarnings("ignore")
import argparse as parser
print("🔵Imported all libraries")
'''📝
START OF ARGUMENT PARSING
------------------------------------------------------------------------------------------------------------------------
'''
#parse the arguments
parser = parser.ArgumentParser(description='Process some integers.')
parser.add_argument('--encoder_weights', type=str, default="None", help='pretrained weights to use')
#batch size
parser.add_argument('--batch_size', type=int, default=2, help='batch size')

#learning rate
parser.add_argument('--learning_rate', type=float, default=0.0007509096722414667, help='learning rate')
#weight decay
parser.add_argument('--weight_decay', type=float, default=0, help='weight decay')
#epochs
parser.add_argument('--epochs', type=int, default=50, help='number of epochs')
#patience
parser.add_argument('--patience', type=int, default=15, help='patience for early stopping')


'''DATASET FUNCTIONS
'''
def encodeLabels(y):
    n,h,w=y.shape
    y=y.reshape(-1,1)
    labelencoder=LabelEncoder()
    y=labelencoder.fit_transform(y)
    y=y.reshape(n,h,w)
    return y
def get_data(folder_name, dataset='/home-mscluster/jknopfmacher/Research/Datasets/DukeData/'):
  # check if the folder name is valid
  if folder_name not in ["test", "train", "val"]:
    print("Invalid folder name. Please enter test, train or val.")
    return None, None
  
  # get the paths of the images and masks folders
  images_path = os.path.join(dataset+folder_name, "images")
  masks_path = os.path.join(dataset+folder_name, "masks")

  # initialize empty lists to store the numpy arrays
  images_list = []
  masks_list = []

  # loop through the files in the images folder
  for file in tqdm(os.listdir(images_path), desc="Loading images", colour="green"):
    # check if the file is a .npy file
    if file.endswith(".npy"):
      # load the numpy array from the file
      image = np.load(os.path.join(images_path, file))
      #resize images to standardize
      #print image shape
     
      image=Image.fromarray(image)
      image=image.convert('RGB')
      image=image.resize((128, 128))
      image=img_as_float(image)

      #get shape
      #print image shape
      #print(image.shape)
      # append it to the images list
      #swap axes to make it (channels, x, y)
      image=np.swapaxes(image, 0, 2)
      image=np.swapaxes(image, 1, 2)
      
      images_list.append(image)
  
  # loop through the files in the masks folder
  for file in tqdm(os.listdir(masks_path), desc="Loading masks", colour="blue"):
    # check if the file is a .npy file
    if file.endswith(".npy"):
      # load the numpy array from the file
      mask = np.load(os.path.join(masks_path, file))
      # append it to the masks list
      mask=Image.fromarray(mask)
      mask=mask.convert('L')
      mask=mask.resize((128, 128))
      mask=img_as_float(mask)
      masks_list.append(mask)
  images_list=np.stack(images_list)
  masks_list=np.stack(masks_list)
 
  # return the lists of numpy arrays

  masks_list=encodeLabels(masks_list)
  #masks_list=np.expand_dims(masks_list, axis=1)
  return images_list, masks_list
def makeOneHot(y):
    y=torch.nn.functional.one_hot(torch.from_numpy(y), num_classes=11)
    y=np.swapaxes(y, 1, 3)  
    y=np.swapaxes(y, 2, 3)
    return y
class MyDataset(Dataset):
    #""Custom Dataset for loading images and labels\"""
    def __init__(self, images, masks):
        self.images = torch.tensor(images, dtype=torch.float)
        self.masks = torch.tensor(masks, dtype=torch.float)
    def __getitem__(self, index):
        image = self.images[index]
        mask = self.masks[index]
        return image, mask
    def __len__(self):
        return len(self.images)
'''

Start of Model functions
'''

class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0, eps=1e-6):
        super(FocalLoss, self).__init__()
        self.alpha = alpha # weight for positive class
        self.gamma = gamma # focusing parameter
        self.eps = eps # small constant to avoid numerical instability

    def forward(self, inputs, targets):
        inputs = torch.sigmoid(inputs) # apply sigmoid activation
        inputs = inputs.view(-1) # flatten the inputs
        targets = targets.view(-1) # flatten the targets
        pt = torch.where(targets == 1, inputs, 1 - inputs) # compute probability of true class
        at = torch.where(targets == 1, self.alpha, 1 - self.alpha) # compute weight of true class
        loss = -at * (1 - pt) ** self.gamma * torch.log(pt + self.eps) # compute focal loss
        return loss.mean() # return mean loss over batch
    
class DiceFocalLoss(nn.Module):
   def __init__(self, alpha=0.5, beta=0.5):
        super(DiceFocalLoss, self).__init__()
        self.alpha = alpha # weight for dice loss
        self.beta = beta # weight for focal loss
        self.dice_loss = smp.utils.losses.DiceLoss()
        self.__name__ = 'DiceFocalLoss'
        self.focal_loss = FocalLoss()

   def forward(self, inputs, targets):
        dice = self.dice_loss(inputs, targets)
        focal = self.focal_loss(inputs, targets)
        return self.alpha * dice+ self.beta * focal
def processMask(mask):
    mask=mask.cpu()
    mask=mask.detach().numpy()
    mask=np.argmax(mask, axis=1)
    mask=mask[0]
    #mask shape=(128, 128)
    return mask
def dice_score(y_true, y_pred, smooth=1):
  # Flatten the arrays
  y_true = y_true.flatten()
  y_pred = y_pred.flatten()
  # Calculate the intersection and the sum of the arrays
  intersection = np.sum(y_true * y_pred)
  union = np.sum(y_true) + np.sum(y_pred)
  # Calculate the Dice score using the formula: 2 * intersection / (union + smooth)
  dice = (2 * intersection + smooth) / (union + smooth)
  # Return the Dice score
  return dice
def per_layer_dice_score(y_true, y_pred, num_classes):
  # Initialize an empty list to store the per layer Dice scores
  per_layer_dice = []
  # Loop through each class
  for i in range(num_classes):
    # Extract the binary masks for the current class from the true and predicted labels
    y_true_i = (y_true == i).astype(np.float32)
    y_pred_i = (y_pred == i).astype(np.float32)
    # Calculate the Dice score for the current class using the dice_score function
    dice_i = dice_score(y_true_i, y_pred_i)
    # Append the Dice score to the per layer Dice list
    per_layer_dice.append(dice_i)
  # Return the per layer Dice list
  return per_layer_dice

'''👀
Start of reading in the data

'''
x_train, y_train = get_data("train")
#get shape
print("🔵Read in training data")
#want to turn the list into a numpy array of numpy arrays with shape( num images, x,y)
# get the validation data
x_val, y_val = get_data("val")
print("🔵Read in validation data")
# get the testing data
#classes are [ 0  1  2  3  4  5  6  7  8  9 10]
x_test, y_test = get_data("test")
print("🔵Read in testing data")
#xtrain shape =(66, 3, 128, 128)
#y_train shape=(66, 128, 128)

#convert to one hot
y_train_cat=makeOneHot(y_train)
y_val_cat=makeOneHot(y_val)
y_test_cat=makeOneHot(y_test)
#y_train_cat shape=[66, 11, 128, 128]

'''❕❕
START OF MODEL CODE
------------------------------------------------------------------------------------------------------------------------
'''
args = parser.parse_args()
BACKBONE = 'resnet50'
encoder_weights =args.encoder_weights
weights=None #DO NOT CHANGE THIS

if(encoder_weights=='imagenet'):
    weights='imagenet'
    print("🔵Loaded imagenet weights")
model=smp.Unet(BACKBONE, encoder_weights=weights, classes=11,   activation='softmax')
print("🔵Created model")
if( encoder_weights=="BYOL"):  
    pretrained_weights=torch.load('/home-mscluster/jknopfmacher/Research/Segmentation/Pretrained/BYOL.pth')
    model.encoder.load_state_dict(pretrained_weights)
    print("🔵Loaded BYOL weights")
if (   encoder_weights=="SimCLR"):
    pretrained_weights=torch.load('/home-mscluster/jknopfmacher/Research/Segmentation/Pretrained/simclr.pth')
    model.encoder.load_state_dict(pretrained_weights)
    print("🔵Loaded SimCLR weights")
if (encoder_weights=="PLC"):
    pretrained_weights=torch.load('/home-mscluster/jknopfmacher/Research/Segmentation/Pretrained/PLC.pth')
    model.encoder.load_state_dict(pretrained_weights)
    print("🔵Loaded PLC weights")
if (encoder_weights=="DINO"):
    pretrained_weights=torch.load('/home-mscluster/jknopfmacher/Research/Segmentation/Pretrained/DINO.pth')
    model.encoder.load_state_dict(pretrained_weights)
    print("🔵Loaded DINO weights")
if (encoder_weights=="Classification"):
    pretrained_weights=torch.load('/home-mscluster/jknopfmacher/Research/Segmentation/Pretrained/Class.pth')
    model.encoder.load_state_dict(pretrained_weights)
    print("🔵Loaded Classification weights")
if (encoder_weights=="SimCLRv2"):
    pretrained_weights=torch.load('/home-mscluster/jknopfmacher/Research/Segmentation/Pretrained/simclrv2.pth')
    model.encoder.load_state_dict(pretrained_weights)
    print("🔵Loaded SimCLRv2 weights")



lr=args.learning_rate
epochs = args.epochs
batch_size = args.batch_size
weight_decay=args.weight_decay

patience=args.patience
metrics = [
    smp.utils.metrics.IoU(threshold=0.5),
    smp.utils.metrics.Fscore(threshold=0.5)
]
focal_weight=1
dice_weight=1
loss = DiceFocalLoss(alpha=dice_weight, beta=focal_weight)
optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
model=nn.DataParallel(model)
trainer= utils.train.TrainEpoch(
    model,
    loss=loss,
    metrics=metrics,
    optimizer=optimizer,
    device='cuda',
    verbose=False,
)
validation_epoch = utils.train.ValidEpoch(
    model,
    loss=loss,
    metrics=metrics,
    device='cuda',
    verbose=False,
)

#init the wandb run, include all the hyperparamaters here
#need to make sure to define all them before the init so that they are kept in sync with what is recorded in wandb
run= wandb.init(
    # set the wandb project where this run will be logged 
    
    project="AllSweep",
    name=encoder_weights+str(np.random.randint(1000000)),
    # track hyperparameters and run metadataOCT Segementation
    config={
   

    "backbone": BACKBONE,
    "pretrained_weights": encoder_weights,
    "epochs": epochs,
    "batch_size": batch_size,
    "learning_rate": lr,
    "weight_decay": weight_decay,
    "early_stopping": not patience==epochs,
    "patience": patience,
    
    }
)


train_dataset = MyDataset(x_train, y_train_cat)
valid_dataset = MyDataset(x_val, y_val_cat)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size, shuffle=True, num_workers=8)
valid_loader = torch.utils.data.DataLoader(valid_dataset, batch_size, shuffle=True, num_workers=8)
#implement early stopping

best_val=0

curr_patience=0
best_model=None
#train the model

for i in tqdm(range(epochs), colour="purple"):
    #train with both training and validation data
   # print('\nEpoch:{} out of {}'.format(i+1 , epochs))
    train_logs = trainer.run(train_loader)
    valid_logs = validation_epoch.run(valid_loader)
    #log the metrics to wandb
    #print a list of the keys in the logs
    #print(train_logs.keys())
    wandb.log({
        "train_loss": train_logs['DiceFocalLoss'],
        "train_iou": train_logs['iou_score'],
        "train_f1": train_logs['fscore'],
        "val_loss": valid_logs['DiceFocalLoss'],
        "val_iou": valid_logs['iou_score'],
        "val_f1": valid_logs['fscore'],
    })
    #check if the early stopping criteria is met
    if not patience==epochs:

        if(valid_logs['iou_score']+valid_logs['fscore']>best_val):
            best_val=valid_logs['iou_score']+valid_logs['fscore']
            curr_patience=0
            best_model=model
        else:
            curr_patience+=1
        if(curr_patience==patience):
            print("🛑Early stop")
            model=best_model
            break



save_dir='/home-mscluster/jknopfmacher/Research/Segmentation/Models/'
if encoder_weights=="None":
    save_dir+='None/'
elif encoder_weights=="imagenet":
    save_dir+='Imagenet/'
elif encoder_weights=="BYOL" :
    save_dir+='BYOL/'
elif encoder_weights=="SimCLR":
    save_dir+='SIMCLR/'
elif encoder_weights=="PLC":
    save_dir+='PLC/'
elif encoder_weights=="DINO":
    save_dir+='DINO/'
elif encoder_weights=="Classification":
    save_dir+='Classification/'
elif encoder_weights=="SimCLRv2":
    save_dir+='SimCLRv2/'

#save the model
torch.save(model.state_dict(), save_dir+'model_{}_{}_{}_{}_{}_{}.pth'.format( epochs, batch_size, lr, weight_decay, early_stopping, patience))

#save on wandb
#torch.save(model.state_dict(), wandb.run.dir+'model.pth')
# wandb.save( wandb.run.dir+'model.pth')

model.eval()
'''📈📉📊
START OF TESTING CODE
------------------------------------------------------------------------------------------------------------------------
'''
test_dataset = MyDataset(x_test, y_test_cat)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=4)
print("🔵Getting sample output")
#test the model

image, mask = next(iter(test_loader))

prediction = model(image)
prediction=processMask(prediction)
#prediction shape=(128, 128)

mask=processMask(mask)
#mask shape=(128, 128)

#set fixed cmap values for a specific class
cmap = mcolors.ListedColormap(colors = ['navy', 'blue', 'cyan', 'lime', 'yellow', 'orange', 'black', 'darkred', "red", "white", "grey"]
)
bounds=[0,1,2,3,4,5,6,7,8,9,10]
norm = mcolors.BoundaryNorm(bounds, cmap.N)

#plot the image, mask, and prediction
plt.subplot(1,3,1)
#turn axes off
plt.axis('off')
plt.imshow(image[0].permute(1,2,0)  )
plt.subplot(1,3,2)
plt.axis('off')
plt.imshow(mask,  cmap=cmap, norm=norm)
plt.subplot(1,3,3)
#get the count of each class in the prediction
plt.axis('off')

plt.imshow(prediction, cmap=cmap, norm=norm)

#save the plot using the same name as the model
#plt.savefig('/home-mscluster/jknopfmacher/Research/Segmentation/Results/test_image_1_{}_{}_{}_{}.png'.format(encoder_weights, epochs, batch_size, lr))
#log the plot to wandb
wandb.log({"test_image": plt})
#log the test metrics to wandb
#compute the test metrics using the same metrics as the training
print("🔵Computing test metrics")
iou_sum=0
f1_sum=0
confusion_matrix=np.zeros((11,11))
per_layer_dice=np.zeros((10))
with torch.no_grad():
    for image, mask in test_loader:
        prediction = model(image)
        mask=mask.cuda()
        iou_sum+=metrics[0].__call__(prediction, mask)
        f1_sum+=metrics[1].__call__(prediction, mask)
        prediction=processMask(prediction)
        mask=processMask(mask)
        #get the per layer dice score
        per_layer_dice+=per_layer_dice_score(mask, prediction, 10)
        #update the confusion matrix
        mask=mask.flatten()
        prediction=prediction.flatten()
        #create own confusion matrix
        this_confusion_matrix=np.zeros((11,11))
        for i in range(len(mask)):
            this_confusion_matrix[mask[i]][prediction[i]]+=1
        confusion_matrix+=this_confusion_matrix



per_layer_dice=per_layer_dice/len(test_loader)
iou=iou_sum/len(test_loader)
f1=f1_sum/len(test_loader)
#plot the confusion matrix
plt.figure(figsize=(10,10))
plt.imshow(confusion_matrix, cmap='jet', interpolation='nearest')
#set axis so that the labels are the numbers from 0 to 10
plt.xticks(np.arange(11))
plt.yticks(np.arange(11))
#save the confusion matrix
#plt.savefig('/home-mscluster/jknopfmacher/Research/Segmentation/Results/confusion_matrix_{}_{}_{}_{}.png'.format(encoder_weights, epochs, batch_size, lr))
print(per_layer_dice)
print("test_iou: ", iou)
print("test_f1: ", f1)
print("val_iou: ", valid_logs['iou_score'])
print("val_f1: ", valid_logs['fscore'])
per_layer_dice=np.round(per_layer_dice, 3)
filename= ""
np.savetxt(filename+"_per_layer_dice_{}_{}_{}_{}.txt".format(encoder_weights, epochs, batch_size, lr), per_layer_dice, delimiter="&", fmt='%.3f')
wandb.log({"test_iou": iou,
           "test_f1": f1,
              "test_per_layer_dice": per_layer_dice,
           "confusion_matrix": [wandb.Image(plt, caption="Confusion Matrix")]})

#finish the wandb run


run.finish()