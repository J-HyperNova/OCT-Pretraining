# OCT Segmentation Pretraining

## Abtract

Diagnosis of ocular diseases is labour-intensive. Automated segmentation of Optical Coherence Tomography (OCT) scans can help diagnose these diseases by showing the structure and thickness of the retinal layers. However, most segmentation methods rely on costly labelled data. This paper examines the effect of different image pretraining methods on OCT image segmentation tasks. We pretrain the encoder of a U-Net using supervised and self-supervised methods on a large OCT dataset with classification labels. The methods compared include BYOL, PLC, SimCLR, SimCLRv2, DINO, and classification pretraining. Then, we fine-tune the whole U-Net on a smaller OCT dataset with segmentation masks for the downstream segmentation task. We compare the dice scores per layer between the pretrained and baseline models. Most of the pretraining methods provide minor improvements compared to the baseline on a dataset that is very homogeneous, with the best results from PLC pretraining and from pretraining on the labelled classification problem via transfer learning.

[View the full paper](https://github.com/J-HyperNova/OCT-Pretraining/blob/main/paper.pdf)

## Structure

### Directories

- `Envs/` contains all the conda environments files, required for running the code.
- `Pretraining/` contains the code for the pretraining experiments.

### Files
- `Segmentation.py` contains the code for the segmentation experiments, requires the pretraining experiments to be run first and models placed in the directories expected by the code. This uses the SMTorch environment.
- `Pretraining/myBYOL.py` contains the code for BYOL pretraining. Uses the BYOL environment.
- `Pretraining/myPLC.py` contains the code for PLC pretraining. Uses the PLC environment.
- `Pretraining/mySimCLR.py` contains the code for SimCLR pretraining. Uses the SimCLR environment.
- `Pretraining/myClass.py` contains the code for the classification pretraining. Uses the SMTorch environment as no major dependencies are required.
- `Pretraining/myDino.py` contains the code for DINO pretraining. Uses the pyssl environment.
-  `Pretraining/mySimCLRv2.py` contains the code for SimCLRv2 pretraining. Uses the pyssl environment.

## Environments
All of the environments can be installed in conda on python 3.8 with the enviroments specific for each python file to avoid any dependency issues.

NOTE: for the pySSL environment, the pyssl package needs to be downloaded from [here](https://github.com/giakou4/pyssl) and added to the path in the code as it is not available on pip.

## Datasets
- To pretrain the encoders, you need to download the classification `OCT2017.tar.gz` dataset from [this link](https://data.mendeley.com/datasets/rscbjbr9sj/2). 
- To train the segmentation model, you need to use the segmentation dataset. You can find the instructions on how to download and unzip the dataset in the YNet repository [here](https://github.com/azadef/ynet/tree/master#datasets-downloading-and-preproccesing). You only need the 2015_BOE_Chiu dataset for the segmentation experiments.


## Notes
- The code has wandb logging enabled, so you need to have a wandb account and be logged in to run the code. You can disable wandb logging by commenting out the wandb.init() and wandb.log() lines in the code.
- The code has hard coded paths, so you need to change the paths to the correct directories in the code, so that the data can be loaded and the models can be saved.
- The code is written to run on a GPU, so you need to change the device to CPU if you want to run it on a CPU.

## Running the segmentation experiments
The segmentation code takes in the following arguments:

- `--encoder_weights`: This argument specifies the pretrained weights to use for the encoder. The default value is "None", which means no pretrained weights are used. The possible values are "None", "BYOL", "SimCLR", "PLC", "imagenet", "DINO", "Classification", "SimCLRv2". Currently the paths are hard coded, so you need to change the paths in the code to the correct directories of the pretrained models.
- `--batch_size`: This argument determines the batch size for the training and evaluation. The default value is 2, which means two samples are processed at a time. You can change this value to any positive integer.
- `--learning_rate`: This argument sets the learning rate for the optimizer. The learning rate is a hyperparameter that affects how fast the model learns from the data. The default value is 0.0007509096722414667, which is a small number. You can adjust this value to any positive float.
- `--weight_decay`: This argument sets the weight decay for the optimizer. Weight decay is a regularization technique that prevents the model from overfitting by adding a penalty to the weights. The default value is 0, which means no weight decay is applied. You can increase this value to any non-negative float to apply weight decay.
- `--epochs`: This argument specifies the number of epochs for the training. An epoch is a complete pass through the dataset. The default value is 50, which means the model will train for 50 epochs. You can change this value to any positive integer.
- `--patience`: This argument determines the patience for early stopping. Patience is the number of epochs to wait before stopping the training if the validation loss does not improve. The default value is 15, which means the model will stop training if the validation loss does not improve for 15 consecutive epochs. You can change this value to any positive integer. If you want to disable early stopping, you can set this value to the same value as the number of epochs.

## Acknowledgements
We made extensive use of the following repositories:
- [BYOL](https://github.com/lucidrains/byol-pytorch)
- [PLC](https://github.com/lucidrains/pixel-level-contrastive-learning)
- [SimCLR](https://github.com/Spijkervet/SimCLR)
- [PySSL](https://github.com/giakou4/pyssl)
- [Segmentation Models](https://github.com/qubvel/segmentation_models.pytorch)
- [YNet](https://github.com/azadef/ynet)
