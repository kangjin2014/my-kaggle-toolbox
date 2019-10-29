import os, sys
import logging
import numpy as np
import pandas as pd
import torch
import random
from PIL import Image
from matplotlib import pyplot as plt
import copy
import torchvision
from torchvision import transforms, utils
from torch.utils.data import DataLoader
import random
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import time
import copy
import torchvision.models as models
from torch import Tensor
import subprocess
import cv2
import seaborn as sns

from torch.optim.lr_scheduler import ReduceLROnPlateau
import torch.backends.cudnn as cudnn

from albumentations import HorizontalFlip, GridDistortion, ShiftScaleRotate, Normalize, Resize, Compose, VerticalFlip, GaussNoise, Rotate
from albumentations.pytorch import ToTensor
from albumentations import CropNonEmptyMaskIfExists, OpticalDistortion, ElasticTransform
from torch.utils.data.sampler import WeightedRandomSampler

from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
import random
import warnings
warnings.filterwarnings("ignore")
seed = 10
random.seed(seed)
os.environ["PYTHONHASHSEED"] = str(seed)
np.random.seed(seed)
torch.cuda.manual_seed(seed)
torch.backends.cudnn.deterministic = True

# %% [code]
import subprocess, sys

# %% [code]
def install(package):
    subprocess.call([sys.executable, "-m", "pip", "install", package])    
install('../input/pretrainedmodels/pretrainedmodels-0.7.4/pretrainedmodels-0.7.4')
install('../input/segmentationpytorch/segmentation_models.pytorch-0.0.3')
install('albumentations')
import segmentation_models_pytorch as smp

# %% [code]
folder_input = '/kaggle/input/severstal-steel-defect-detection'
folder_train_input = '/kaggle/input/severstal-steel-defect-detection/train_images/'
folder_test_input = '/kaggle/input/severstal-steel-defect-detection/test_images/'
path_mask_train = '/kaggle/input/severstal-steel-defect-detection/train.csv'
folder = '../input/severstal-steel-defect-detection/'
paths_imgs_train = [folder_train_input + e for e in os.listdir(folder_train_input)]
paths_imgs_test = [folder_test_input + e for e in os.listdir(folder_test_input)]

df = pd.read_csv( folder + 'train.csv')
df['ImageId'], df['ClassId'] = zip(*df['ImageId_ClassId'].str.split('_'))
df['ClassId'] = df['ClassId'].astype(int)
df = df.pivot(index='ImageId',columns='ClassId',values='EncodedPixels')

df['defects'] = df.count(axis=1)
sns.barplot(df.defects.value_counts().index,df.defects.value_counts().values)

# %% [code]
def reform(df):
    '''
    add labels info to initial pivot table. so that pivot table is bigger
    initial defects distribution
        1    6239
        0    5902
        2     425
        3       2
    len = 12568
    return 12997
    '''
#     df = df.fillna('nan')

    # classes 1-4
    df_class_1 = df[df[1].isnull() == False]
    df_class_2 = df[df[2].isnull() == False]
    df_class_3 = df[df[3].isnull() == False]
    df_class_4 = df[df[4].isnull() == False]
    # negative
    df_class_0 = df[df['defects'] == 0]

    df_class_1['ClassId'] = 1
    df_class_2['ClassId'] = 2
    df_class_3['ClassId'] = 3
    df_class_4['ClassId'] = 4
    df_class_0['ClassId'] = 0

    df = pd.concat([df_class_1, df_class_2, df_class_3, df_class_4, df_class_0])
    return df

df = reform(df)

class_number = 3
df = df[df[class_number].isnull() == False]
df = pd.DataFrame(df[class_number])

# %% [code]
df

# %% [code]
def udf_transformer(target, phase):
    
    list_transforms = []
    
    if target in [1,3,4]:
        pc_min, pc_max = 150, 50000
    else:
        pc_min, pc_max = 5, 50000

    if phase == 'train':
        list_transforms.extend([CropNonEmptyMaskIfExists(height = 224, 
                                                     width = 1440, # modify this
#                                                      ignore_values = [pc_min,pc_max], # [min, max], 
                                                     p=0.8),
                            Resize(height = 256,
                                   width = 1600)
                           ]
                          ),
#         list_transforms.append(ElasticTransform(p=0.4)),
#         list_transforms.append(OpticalDistortion(p=0.4))
        list_transforms.append(HorizontalFlip(p=0.5))
        list_transforms.append(VerticalFlip(p=0.5))
        list_transforms.append(Rotate(limit=20, p=0.5))
        list_transforms.append(GaussNoise())
        list_transforms.append(GridDistortion(distort_limit=(-0.2, 0.2), p =0.6))
    list_transforms.append(Normalize([0.5], [0.5]))  # important step
    list_transforms.append(ToTensor()) # using albumentations transformers, must use ToTensor from albumentations too
    return Compose(list_transforms)

# %% [code]
def make_mask(df,row_id):
    '''Given a row index, return image_id and mask (256, 1600, 4) from the dataframe `df`'''
    fname = df.iloc[row_id].name
    
    # -------- here I only need 1 column
    labels = df.iloc[row_id][:1]
    masks = np.zeros((256, 1600, 1), dtype=np.float32)
    # -----------------------------------

    for idx, label in enumerate(labels.values):
        if label is not np.nan:
            label = label.split(" ")
            positions = map(int, label[0::2])
            length = map(int, label[1::2])
            mask = np.zeros(256 * 1600, dtype=np.uint8)
            for pos, le in zip(positions, length):
                mask[pos:(pos + le)] = 1
            masks[:, :, idx] = mask.reshape(256, 1600, order='F')
    return fname, masks

class SteelDataset():
    def __init__(self, df, image_folder, target, phase, crop_size = 400):
        self.df = df
        self.root = image_folder
        self.phase = phase
        self.fnames = self.df.index.tolist()
        self.crop_size = crop_size
        self.target = target

    def __getitem__(self, idx):
        def crop_image_box(loc_c):
            top, bottom = 0, 256
            left = crop_size * loc_c
            right = left + crop_size
            return (left, top, right, bottom)
        
        image_id, mask = make_mask(self.df, idx)
#         target = self.df['ClassId'][idx]
        image_path = os.path.join(self.root, image_id)
        image = Image.open(image_path).convert('L')
        image = np.array(image)
        
        self.transformer = udf_transformer(self.target, self.phase)
        augmented = self.transformer(image=image, mask=mask)
        image = augmented['image']
        mask = augmented['mask']
        
#         to make image and mask 3 dimension
        image = image.reshape(1, image.shape[0], image.shape[1])
        mask = mask.permute(0,3,2,1)[0]
        
        return image, mask, self.target

    def __len__(self):
        return len(self.fnames)
    
# SteelData tester
# value = SteelDataset(df, '../input/severstal-steel-defect-detection/train_images', 'train')[0]

# plt.imshow(value[0][0]), print (value[0].shape, value[1].shape)

# %% [code]
df_train, df_val = train_test_split(df, test_size=0.10, random_state=7)

# %% [code]
# a big copy

def predict(X, threshold):
    '''X is sigmoid output of the model'''
    X_p = np.copy(X)
    preds = (X_p > threshold).astype('uint8')
    return preds

class Meter:
    '''A meter to keep track of iou and dice scores throughout an epoch'''
    def __init__(self, phase, epoch):
        self.base_threshold = 0.5 # <<<<<<<<<<< here's the threshold
        self.base_dice_scores = []
        self.dice_neg_scores = []
        self.dice_pos_scores = []
        self.iou_scores = []

    def update(self, targets, outputs):
        probs = torch.sigmoid(outputs)
        dice, dice_neg, dice_pos, _, _ = metric(probs, targets, self.base_threshold)
        self.base_dice_scores.append(dice)
        self.dice_pos_scores.append(dice_pos)
        self.dice_neg_scores.append(dice_neg)
        # critical here. predict function, take self.base_threshold
        preds = predict(probs, self.base_threshold)
        iou = compute_iou_batch(preds, targets, classes=[1])
        self.iou_scores.append(iou)

    def get_metrics(self):
        dice = np.mean(self.base_dice_scores)
        dice_neg = np.mean(self.dice_neg_scores)
        dice_pos = np.mean(self.dice_pos_scores)
        dices = [dice, dice_neg, dice_pos]
        iou = np.nanmean(self.iou_scores)
        return dices, iou

def epoch_log(phase, epoch, epoch_loss, meter, start):
    '''logging the metrics at the end of an epoch'''
    dices, iou = meter.get_metrics()
    dice, dice_neg, dice_pos = dices
    print("Loss: %0.4f | IoU: %0.4f | dice: %0.4f | dice_neg: %0.4f | dice_pos: %0.4f" % (epoch_loss, iou, dice, dice_neg, dice_pos))
    return dice, iou

def compute_ious(pred, label, classes, ignore_index=255, only_present=True):
    '''computes iou for one ground truth mask and predicted mask'''
    pred[label == ignore_index] = 0
    ious = []
    for c in classes:
        label_c = label == c
        if only_present and np.sum(label_c) == 0:
            ious.append(np.nan)
            continue
        pred_c = pred == c
        intersection = np.logical_and(pred_c, label_c).sum()
        union = np.logical_or(pred_c, label_c).sum()
        if union != 0:
            ious.append(intersection / union)
    return ious if ious else [1]

def compute_iou_batch(outputs, labels, classes=None):
    '''computes mean iou for a batch of ground truth masks and predicted masks'''
    ious = []
    preds = np.copy(outputs) # copy is imp
    labels = np.array(labels) # tensor to np
    for pred, label in zip(preds, labels):
        ious.append(np.nanmean(compute_ious(pred, label, classes)))
    iou = np.nanmean(ious)
    return iou

# %% [code]
def metric(probability, truth, threshold=0.5, reduction='none'):
    '''Calculates dice of positive and negative images seperately'''
    '''probability and truth must be torch tensors'''
    batch_size = len(truth)
    with torch.no_grad():
        probability = probability.view(batch_size, -1)
        truth = truth.view(batch_size, -1)
        assert(probability.shape == truth.shape)

        p = (probability > threshold).float()
        t = (truth > 0).float()
        # =========
#         print ("probability, truth, p, t", probability.shape, truth.shape, p.shape, t.shape)
        # =========

        t_sum = t.sum(-1)
        p_sum = p.sum(-1)
        neg_index = torch.nonzero(t_sum == 0)
        pos_index = torch.nonzero(t_sum >= 1)
        
        dice_neg = (p_sum == 0).float()
        dice_pos = 2 * (p*t).sum(-1)/((p+t).sum(-1))
        
        dice_neg = dice_neg[neg_index]
        dice_pos = dice_pos[pos_index]
        dice = torch.cat([dice_pos, dice_neg])
        
        dice_neg = np.nan_to_num(dice_neg.mean().item(), 0)
        dice_pos = np.nan_to_num(dice_pos.mean().item(), 0)
                
        dice = dice.mean().item()

        num_neg = len(neg_index)
        num_pos = len(pos_index)
    
    return dice, dice_neg, dice_pos, num_neg, num_pos

# %% [code]
class Trainer(object):
    '''This class takes care of training and validation of our model'''
    def __init__(self, model, df_train, df_val):
        self.image_folder = '../input/severstal-steel-defect-detection/train_images'
        self.num_workers = 6
        self.batch_size = 8
        self.accumulation_steps = 96 // self.batch_size
        self.lr = 1.3e-4
        self.num_epochs = 30
        self.best_loss = float("inf")
        self.phases = ["train", "val"]
        self.target = 3
        self.device = torch.device("cuda:0")
#         torch.set_default_tensor_type("torch.cuda.FloatTensor")  #this line is toxic... dont use it
        self.net = model
        self.criterion = torch.nn.BCEWithLogitsLoss()
        self.optimizer = optim.Adam(self.net.parameters(), lr=self.lr)
        self.scheduler = ReduceLROnPlateau(self.optimizer, factor=0.333, mode="min", patience=3, verbose=True)
        self.net = self.net.to(self.device)
        cudnn.benchmark = True
        
        self.train_dataset = SteelDataset(df_train, self.image_folder, self.target, phase = 'train')
        self.val_dataset = SteelDataset(df_val, self.image_folder, self.target, phase = 'val', )
        
        self.dataloaders = {
            'train' : DataLoader(
                self.train_dataset,
                batch_size = self.batch_size,
                num_workers = self.num_workers,
                shuffle = True,   
                ),
            'val' : DataLoader(
                self.val_dataset,
                batch_size = self.batch_size,
                num_workers = self.num_workers,
                shuffle = True,   
                ) 
        }
        
        self.losses = {phase: [] for phase in self.phases}
        self.iou_scores = {phase: [] for phase in self.phases}
        self.dice_scores = {phase: [] for phase in self.phases}
        
    def forward(self, images, targets):
        images = images.to(self.device)
        masks = targets.to(self.device)
        outputs = self.net(images)
        loss = self.criterion(outputs, masks)
        return loss, outputs

    def iterate(self, epoch, phase):
        meter = Meter(phase, epoch)
        start = time.strftime("%H:%M:%S")
        print(f" ⏰: {start} \nStarting epoch: {epoch} | phase: {phase} ")
        batch_size = self.batch_size
        self.net.train(phase == "train")
        dataloader = self.dataloaders[phase]
        running_loss = 0.0
        total_batches = len(dataloader)
        self.optimizer.zero_grad()
        
        n = 0
        for itr, batch in enumerate(dataloader):
            images, targets, _ = batch
            loss, outputs = self.forward(images, targets)
            loss = loss / self.accumulation_steps
            if phase == "train":
                loss.backward()
                if (itr + 1 ) % self.accumulation_steps == 0:
                    self.optimizer.step()
                    self.optimizer.zero_grad()
            running_loss += loss.item()
            outputs = outputs.detach().cpu()
            meter.update(targets, outputs)
            n += 1
            if n % 200 == 0:
                print ("iteration", n )
#                 checker.append(outputs[0])
#                 checker_target.append(targets[0])
#                 checker_image.append(images[0])
#                 plt.imshow(checker_image[-1][0])
#                 plt.show()
#                 plt.imshow(checker_target[-1][0])
#                 plt.show()
#                 plt.imshow(Tensor.cpu(checker[-1][0]).detach().numpy())
#                 plt.show()
                
        epoch_loss = (running_loss * self.accumulation_steps) / total_batches
        dice, iou = epoch_log(phase, epoch, epoch_loss, meter, start)
        self.losses[phase].append(epoch_loss)
        self.dice_scores[phase].append(dice)
        self.iou_scores[phase].append(iou)
        torch.cuda.empty_cache()
        return epoch_loss

    def start(self):
        for epoch in range(self.num_epochs):
            self.iterate(epoch, "train")
            state = {
                "epoch": epoch,
#                 "best_dice": self.best_dice, 
                "best_loss": self.best_loss,
                "state_dict": self.net.state_dict(),
                "optimizer": self.optimizer.state_dict(),
            }
            with torch.no_grad():
                val_loss = self.iterate(epoch, "val")
                self.scheduler.step(val_loss)
            if val_loss < self.best_loss:
                print("******** New optimal found, saving state ********")
                state["best_loss"] = self.best_loss = val_loss
                torch.save(state, "./model.pth")
            print()

# %% [code]
def prepare_model( backbone = 'se_resnet50', weight='None', checkpoint_path=None):
    '''
    change channel 3 to channel 1
    original as the following
        # first layer
           model.encoder.layer0.conv1 = nn.Conv2d(3, 64,...)
        # last layer
           model.decoder.final_conv = nn.Conv2d(512, 3,...)
    '''
    if weight == 'None':
        model = smp.PSPNet(backbone, encoder_weights=None, classes=1, activation=None)
    elif weight == 'imagenet':
        model = smp.PSPNet(backbone, encoder_weights='imagenet', classes=1, activation=None)
    elif weight == 'pretrained':
        model = smp.PSPNet(backbone, encoder_weights=None, classes=1, activation=None)
        model.encoder.layer0.conv1 = nn.Conv2d(1, 64,kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        model.to(torch.device("cuda:0"))
        state = torch.load(checkpoint_path, map_location=lambda storage, loc: storage)
        model.load_state_dict(state["state_dict"])
        
    model.encoder.layer0.conv1 = nn.Conv2d(1, 64,kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
    
    return model

model = prepare_model(backbone = 'se_resnet101', weight='pretrained', checkpoint_path='../input/segmenter-class3/model.pth')
model_trainer = Trainer(model, df_train, df_val)
model_trainer.start()

losses = model_trainer.losses
dice_scores = model_trainer.dice_scores # overall dice
iou_scores = model_trainer.iou_scores

def plot(scores, name):
    plt.figure(figsize=(15,5))
    plt.plot(range(len(scores["train"])), scores["train"], label=f'train {name}')
    plt.plot(range(len(scores["train"])), scores["val"], label=f'val {name}')
    plt.title(f'{name} plot'); plt.xlabel('Epoch'); plt.ylabel(f'{name}');
    plt.legend(); 
    plt.show()

plot(losses, "BCE loss")
plot(dice_scores, "Dice score")
plot(iou_scores, "IoU score")