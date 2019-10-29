import os, sys
import subprocess
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
from torch import functional as F

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
Using TensorFlow backend.
# target = torch.ones([10, 4], dtype=torch.float32)
# output = torch.full([10, 4], 0.332)
# pos_weight = torch.tensor([1.,1.,1.,1.])  

# # BCEWithLogitsLoss
# criterion = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)
# loss1 = criterion(output, target) 

# # BCELoss
# m = nn.Sigmoid()
# criterion = torch.nn.BCELoss()
# loss2 = criterion(m(output), target)

# print ("loss1, loss2:", loss1, loss2)

# Summerization:
# BCELoss need logit(probs) as input, while BCEWithLogitsLoss need input of probs.
# BCEWithLogitsLoss is more flexible it has pos_weight as factor
def install(package):
    subprocess.call([sys.executable, "-m", "pip", "install", package])    
install('../input/pretrainedmodels/pretrainedmodels-0.7.4/pretrainedmodels-0.7.4')
install('../input/segmentationpytorch/segmentation_models.pytorch-0.0.3')
import segmentation_models_pytorch as smp
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
<matplotlib.axes._subplots.AxesSubplot at 0x7f0d931215f8>

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
df.head(1) # make sure the name/index is correct
ClassId	1	2	3	4	defects	ClassId
ImageId						
0002cc93b.jpg	29102 12 29346 24 29602 24 29858 24 30114 24 3...	NaN	NaN	NaN	1	1
def udf_transformer(phase):
    
    list_transforms = []
    
#     if target in [1,3,4]:
#         pc_min, pc_max = 150, 50000
#     else:
#         pc_min, pc_max = 5, 50000
        
#     list_transforms.extend([CropNonEmptyMaskIfExists(height = 256, 
#                                                      width = 400, # modify this
# #                                                      ignore_values = [pc_min,pc_max], # [min, max], 
#                                                      p=1)
# #                             Resize(height = 256,
# #                                    width = 1600)
#                            ]
#                           )
#     list_transforms.append(ElasticTransform(p=0.6))
#     list_transforms.append(OpticalDistortion(p=0.6))
    if phase == 'train':
        list_transforms.append(HorizontalFlip(p=0.5))
        list_transforms.append(VerticalFlip(p=0.5))
        list_transforms.append(Rotate(limit=20, p=0.5))
    #     list_transforms.append(GaussNoise())
        list_transforms.append(GridDistortion(distort_limit=(-0.2, 0.2), p =0.6))

    list_transforms.append(Normalize([0.5], [0.5]))  # important step
    list_transforms.append(ToTensor()) # using albumentations transformers, must use ToTensor from albumentations too
    return Compose(list_transforms)
def make_mask(df,row_id):
    '''Given a row index, return image_id and mask (256, 1600, 4) from the dataframe `df`'''
    fname = df.iloc[row_id].name
    labels = df.iloc[row_id][:4] # don't change :4 take 4 columns 
    masks = np.zeros((256, 1600, 4), dtype=np.float32)

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
    def __init__(self, df, image_folder, phase, crop_size = 400):
        self.df = df
        self.root = image_folder
        self.phase = phase
        self.fnames = self.df.index.tolist()
        self.crop_size = crop_size

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
        label = (self.df.iloc[idx,:4].isnull() == False).astype(int).values.reshape(1,-1)
        label = torch.tensor(label, dtype = torch.float32)
        
        self.transformer = udf_transformer(self.phase)
        augmented = self.transformer(image=image, mask=mask)
        image = augmented['image']
        mask = augmented['mask']
        
#         to make image and mask 3 dimension
        image = image.reshape(1, image.shape[0], image.shape[1])
        mask = mask.permute(0,3,2,1)[0]
        
        return image, mask, label

    def __len__(self):
        return len(self.fnames)
    
## SteelData tester
value = SteelDataset(df, '../input/severstal-steel-defect-detection/train_images', 'train')[2]

plt.imshow(value[0][0])
plt.show()

plt.imshow(value[1][0])  # class 1 mask
plt.show()

plt.imshow(value[1][1])  # class 3 mask
plt.show()

print (value[0].shape, value[1].shape, value[2].shape)

print (value[2])



torch.Size([1, 1600, 256]) torch.Size([4, 1600, 256]) torch.Size([1, 4])
tensor([[1., 1., 0., 0.]])
# following code used to see stratify standard 

df_train, df_val = train_test_split(df, test_size=0.15, stratify=df['ClassId'], random_state=7)

# print ("df_train.shape, df_val.shape, df_train.ClassId.value_counts(), df_val.ClassId.value_counts()",
#        df_train.shape, df_val.shape, df_train.ClassId.value_counts(), df_val.ClassId.value_counts())

classes_train = df_train.ClassId.value_counts().index
samples_train = df_train.ClassId.value_counts().values
sns.barplot(classes_train, samples_train)

# # comparing with stratify with df['defects'], the distribution is simlar, but more straight forward
# # df_train, df_val = train_test_split(df, test_size=0.15, stratify=df['defects'], random_state=7)

# # print ("df_train.shape, df_val.shape, df_train.ClassId.value_counts(), df_val.ClassId.value_counts()",
# #        df_train.shape, df_val.shape, df_train.ClassId.value_counts(), df_val.ClassId.value_counts())

# # classes_train = df_train.ClassId.value_counts().index
# # samples_train = df_train.ClassId.value_counts().values
# # sns.barplot(classes_train, samples_train)
<matplotlib.axes._subplots.AxesSubplot at 0x7f0d931ba630>

balance classes
class DatasetPrep():
    
    def __init__(self, df, random_state=10, frac_0 = 1, frac_3 =1, balance = True):
        '''
        random_state decides how split. normally don't change in all trainings.
        num_drop means cutting number of class-3. decides balancing. can change to see balance impacts the training. normally don't change in a single training.
        '''
        self.df = df
        self.random_state = random_state
        self.frac_0 = frac_0
        self.frac_3 = frac_3
        self.balance = balance

    def split(self, df, random_state):
        df_train, df_val = train_test_split(df, test_size=0.15, stratify=df["ClassId"], random_state=random_state)
        return df_train, df_val
    
    def downsample(self, df):
        print (df.shape)

        df_class_0 = df[df.ClassId == 0].sample(frac = self.frac_0, replace = False)
        df_class_1 = df[df.ClassId == 1]
        df_class_2 = df[df.ClassId == 2]
        df_class_3 = df[df.ClassId == 3].sample(frac = self.frac_3, replace = False)
        df_class_4 = df[df.ClassId == 4]

        df = pd.concat([df_class_0, df_class_1, df_class_2, df_class_3, df_class_4])
        print (df.shape)
        return df

    def oversample(self, df):
        '''
        # oversample by x factor
        '''
        class_3_count = df[df.ClassId == 3].ClassId.count()
        factor_1 = class_3_count // df[df.ClassId == 1].ClassId.count()-1
        factor_2 = class_3_count // df[df.ClassId == 2].ClassId.count()-3
        factor_4 = class_3_count // df[df.ClassId == 4].ClassId.count()-1
        
        print ("factors", factor_1, factor_2, factor_4)
        
        df_1 = pd.concat([df[df.ClassId == 1]]*factor_1)
        df_2 = pd.concat([df[df.ClassId == 2]]*(factor_2))
        df_3 = pd.concat([df[df.ClassId == 3]])
        df_4 = pd.concat([df[df.ClassId == 4]]*factor_4)
        df_0 = pd.concat([df[df.ClassId == 0]])
        return pd.concat([df_1, df_2, df_3, df_4, df_0])
    
    def run(self):
        self.df_train, self.df_val = self.split(self.df, self.random_state)
        if self.balance == True:
            self.df_train = self.downsample(self.df_train)
            self.df_train = self.oversample(self.df_train)
            
        return self.df_train, self.df_val
# generagte df_train, df_val for one training
df_train, df_val = DatasetPrep(df, balance=True).run()

print ('distribution\n', df_train.ClassId.value_counts())

classes_train = df_train.ClassId.value_counts().index
samples_train = df_train.ClassId.value_counts().values
sns.barplot(classes_train, samples_train)

# for num_sample in range(10):
#     val = SteelDataset(df_classes_out, image_folder, phase = 'train')[num_sample]
#     plt.imshow(val[0])
#     plt.show()
#     plt.imshow(val[1])
#     plt.show()
(11047, 6)
(11047, 6)
factors 4 17 5
distribution
 0    5017
3    4377
2    3570
4    3405
1    3048
Name: ClassId, dtype: int64
<matplotlib.axes._subplots.AxesSubplot at 0x7f0d92880898>

# def metric(probability, truth, threshold=0.5, reduction='none'):
#     '''Calculates dice of positive and negative images seperately'''
#     '''probability and truth must be torch tensors'''
#     batch_size = len(truth)
#     with torch.no_grad():
#         probability = probability.view(batch_size, -1)
#         truth = truth.view(batch_size, -1)
#         assert(probability.shape == truth.shape)

#         p = (probability > threshold).float()
#         t = (truth > 0).float()
#         # =========
# #         print ("probability, truth, p, t", probability.shape, truth.shape, p.shape, t.shape)
#         # =========
#         t_sum = t.sum(-1)
#         p_sum = p.sum(-1)
#         neg_index = torch.nonzero(t_sum == 0)
#         pos_index = torch.nonzero(t_sum >= 1)        
# #         print ("Truth: neg_index, pos_index", neg_index, pos_index)
#         dice_neg = (p_sum == 0).float()
#         dice_pos = 2 * (p*t).sum(-1)/((p+t).sum(-1))        
# #         print ("Predict: dice_neg, dice_pos", dice_neg, dice_pos)
#         dice_neg = dice_neg[neg_index]
#         dice_pos = dice_pos[pos_index]
#         dice = torch.cat([dice_pos, dice_neg])        
# #         print ("Again Predict: dice_neg, dice_pos", dice_neg, dice_pos)
#         dice_neg = np.nan_to_num(dice_neg.mean().item(), 0)
#         dice_pos = np.nan_to_num(dice_pos.mean().item(), 0)      
# #         print ("Once more Predict: dice_neg, dice_pos", dice_neg, dice_pos)       
#         dice = dice.mean().item()
#         num_neg = len(neg_index)
#         num_pos = len(pos_index)       
# #         print ("num_neg, num_pos", num_neg, num_pos, '\n')

#     return dice, dice_neg, dice_pos, num_neg, num_pos
Train Classifier
class Trainer(object):
    '''This class takes care of training and validation of our model'''
    def __init__(self, model, df_train, df_val):
        self.image_folder = '../input/severstal-steel-defect-detection/train_images'
        self.num_workers = 6
        self.batch_size = 8
        self.accumulation_steps = 96 // self.batch_size
        self.lr = 5e-3
        self.num_epochs = 20
        self.best_loss = float("inf")
        self.phases = ["train", "val"]
#         self.device = torch.device("cuda:0")
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.net = model
        self.criterion = torch.nn.BCEWithLogitsLoss()
        self.optimizer = optim.Adam(self.net.parameters(), lr=self.lr)
        self.scheduler = ReduceLROnPlateau(self.optimizer, factor=0.333, mode="min", patience=2, verbose=True)
        self.net = self.net.to(self.device)
        cudnn.benchmark = True
        
        self.train_dataset = SteelDataset(df_train, self.image_folder , phase = 'train')
        self.val_dataset = SteelDataset(df_val, self.image_folder, phase = 'val')
        
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
        self.accuracy = {phase: [] for phase in self.phases}
        
    def forward(self, images, targets, labels):
        images = images.to(self.device)
        masks = targets.to(self.device)
        labels = labels.to(self.device)
        outputs = self.net(images)
        # reshape
        outputs = outputs.reshape(-1,1,4)   
        # ===
#         print ("labels and outputs shape", labels.shape, outputs.shape)
#         print ("labels and outputs", labels, outputs)
        loss = self.criterion(outputs, labels) # here use labels to compare with outputs for classification
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
            images, targets, labels = batch
            loss, outputs = self.forward(images, targets, labels)
            loss = loss / self.accumulation_steps
            if phase == "train":
                loss.backward()
                if (itr + 1 ) % self.accumulation_steps == 0:
                    self.optimizer.step()
                    self.optimizer.zero_grad()
            running_loss += loss.item()
            outputs = outputs.detach().cpu()
            # need to modify following ====
            meter.update(labels, outputs)
            # ==============================
            
            n += 1
            if n % 500 == 0:
                print ("iteration", n )
#                 checker.append(outputs[0])
#                 checker_target.append(targets[0])
#                 checker_image.append(images[0])
#                 plt.imshow(images[0])
#                 plt.show()
#                 plt.imshow(targets[0])
#                 plt.show()
#                 plt.imshow(Tensor.cpu(outputs[0]).detach().numpy())
#                 plt.show()
                
        epoch_loss = (running_loss * self.accumulation_steps) / total_batches
        accuracy = epoch_log(phase, epoch, epoch_loss, meter, start)
        self.losses[phase].append(epoch_loss)
        self.accuracy[phase].append(accuracy)
        torch.cuda.empty_cache()
        return epoch_loss

    def start(self):
        for epoch in range(self.num_epochs):
            self.iterate(epoch, "train")
            state = {
                "epoch": epoch,
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
class Classifier(nn.Module):
    '''
    a classifier
    '''
    def __init__(self, in_channels=2048, out_channels=256, kernel_size=1, stride=1, output_channels=256, num_classes=4):
        super(Classifier, self).__init__()
        
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride)
        self.bn1 = nn.BatchNorm2d(out_channels,momentum=0.5)
        self.pool1 = nn.AdaptiveAvgPool2d(output_size = (1,1)) # or MaxPool2d
        self.conv2 = nn.Conv2d(in_channels=out_channels, out_channels = num_classes, kernel_size=kernel_size, stride=stride)
        
    def forward(self, x):
        x = x[0]
        out = self.conv1(x)
        out = self.bn1(out)
        #print (out.shape) #torch.Size([1, 256, 13, 8])
        out = self.pool1(out) 
        #print (out.shape) #torch.Size([1, 256, 1, 1])
        out = self.conv2(out) 
        #print (out.shape) #torch.Size([1, 4, 1, 1])
        return out
def prepare_model_classifier( backbone = 'resnet50', weight='None'):
    '''
    change channel 3 to channel 1
    original as the following
        # first layer
           model.encoder.layer0.conv1 = nn.Conv2d(3, 64,...)
        # last layer
           model.decoder.final_conv = nn.Conv2d(512, 3,...)
    '''
    if weight == 'None':
        model = smp.Unet(backbone, encoder_weights=None, classes=4, activation=None)
    elif weight == 'imagenet':
        model = smp.Unet(backbone, encoder_weights='imagenet', classes=4, activation=None)
        
    if backbone == 'se_resnet50':
        model.encoder.layer0.conv1 = nn.Conv2d(1, 64,kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False) # input size
    elif backbone == 'resnet50':
        model.encoder.conv1 = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)

    return model
encoder = prepare_model_classifier(backbone = 'resnet50', weight='None').encoder
model = nn.Sequential(encoder, Classifier())

checkpoint_path = '../input/classifier-v1/model.pth'

model.to(torch.device("cuda:0"))
state = torch.load(checkpoint_path, map_location=lambda storage, loc: storage)
model.load_state_dict(state["state_dict"])
<All keys matched successfully>
# # test
# val = SteelDataset(df, '../input/severstal-steel-defect-detection/train_images', 'train')[1]
# images = val[0].reshape(1,1,400,256)

# plt.imshow(val[0][0])
# print(val[2][0])

# # plt.imshow(val[0][0])
# plt.imshow(val[1][0])

# output = model(images)
# print (output)
class Meter:
    '''A meter to keep track of iou and dice scores throughout an epoch'''
    def __init__(self, phase, epoch):
        self.base_threshold = 0.5 # <<<<<<<<<<< here's the threshold
        self.accuracy = []
        self.accuracy_class1 = []
        self.accuracy_class2 = []
        self.accuracy_class3 = []
        self.accuracy_class4 = []
#         self.iou_scores = []

    def update(self, labels, outputs):
        # ================ TO-DO : modify ==============
        accuracy, accuracy_class1, accuracy_class2, accuracy_class3, accuracy_class4 = metric(outputs, labels, self.base_threshold) # need to use accuracy
        self.accuracy.append(accuracy)
        self.accuracy_class1.append(accuracy_class1)
        self.accuracy_class2.append(accuracy_class2)
        self.accuracy_class3.append(accuracy_class3)
        self.accuracy_class4.append(accuracy_class4)
#         preds = predict(probs, self.base_threshold)
#         iou = compute_iou_batch(preds, targets, classes=[1])
        # ================== modify ==============

    def get_metrics(self):
        accuracy = np.mean(self.accuracy)
        accuracy_class1 = np.nanmean(self.accuracy_class1)
        accuracy_class2 = np.nanmean(self.accuracy_class2)
        accuracy_class3 = np.nanmean(self.accuracy_class3)
        accuracy_class4 = np.nanmean(self.accuracy_class4)
        return accuracy, accuracy_class1, accuracy_class2, accuracy_class3, accuracy_class4

def epoch_log(phase, epoch, epoch_loss, meter, start):
    '''logging the metrics at the end of an epoch'''
    accuracy, accuracy_class1, accuracy_class2, accuracy_class3, accuracy_class4 = meter.get_metrics()
    print("Loss: %0.4f | Accuracy: %0.4f | Class1: %0.4f | Class2: %0.4f | Class3: %0.4f | Class4: %0.4f" % (epoch_loss, accuracy, accuracy_class1, accuracy_class2, accuracy_class3, accuracy_class4))
    return accuracy

# def compute_ious(pred, label, classes, ignore_index=255, only_present=True):
#     '''computes iou for one ground truth mask and predicted mask'''
#     pred[label == ignore_index] = 0
#     ious = []
#     for c in classes:
#         label_c = label == c
#         if only_present and np.sum(label_c) == 0:
#             ious.append(np.nan)
#             continue
#         pred_c = pred == c
#         intersection = np.logical_and(pred_c, label_c).sum()
#         union = np.logical_or(pred_c, label_c).sum()
#         if union != 0:
#             ious.append(intersection / union)
#     return ious if ious else [1]

# def compute_iou_batch(outputs, labels, classes=None):
#     '''computes mean iou for a batch of ground truth masks and predicted masks'''
#     ious = []
#     preds = np.copy(outputs) # copy is imp
#     labels = np.array(labels) # tensor to np
#     for pred, label in zip(preds, labels):
#         ious.append(np.nanmean(compute_ious(pred, label, classes)))
#     iou = np.nanmean(ious)
#     return iou
def metric(outputs, labels, threshold=0.5):
    '''
    calculate classification accuracy
    '''
    with torch.no_grad():
        probs = torch.nn.Sigmoid()(outputs)  # sigmoid is critical. for loss function, it takes log first. oh i can't put this into NN as activation
        preds = torch.where(probs > 0.5 , torch.ones_like(probs), torch.zeros_like(probs))
#         print ("labels, probs", labels, preds)
#         print (labels.shape)
        total = labels.size(0)*4
        correct = (labels == preds).sum().item()
        accuracy = correct/total
        
        accuracy_1 = (labels[:,:,0] == preds[:,:,0]).sum().item()/(labels.size(0))
        accuracy_2 = (labels[:,:,1] == preds[:,:,1]).sum().item()/(labels.size(0))
        accuracy_3 = (labels[:,:,2] == preds[:,:,2]).sum().item()/(labels.size(0))
        accuracy_4 = (labels[:,:,3] == preds[:,:,3]).sum().item()/(labels.size(0))
        
    return accuracy, accuracy_1, accuracy_2, accuracy_3, accuracy_4
model_trainer = Trainer(model, df_train, df_val)
model_trainer.start()