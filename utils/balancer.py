import segmentation_models_pytorch as smp
import os
import sys
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
folder_input = '/kaggle/input/severstal-steel-defect-detection'
folder_train_input = '/kaggle/input/severstal-steel-defect-detection/train_images/'
folder_test_input = '/kaggle/input/severstal-steel-defect-detection/test_images/'
path_mask_train = '/kaggle/input/severstal-steel-defect-detection/train.csv'
folder = '../input/severstal-steel-defect-detection/'
paths_imgs_train = [folder_train_input +
                    e for e in os.listdir(folder_train_input)]
paths_imgs_test = [folder_test_input +
                   e for e in os.listdir(folder_test_input)]

df = pd.read_csv(folder + 'train.csv')
df['ImageId'], df['ClassId'] = zip(*df['ImageId_ClassId'].str.split('_'))
df['ClassId'] = df['ClassId'].astype(int)
df = df.pivot(index='ImageId', columns='ClassId', values='EncodedPixels')

df['defects'] = df.count(axis=1)
sns.barplot(df.defects.value_counts().index, df.defects.value_counts().values)
<matplotlib.axes._subplots.AxesSubplot at 0x7f6299b515f8 >


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

    df = pd.concat(
        [df_class_1, df_class_2, df_class_3, df_class_4, df_class_0])
    return df


df = reform(df)


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
        list_transforms.append(GridDistortion(
            distort_limit=(-0.2, 0.2), p=0.6))

    list_transforms.append(Normalize([0.5], [0.5]))  # important step
    # using albumentations transformers, must use ToTensor from albumentations too
    list_transforms.append(ToTensor())
    return Compose(list_transforms)


def make_mask(df, row_id):
    '''Given a row index, return image_id and mask (256, 1600, 4) from the dataframe `df`'''
    fname = df.iloc[row_id].name
    labels = df.iloc[row_id][:4]  # don't change :4 take 4 columns
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
    def __init__(self, df, image_folder, phase, crop_size=400):
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
        label = (self.df.iloc[idx, :4].isnull() ==
                 False).astype(int).values.reshape(1, -1)
        label = torch.tensor(label, dtype=torch.float32)

        self.transformer = udf_transformer(self.phase)
        augmented = self.transformer(image=image, mask=mask)
        image = augmented['image']
        mask = augmented['mask']

#         to make image and mask 3 dimension
        image = image.reshape(1, image.shape[0], image.shape[1])
        mask = mask.permute(0, 3, 2, 1)[0]

        return image, mask, label

    def __len__(self):
        return len(self.fnames)


# SteelData tester
value = SteelDataset(
    df, '../input/severstal-steel-defect-detection/train_images', 'train')[2]

plt.imshow(value[0][0])
plt.show()

plt.imshow(value[1][0])  # class 1 mask
plt.show()

plt.imshow(value[1][1])  # class 3 mask
plt.show()

print(value[0].shape, value[1].shape, value[2].shape)

print(value[2])


torch.Size([1, 1600, 256]) torch.Size([4, 1600, 256]) torch.Size([1, 4])
tensor([[1., 1., 0., 0.]])
# following code used to see stratify standard

df_train, df_val = train_test_split(
    df, test_size=0.15, stratify=df['ClassId'], random_state=7)

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
<matplotlib.axes._subplots.AxesSubplot at 0x7f6299be8c88 >

balance classes


class DatasetPrep():

    def __init__(self, df, random_state=10, frac_0=0.95, frac_3=0.95, balance=True):
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
        df_train, df_val = train_test_split(
            df, test_size=0.15, stratify=df["ClassId"], random_state=random_state)
        return df_train, df_val

    def downsample(self, df):
        print(df.shape)

        df_class_0 = df[df.ClassId == 0].sample(
            frac=self.frac_0, replace=False)
        df_class_1 = df[df.ClassId == 1]
        df_class_2 = df[df.ClassId == 2]
        df_class_3 = df[df.ClassId == 3].sample(
            frac=self.frac_3, replace=False)
        df_class_4 = df[df.ClassId == 4]

        df = pd.concat(
            [df_class_0, df_class_1, df_class_2, df_class_3, df_class_4])
        print(df.shape)
        return df

    def oversample(self, df):
        '''
        # oversample by x factor
        '''
        class_3_count = df[df.ClassId == 3].ClassId.count()
        factor_1 = class_3_count // df[df.ClassId == 1].ClassId.count()-1
        factor_2 = class_3_count // df[df.ClassId == 2].ClassId.count()-3
        factor_4 = class_3_count // df[df.ClassId == 4].ClassId.count()-1

        print("factors", factor_1, factor_2, factor_4)

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

print('distribution\n', df_train.ClassId.value_counts())

classes_train = df_train.ClassId.value_counts().index
samples_train = df_train.ClassId.value_counts().values
sns.barplot(classes_train, samples_train)
