#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Filename : dataloader
# @Date : 2022-09-01-09-06
# @Project : pytorch_basic
# @Author : Jang

import os
from glob import glob
import pandas as pd

from PIL import Image
import numpy as np
import cv2

from torch.utils.data import DataLoader, Dataset
from torch.utils.data.sampler import SubsetRandomSampler
import torchvision.transforms as T

np.random.seed(0)

def get_train_data(data_dir):
    label_df = pd.read_csv('../dataset/train.csv')
    img_path_list = []
    label_list = []

    img_path_list.extend(glob(os.path.join(data_dir,'*.PNG')))
    img_path_list.sort(key = lambda x :int(x.split('/')[-1].split('.')[0][-3:]))

    label_list.extend(label_df['label'])

    return img_path_list, label_list

def get_test_data(data_dir):
    img_path_list = []

    img_path_list.extend(glob(os.path.join(data_dir,'*.PNG')))
    img_path_list.sort(key = lambda x :int(x.split('/')[-1].split('.')[0][-3:]))

    return img_path_list


# def make_datapath_list(root_path):
#     txt_list = os.listdir(root_path)

#     data_list = []
#     for idx, txt in enumerate(txt_list):
#         with open(os.path.join(root_path, txt)) as f:
#             file_list = [line.rstrip() for line in f]
#             file_list = [line for line in file_list if line]
#             data_list.extend(file_list)

#     print("\nNumber of classes: {}".format(len(txt_list)))
#     print("Number of training data: {}".format(len(data_list)))
#     return data_list, txt_list

class CustomDataset(Dataset):
    
    def __init__(self, img_path_list, label_list, train_mode=True, transforms=None):
        self.transforms = transforms
        self.train_mode = train_mode
        self.img_path_list = img_path_list
        self.label_list = label_list

    def __len__(self):
        return len(self.img_path_list)

    def __getitem__(self, index):
        img_path = self.img_path_list[index]

        image = cv2.imread(img_path) # 필요한 경우, .resize((224,224)), .convert('RGB')

        if self.transforms is not None:
            image = self.transforms(image)

        if self.train_mode:
            label = self.label_list[index]
            return image, label
        else:
            return image

# class MyDataset(object):

#     def __init__(self, file_list, transform=None):
#         self.file_list = file_list
#         self.transform = transform

#     def __len__(self):
#         return len(self.file_list)

#     def __getitem__(self, index):
#         img_path = self.file_list[index]

#         img = Image.open(img_path) # 필요한 경우, .resize((224,224)), .convert('RGB')

#         if self.transform is not None:
#             img = self.transform(img)

#         if img_path.split('/')[-2].split('_')[-1][:1] == "a": # 파일명, 또는 폴더명에 a 가 들어가면 비정상으로 라벨링 함.
#             label = 1 ## 비정상
#         else:
#             label = 0 ## 정상

#         return img, label

# 학습 데이터셋 wrapper 클래스
class MyTrainSetWrapper(object):

    def __init__(self, batch_size, num_workers, valid_size, train_path, image_size):
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.valid_size = valid_size
        self.path = train_path
        self.image_size = image_size

    def get_data_loaders(self):
        data_augment = self._get_train_transform()

        train_dataset = CustomDataset(get_train_data(self.path)[0], get_train_data(self.path)[1], train_mode=True,transforms=data_augment)

        train_loader, valid_loader = self.get_train_validation_data_loaders(train_dataset)

        return train_loader, valid_loader

    def _get_train_transform(self):
        data_transforms = T.Compose([T.ToPILImage(),
                                     T.Resize(self.image_size),
                                     T.RandomHorizontalFlip(),
                                     T.RandomVerticalFlip(),
                                     T.RandomRotation(10),
                                     T.ToTensor(),
                                     T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])])
        return data_transforms

    def get_train_validation_data_loaders(self, train_dataset):
        # obtain training indices that will be used for validation
        num_train = len(train_dataset)
        indices = list(range(num_train))
        #print(indices)
        np.random.shuffle(indices)

        split = int(np.floor(self.valid_size * num_train))
        train_idx, valid_idx = indices[split:], indices[:split]

        # define samplers for obtaining training and validation batches
        train_sampler = SubsetRandomSampler(train_idx)
        valid_sampler = SubsetRandomSampler(valid_idx)

        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, sampler=train_sampler,
                                  num_workers=self.num_workers, shuffle=False,
                                  pin_memory=True)

        ## 입력 텐서를 직접 확인하고 싶은 경우, 이하를 실행함.

        #print('-----')
        #for x, y in train_loader:
        #    print("x_len:{0}, x_shape:{1}, x_type:{2}, y:{3}".format(len(x), x[0].shape, type(x[0]), y))
        #print('-----')

        valid_loader = DataLoader(train_dataset, batch_size=self.batch_size, sampler=valid_sampler,
                                  num_workers=self.num_workers, drop_last=True,
                                  pin_memory=True)

        print("\nValidation size: {}%".format(self.valid_size*100))
        print("Train set: {} / Validation set: {}".format(len(train_loader), len(valid_loader)))
        return train_loader, valid_loader

# 테스트 데이터셋 wrapper 클래스
class MyTestSetWrapper(object):

    def __init__(self, batch_size, num_workers, test_path):
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.path = test_path

    def _get_test_transform(self):
        data_transforms = T.Compose([T.ToPILImage(),
                                     T.Resize(self.image_size),
                                    #  T.RandomHorizontalFlip(),
                                    #  T.RandomVerticalFlip(),
                                    #  T.RandomRotation(10),
                                     T.ToTensor(),
                                     T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])])
        return data_transforms

    def get_test_loaders(self):
        data_augment = self._get_test_transform()

        test_dataset = CustomDataset(get_test_data(self.path), train_mode=False,transforms=data_augment)

        test_loader = DataLoader(test_dataset, batch_size=self.batch_size, num_workers=self.num_workers,
                                 shuffle=False, pin_memory=True)

        return test_loader