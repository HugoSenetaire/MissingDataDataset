
import os
import torch
import torchvision
import h5py
import numpy as np
import pickle
from ..custom_image_dataset import CustomImageDataset
import copy
import cv2
import random


class DatasetTrainAug(torch.utils.data.Dataset):
    def __init__(self, imgs, poses):
        self.imgs = imgs
        self.poses = poses
        self.crop_size = 64

        self.num_examples = self.imgs.shape[0]

        print ("DatasetTrainAug - number of images: %d" % self.num_examples)
        print ("DatasetTrainAug - max Yaw: %g" % np.max(self.poses[:, 0]))
        print ("DatasetTrainAug - min Yaw: %g" % np.min(self.poses[:, 0]))
        print ("DatasetTrainAug - max Pitch: %g" % np.max(self.poses[:, 1]))
        print ("DatasetTrainAug - min Pitch: %g" % np.min(self.poses[:, 1]))
        print ("DatasetTrainAug - max Roll: %g" % np.max(self.poses[:, 2]))
        print ("DatasetTrainAug - min Roll: %g" % np.min(self.poses[:, 2]))

    def __len__(self):
        return self.num_examples

    def __getitem__(self, index):
        img = self.imgs[index] # (shape: (64, 64, 3))
        pose = self.poses[index] # (3, ) (Yaw, Pitch, Roll)

        # # # # # # # # # # # # # # # # # # # # # # # # debug visualization START
        # print (pose)
        # print (img.shape)
        # cv2.imshow("img", img)
        # cv2.waitKey(0)
        # print ("#####")
        # # # # # # # # # # # # # # # # # # # # # # # # debug visualization END

       # flip img along the vertical axis with 0.5 probability:
        flip = np.random.randint(low=0, high=2)
        if flip == 1:
            img = cv2.flip(img, 1)
            pose[0] = -1.0*pose[0]
            pose[2] = -1.0*pose[2]

        # # # # # # # # # # # # # # # # # # # # # # # # debug visualization START
        # print (flip)
        # print (pose)
        # print (img.shape)
        # cv2.imshow("img", img)
        # cv2.waitKey(0)
        # print ("#####")
        # # # # # # # # # # # # # # # # # # # # # # # # debug visualization END

        # scale the size of the image with factor in [0.7, 1.4]:
        f_scale = 0.7 + random.randint(0, 8)/10.0
        img = cv2.resize(img, None, fx=f_scale, fy=f_scale, interpolation=cv2.INTER_LINEAR)

        # # # # # # # # # # # # # # # # # # # # # # # # debug visualization START
        # print (f_scale)
        # print (pose)
        # print (img.shape)
        # cv2.imshow("img", img)
        # cv2.waitKey(0)
        # print ("#####")
        # # # # # # # # # # # # # # # # # # # # # # # # debug visualization END

        # pad the image if needed:
        img_h, img_w, _ = img.shape
        pad_h = max(self.crop_size - img_h, 0)
        pad_w = max(self.crop_size - img_w, 0)
        if pad_h > 0 or pad_w > 0:
            img = cv2.copyMakeBorder(img, 0, pad_h, 0, pad_w, cv2.BORDER_CONSTANT, value=(0.0, 0.0, 0.0))

        # # # # # # # # # # # # # # # # # # # # # # # # debug visualization START
        # print (pose)
        # print (img.shape)
        # cv2.imshow("img", img)
        # cv2.waitKey(0)
        # print ("#####")
        # # # # # # # # # # # # # # # # # # # # # # # # debug visualization END

        # select a random (64, 64) crop:
        img_h, img_w, _ = img.shape
        h_off = random.randint(0, img_h - self.crop_size)
        w_off = random.randint(0, img_w - self.crop_size)
        img = img[h_off:(h_off+self.crop_size), w_off:(w_off+self.crop_size)] # (shape: (crop_size, crop_size, 3))

        # # # # # # # # # # # # # # # # # # # # # # # # debug visualization START
        # print (pose)
        # print (img.shape)
        # cv2.imshow("img", img)
        # cv2.waitKey(0)
        # # # # # # # # # # # # # # # # # # # # # # # # debug visualization END

        img = img/255.0
        img = img - np.array([0.485, 0.456, 0.406])
        img = img/np.array([0.229, 0.224, 0.225])
        img = np.transpose(img, (2, 0, 1)) # (shape: (3, 64, 64))
        img = torch.from_numpy(img.astype(np.float32))

        pose = torch.from_numpy(pose.astype(np.float32))

        return (img, pose)


class DatasetTest(torch.utils.data.Dataset):
    def __init__(self, imgs, poses):

        self.imgs = imgs # (shape: (5065, 64, 64, 3))
        self.poses = poses # (shape: (5065, 3)) (Yaw, Pitch, Roll)


        self.num_examples = self.imgs.shape[0]

        print ("DatasetTest - number of images: %d" % self.num_examples)
        print ("DatasetTest - max Yaw: %g" % np.max(self.poses[:, 0]))
        print ("DatasetTest - min Yaw: %g" % np.min(self.poses[:, 0]))
        print ("DatasetTest - max Pitch: %g" % np.max(self.poses[:, 1]))
        print ("DatasetTest - min Pitch: %g" % np.min(self.poses[:, 1]))
        print ("DatasetTest - max Roll: %g" % np.max(self.poses[:, 2]))
        print ("DatasetTest - min Roll: %g" % np.min(self.poses[:, 2]))

    def __len__(self):
        return self.num_examples

    def __getitem__(self, index):
        img = self.imgs[index] # (shape: (64, 64, 3))
        pose = self.poses[index] # (3, ) (Yaw, Pitch, Roll)

        img = img/255.0
        img = img - np.array([0.485, 0.456, 0.406])
        img = img/np.array([0.229, 0.224, 0.225])
        img = np.transpose(img, (2, 0, 1)) # (shape: (3, 64, 64))
        img = torch.from_numpy(img.astype(np.float32))

        pose = torch.from_numpy(pose.astype(np.float32))

        return (img, pose)

class HeadPoseBIWI():
    def __init__(self,
            root_dir: str,
            transform = None,
            target_transform = None,
            download: bool = False,
            seed = None,
            **kwargs,):
        '''
        Downloaded at : https://drive.google.com/file/d/1j6GMx33DCcbUOS8J3NHZ-BMHgk7H-oC_/view?usp=sharing from https://github.com/shamangary/FSA-Net
        Already preprocessed
        '''
        root_dir = os.path.join(root_dir, "HeadPoseBIWI")
        biwi_train_file = np.load(os.path.join(root_dir, "data/BIWI_train.npz"))
        biwi_test_file = np.load(os.path.join(root_dir, "data/BIWI_test.npz"))
        
        self.imgs = biwi_train_file['image']
        self.poses = biwi_train_file['pose']

        self.crop_size = 64
        self.num_examples = self.imgs.shape[0]
        indexes = list(range(self.num_examples))
        np.random.shuffle(indexes)
        indexes_train, indexes_val = indexes[:int(0.8*self.num_examples)], indexes[int(0.8*self.num_examples):]

        self.imgs_val = self.imgs[indexes_val]
        self.poses_val = self.poses[indexes_val]

        self.imgs_train = self.imgs[indexes_train]
        self.poses_train = self.poses[indexes_train]

        self.imgs_test = biwi_test_file['image']
        self.poses_test = biwi_test_file['pose']


        self.dataset_train = DatasetTrainAug(self.imgs_train, self.poses_train)
        self.dataset_test = DatasetTest(self.imgs_test, self.poses_test)
        self.dataset_val = DatasetTest(self.imgs_val, self.poses_val)

    def get_dim_input(self,):
        return (3,64,64)

    def get_dim_output(self,):
        return (3,)
    
    def transform_back(self, x):
        batch_size = x.shape[0]
        x = x.view(batch_size, 1, 64, 64)
        transform_mean = torch.tensor((0.485, 0.456, 0.406)).view(1,1,1).unsqueeze(0).expand(batch_size,1,64,64)
        transform_std = torch.tensor((0.229, 0.224, 0.225)).view(1,1,1).unsqueeze(0).expand(batch_size,1,64,64)
        x_transform = x * transform_std + transform_mean
        return x_transform