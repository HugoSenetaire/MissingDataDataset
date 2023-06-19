
import os
import torch
import torchvision
import h5py
import numpy as np
import pickle
from ..custom_image_dataset import CustomImageDataset
import copy


def predump_file(root_dir,):
    '''
    Downloaded at : https://1drv.ms/u/s!Arj2pETbYnWQstIt73ZfGOAjBMiTmQ?e=cvxFIN from https://github.com/UBCDingXin/improved_CcGAN/tree/master
    '''
    file_path = os.path.join(root_dir, 'Cell200_64x64.h5')
    hf = h5py.File(file_path, 'r')
    
    hf = h5py.File(file_path, 'r')
    counts = hf['CellCounts'][:]
    counts = counts.astype(float)
    images = hf['IMGs_grey'][:]
    hf.close()

    images = copy.deepcopy(images)
    print("Original shape")
    print(images.shape)
    counts = copy.deepcopy(counts)
    print("Original shape")
    print(counts.shape)

    num_examples = len(images)
    print (num_examples)
    index_images = list(range(num_examples))
    np.random.shuffle(index_images)
    np.random.shuffle(index_images)
    np.random.shuffle(index_images)
    np.random.shuffle(index_images)

    num_imgs_train = 10000
    num_imgs_test = 10000
    nums_imgs_val = 2000
    index_images_train = index_images[0:num_imgs_train]
    index_images_test = index_images[num_imgs_train:(num_imgs_train+num_imgs_test)]
    index_images_val = index_images[(num_imgs_train+num_imgs_test):(num_imgs_train+num_imgs_test+nums_imgs_val)]
    print (len(index_images_train))
    print (len(index_images_test))
    print (len(index_images_val))

    images_train = images[index_images_train].reshape(-1,1,64,64).repeat(3, axis=1)
    labels_train = counts[index_images_train].reshape(-1,1)
    images_val = images[index_images_val].reshape(-1,1,64,64).repeat(3, axis=1)
    labels_val = counts[index_images_val].reshape(-1,1)
    images_test = images[index_images_test].reshape(-1,1,64,64).repeat(3, axis=1)
    labels_test = counts[index_images_test].reshape(-1,1)


    print (labels_train.shape)
    print (images_train.shape)
    print (labels_test.shape)
    print (images_test.shape)
    with open(os.path.join(root_dir,"labels_train.pkl"), "wb") as file:
        pickle.dump(labels_train, file)
    with open(os.path.join(root_dir,"images_train.pkl"), "wb") as file:
        pickle.dump(images_train, file)
    with open(os.path.join(root_dir,"labels_val.pkl"), "wb") as file:
        pickle.dump(labels_val, file)
    with open(os.path.join(root_dir,"images_val.pkl"), "wb") as file:
        pickle.dump(images_val, file)
    with open(os.path.join(root_dir,"labels_test.pkl"), "wb") as file:
        pickle.dump(labels_test, file)
    with open(os.path.join(root_dir,"images_test.pkl"), "wb") as file:
        pickle.dump(images_test, file)


    

default_UTK_transform = torchvision.transforms.Compose([
                                    torchvision.transforms.ToTensor(),
                                    torchvision.transforms.Normalize(
                                        (0.485, 0.456, 0.406), (0.229, 0.224, 0.225,))
                                    ])


class CellCount():
    def __init__(self,
            root_dir: str,
            transform = default_UTK_transform,
            target_transform = None,
            download: bool = False,
            seed = None,
            **kwargs,):
        root_dir = os.path.join(root_dir, "CellCount")
        label_train_path = os.path.join(os.path.join(root_dir,"labels_train.pkl"))
        label_test_path = os.path.join(os.path.join(root_dir,"labels_test.pkl"))
        label_val_path = os.path.join(os.path.join(root_dir,"labels_val.pkl"))
        
        data_train_path = os.path.join(os.path.join(root_dir, "images_train.pkl"))
        data_test_path = os.path.join(os.path.join(root_dir, "images_test.pkl"))
        data_val_path = os.path.join(os.path.join(root_dir, "images_val.pkl"))
        for path in [label_train_path, label_test_path, data_train_path, data_test_path]:
            if not os.path.exists(path):
                print(f"{path} not found")
                predump_file(root_dir=root_dir)  
                break
        with open(data_train_path, "rb") as data_train_path:
            data_train = pickle.load(data_train_path)
        with open(label_train_path, "rb") as label_train_path:
            label_train = pickle.load(label_train_path)
        with open(data_test_path, "rb") as data_test_path:
            data_test = pickle.load(data_test_path)
        with open(label_test_path, "rb") as label_test_path:
            label_test = pickle.load(label_test_path)
        with open(data_val_path, "rb") as data_val_path:
            data_val = pickle.load(data_val_path)
        with open(label_val_path, "rb") as label_val_path:
            label_val = pickle.load(label_val_path)
        print(data_train.shape)
        self.dataset_train = CustomImageDataset(data_train.transpose(0,2,3,1), label_train, transform=transform, target_transform=target_transform)
        self.dataset_test = CustomImageDataset(data_test.transpose(0,2,3,1), label_test, transform=transform, target_transform=target_transform)
        self.dataset_val = CustomImageDataset(data_val.transpose(0,2,3,1), label_val, transform=transform, target_transform=target_transform)

    def get_dim_input(self,):
        return (3,64,64)

    def get_dim_output(self,):
        return (1,)
    
    def transform_back(self, x):
        batch_size = x.shape[0]
        x = x.view(batch_size, 3, 64, 64)
        transform_mean = torch.tensor((0.485, 0.456, 0.406)).view(3,1,1).unsqueeze(0).expand(batch_size,3,64,64)
        transform_std = torch.tensor((0.229, 0.224, 0.225)).view(3,1,1).unsqueeze(0).expand(batch_size,3,64,64)
        x_transform = x * transform_std + transform_mean
        return x_transform