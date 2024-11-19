#-*- coding:utf-8 -*-

import os.path as osp
import sys
sys.path.append('..')
import numpy as np
from torch.utils.data import Dataset
from clip_hdr11.utils.utils import *
from .modules import *

def rgb_raw(imgs):
    result=[]
    for i in range(len(imgs)):
        img=imgs[i]
        img_8bit = (img * 255)
        result.append(img_8bit)


    return result

def turn2mask(img):


    img_2channel = np.zeros((2, img.shape[0], img.shape[1]), dtype=np.float32)
    img_2channel[0, ...] = np.where(img == 255, 1, 0)
    img_2channel[1, ...] = np.where(img == 0, 1, 0)
    return img_2channel
class SIG17_Validation_Dataset(Dataset):

    def __init__(self, root_dir, is_training=False, crop=True, crop_size=512):
        self.root_dir = root_dir
        self.is_training = is_training
        self.crop = crop
        self.crop_size = crop_size

        # sample dir
        self.scenes_dir = osp.join(root_dir, 'Test')
        self.scenes_list = sorted(os.listdir(self.scenes_dir))

        self.image_list = []
        for scene in range(len(self.scenes_list)):
            exposure_file_path = os.path.join(self.scenes_dir, self.scenes_list[scene], 'exposure.txt')
            ldr_file_path = list_all_files_sorted(os.path.join(self.scenes_dir, self.scenes_list[scene]), '.tif')
            label_path = os.path.join(self.scenes_dir, self.scenes_list[scene])
            self.image_list += [[exposure_file_path, ldr_file_path, label_path]]

    def __getitem__(self, index):
        # Read exposure times
        expoTimes = read_expo_times(self.image_list[index][0])
        # Read LDR images
        ldr_images = read_images(self.image_list[index][1])
        # Read HDR label
        label = read_label(self.image_list[index][2], 'HDRImg.hdr') # 'HDRImg.hdr' for test data
        # ldr images process
        pre_img0 = ldr_to_hdr(ldr_images[0], expoTimes[0], 2.2)
        pre_img1 = ldr_to_hdr(ldr_images[1], expoTimes[1], 2.2)
        pre_img2 = ldr_to_hdr(ldr_images[2], expoTimes[2], 2.2)
        edge=rgb_raw(ldr_images)
        # concat: linear domain + ldr domain
        pre_img0 = np.concatenate((pre_img0, ldr_images[0]), 2)
        pre_img1 = np.concatenate((pre_img1, ldr_images[1]), 2)
        pre_img2 = np.concatenate((pre_img2, ldr_images[2]), 2)

        if self.crop:
            x = 0
            y = 0
            img0 = pre_img0[x:x + self.crop_size, y:y + self.crop_size].astype(np.float32).transpose(2, 0, 1)
            img1 = pre_img1[x:x + self.crop_size, y:y + self.crop_size].astype(np.float32).transpose(2, 0, 1)
            img2 = pre_img2[x:x + self.crop_size, y:y + self.crop_size].astype(np.float32).transpose(2, 0, 1)
            label = label[x:x + self.crop_size, y:y + self.crop_size].astype(np.float32).transpose(2, 0, 1)
        else:
            img0 = pre_img0.astype(np.float32).transpose(2, 0, 1)
            img1 = pre_img1.astype(np.float32).transpose(2, 0, 1)
            img2 = pre_img2.astype(np.float32).transpose(2, 0, 1)
            label = label.astype(np.float32).transpose(2, 0, 1)

        img0 = torch.from_numpy(img0)
        img1 = torch.from_numpy(img1)
        img2 = torch.from_numpy(img2)
        label = torch.from_numpy(label)
        edge1=torch.from_numpy(edge[0])
        edge2=torch.from_numpy(edge[1])
        edge3=torch.from_numpy(edge[2])
        edge1=edge1.permute(2,0,1)
        edge2=edge2.permute(2,0,1)
        edge3=edge3.permute(2,0,1)
        sample = {
            'input0': img0,
            'input1': img1,
            'input2': img2,
            'label': label,
            'edge1':edge1
            ,'edge2':edge2
            ,'edge3':edge3
            }
        return sample

    def __len__(self):
        return len(self.scenes_list)





