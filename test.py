# -*- coding: utf-8 -*-
# also extract features

from __future__ import print_function, division

import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.autograd import Variable
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import time
import os
import scipy.io
from model import resnet50_test

######################################################################
# Options
# --------

# paths = {
#     'market': '/public/users/wanggc/datasets/reid/Market-1501-v15.09.15/pytorch',
#     'duke': '/public/users/wanggc/datasets/reid/DukeMTMC-reID/pytorch',
#     'cuhk03': '/media/data2/xiezhy/data/Dataset/cuhk03-np/pytorch'
# }

class_nums = {
    'market': 751,
    'duke': 702,
    'cuhk03': 767
}

parser = argparse.ArgumentParser(description='Training')
parser.add_argument('--gpu_ids',default='0', type=str,help='gpu_ids: e.g. 0  0,1,2  0,2')
parser.add_argument('--which_epoch',default='last', type=str, help='0,1,2,3...or last')
parser.add_argument('--test_dir',default='/media/data2/xiezhy/data/Dataset/cuhk03-np/pytorch',type=str, help='./test_data')
parser.add_argument('--name', default='ft_ResNet50_scratch', type=str, help='save model path')
parser.add_argument('--batchsize', default=16, type=int, help='batchsize')
parser.add_argument('--pos',default='3', type=str, help='for saving different results')
parser.add_argument('--dataset_name', type=str, help='market, duke, cuhk03')
opt = parser.parse_args()

str_ids = opt.gpu_ids.split(',')
#which_epoch = opt.which_epoch
name = opt.name
test_dir = opt.test_dir
# test_dir = paths[opt.dataset_name]

gpu_ids = []
for str_id in str_ids:
    id = int(str_id)
    if id >=0:
        gpu_ids.append(id)

# set gpu ids
if len(gpu_ids)>0:
   torch.cuda.set_device(gpu_ids[0])

######################################################################
# Load Data
# ---------
#
# We will use torchvision and torch.utils.data packages for loading the
# data.
#
data_transforms = transforms.Compose([
        transforms.Resize((288,144), interpolation=3),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])


data_dir = test_dir
image_datasets = {x: datasets.ImageFolder( os.path.join(data_dir,x) ,data_transforms) for x in ['gallery','query']}
dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=opt.batchsize,
                                             shuffle=False, num_workers=4) for x in ['gallery','query']}

class_names = image_datasets['query'].classes
use_gpu = torch.cuda.is_available()

######################################################################
# Load model
#---------------------------
def load_network(network):
    save_path = os.path.join('./model',name,'net_%s.pth'%opt.which_epoch)
    network.load_state_dict(torch.load(save_path))
    return network


######################################################################

# Extract feature
# ----------------------
#
# Extract feature from  a trained model.
#
def fliplr(img):
    '''flip horizontal'''
    inv_idx = torch.arange(img.size(3)-1,-1,-1).long()  # N x C x H x W
    img_flip = img.index_select(3,inv_idx)
    return img_flip

def extract_feature(model,dataloaders):
    features = torch.FloatTensor()
    count = 0
    for data in dataloaders:
        img, label = data
        n, c, h, w = img.size()
        count += n
        print(count)
        ff = torch.FloatTensor(n,2048).zero_()
        ff2 = torch.FloatTensor(n,64).zero_()
        for i in range(2):
            if(i==1):
                img = fliplr(img)
            input_img = Variable(img.cuda())
            outputs,outputs2 = model(input_img) 
            f = outputs.data.cpu()
            f2 = outputs2.data.cpu()
            ff = ff+f
            ff2 = ff2+f2
       
        # norm feature
        fnorm = torch.norm(ff, p=2, dim=1, keepdim=True)
        ff = ff.div(fnorm.expand_as(ff))

        fnorm = torch.norm(ff2, p=2, dim=1, keepdim=True)
        ff2 = ff2.div(fnorm.expand_as(ff2))

        ff=torch.cat((ff,ff2),1)
        features = torch.cat((features,ff), 0)

    return features

def get_id(img_path):
    camera_id = []
    labels = []
    for path, v in img_path:
        filename = path.split('/')[-1]
        label = filename[0:4]
        camera = filename.split('c')[1]
        if label[0:2]=='-1':
            labels.append(-1)
        else:
            labels.append(int(label))
        camera_id.append(int(camera[0]))
    return camera_id, labels

gallery_path = image_datasets['gallery'].imgs
query_path = image_datasets['query'].imgs

gallery_cam,gallery_label = get_id(gallery_path)
query_cam,query_label = get_id(query_path)

######################################################################
# Load Collected data Trained model
print('-------test-----------')
print(len(class_names))
class_num = class_nums[opt.dataset_name]
model_structure = resnet50_test(class_num)   
print('***************************resnet************************')
model = load_network(model_structure)

# Remove the final fc layer and classifier layer

model.fc1 = nn.Sequential()
model.fc2 = nn.Sequential()
model.bn2 = nn.Sequential()
model.relu2 = nn.Sequential()
model.dropout2 = nn.Sequential()

model.fc1_2 = nn.Sequential()
model.fc2_2 = nn.Sequential()
model.bn2_2 = nn.Sequential()
model.relu2_2 = nn.Sequential()
model.dropout2_2 = nn.Sequential()


print('model:',model)

# Change to test mode
model = model.eval()
# model = model.train(True)
if use_gpu:
    model = model.cuda()

# Extract feature
gallery_feature = extract_feature(model,dataloaders['gallery'])
query_feature = extract_feature(model,dataloaders['query'])


# Save to Matlab for check
result = {'gallery_f':gallery_feature.numpy(),'gallery_label':gallery_label,'gallery_cam':gallery_cam,'query_f':query_feature.numpy(),'query_label':query_label,'query_cam':query_cam}
scipy.io.savemat('model/'+name+'/'+'pytorch_result'+opt.pos+'.mat',result)