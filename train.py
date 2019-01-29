# -*- coding: utf-8 -*-

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
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
from PIL import Image
import time
import os
from model import resnet50
from random_erasing import RandomErasing
import json

######################################################################
# Options
# --------
# paths = {
#     'market': '/public/users/wanggc/datasets/reid/Market-1501-v15.09.15/pytorch',
#     'duke': '/public/users/wanggc/datasets/reid/DukeMTMC-reID/pytorch',
#     'cuhk03': '/media/data2/xiezhy/data/Dataset/cuhk03-np/pytorch'
# }

parser = argparse.ArgumentParser(description='Training')
parser.add_argument('--gpu_ids',default='0', type=str,help='gpu_ids: e.g. 0  0,1,2  0,2')
parser.add_argument('--name',default='ft_ResNet50_scratch', type=str, help='output model name')
parser.add_argument('--data_dir',default='/home/user/dataset/pytorch',type=str, help='training dir path')
parser.add_argument('--baseline', action='store_true', help='train the baseline network')
parser.add_argument('--train_all', action='store_true', help='use all training data' )
parser.add_argument('--batchsize', default=32, type=int, help='batchsize')
parser.add_argument('--erasing_p', default=0, type=float, help='Random Erasing probability, in [0,1]')
parser.add_argument('--lambda', default=0, type=float, help='lamada for two traing loss weighting')
parser.add_argument('--start_point', default=40, type=int, help='start_point for dropping lr')
parser.add_argument('--end_point', default=70, type=int, help='end_point for stopping the algorithm')
# parser.add_argument('--dataset_name', type=str)
opt = parser.parse_args()

data_dir = opt.data_dir
# data_dir = paths[opt.dataset_name]
name = opt.name
str_ids = opt.gpu_ids.split(',')
gpu_ids = []
for str_id in str_ids:
    gid = int(str_id)
    if gid >=0:
        gpu_ids.append(gid)

# set gpu ids
if len(gpu_ids)>0:
    torch.cuda.set_device(gpu_ids[0])

def print_config():
    print('**********************************************')
    print('data_dir:',data_dir)
    print('gpu_ids,name,batchsize,lamada,start_point,end_point:',opt.gpu_ids,opt.name,opt.batchsize,opt.lamada,opt.start_point,opt.end_point)

######################################################################
# Load Data
transform_train_list = [
        transforms.Resize((288,144), interpolation=3),
        transforms.RandomCrop((256,128)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]

if opt.erasing_p>0:
    transform_train_list = transform_train_list + [RandomErasing(opt.erasing_p)]

print(transform_train_list)

transform_val_list = [
        transforms.Resize(size=(256,128),interpolation=3), #Image.BICUBIC
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]


data_transforms = {
    'train': transforms.Compose(transform_train_list),
    'val': transforms.Compose(transform_val_list),
}

train_all = ''
if opt.train_all:
     train_all = '_all'

image_datasets = {}
image_datasets['train'] = datasets.ImageFolder(os.path.join(data_dir, 'train' + train_all),
                                          data_transforms['train'])
image_datasets['val'] = datasets.ImageFolder(os.path.join(data_dir, 'val'),
                                          data_transforms['val'])

dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=opt.batchsize,
                                             shuffle=True, num_workers=4)
              for x in ['train', 'val']}
dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}
class_names = image_datasets['train'].classes

use_gpu = torch.cuda.is_available()

inputs, classes = next(iter(dataloaders['train']))

######################################################################
# Training the model
# ------------------
#
# Now, let's write a general function to train a model. Here, we will
# illustrate:
#
# -  Scheduling the learning rate
# -  Saving the best model
#
# In the following, parameter ``scheduler`` is an LR scheduler object from
# ``torch.optim.lr_scheduler``.

y_loss = {} # loss history
y_loss['train'] = []
y_loss['val'] = []
y_err = {}
y_err['train'] = []
y_err['val'] = []

def train_model(model, criterion, optimizer, scheduler, num_epochs=25, baseline=False):
    since = time.time()

    best_model_wts = model.state_dict()
    best_acc = 0.0

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                scheduler.step()
                model.train(True)  # Set model to training mode
            else:
                model.train(False)  # Set model to evaluate mode

            running_loss = 0.0
            running_loss1 = 0.0
            running_loss2 = 0.0
            running_corrects = 0
            running_corrects1 = 0
            running_corrects2 = 0

            # Iterate over data.
            for data in dataloaders[phase]:
                # get the inputs
                inputs, labels = data
                         
                # wrap them in Variable
                if use_gpu:
                    inputs = Variable(inputs.cuda())
                    labels = Variable(labels.cuda())
                else:
                    inputs, labels = Variable(inputs), Variable(labels)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                if not baseline:
                    outputs,outputs2 = model(inputs)

                    _, preds = torch.max(outputs.data, 1)
                    loss1 = criterion(outputs, labels)
                    
                    _, preds2 = torch.max(outputs2.data, 1)
                    loss2 = criterion(outputs2, labels)

                    loss=(1-opt.lamada)*loss1+opt.lamada*loss2
                else:
                    outputs = model(inputs)
                    _, preds = torch.max(outputs.data, 1)
                    loss1 = criterion(outputs, labels)

                    loss=loss1
                # backward + optimize only if in training phase
                if phase == 'train':
                    loss.backward()
                    optimizer.step()

                # statistics
                running_loss += loss.data.item()
                running_loss1 += loss1.data.item()
                running_corrects += torch.sum(preds == labels.data).item()
                running_corrects1 += torch.sum(preds == labels.data).item()
                if not baseline:
                    running_loss2 += loss2.data.item()
                    running_corrects2 += torch.sum(preds2 == labels.data).item()

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_loss1 = running_loss1 / dataset_sizes[phase]

            epoch_acc = running_corrects / dataset_sizes[phase]
            epoch_acc1 = running_corrects1 / dataset_sizes[phase]

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc))
            print('{} Loss1: {:.4f} Acc1: {:.4f}'.format(phase, epoch_loss1, epoch_acc1))

            if not baseline:
                epoch_loss2 = running_loss2 / dataset_sizes[phase]
                epoch_acc2 = running_corrects2 / dataset_sizes[phase]
                print('{} Loss2: {:.4f} Acc2: {:.4f}'.format(phase, epoch_loss2, epoch_acc2))
            
            y_loss[phase].append(epoch_loss1)
            y_err[phase].append(1.0-epoch_acc1)            
            # deep copy the model
            if phase == 'val':
                last_model_wts = model.state_dict()
                if epoch%10 == 9:
                    save_network(model, epoch)
                draw_curve(epoch)
        print_config()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    #print('Best val Acc: {:4f}'.format(best_acc))

    # load best model weights
    model.load_state_dict(last_model_wts)
    save_network(model, 'last')


    with open('./model/'+name+'/log.txt', 'w') as f:
        f.write('train_loss')
        for loss in y_loss['train']:
            f.write(' ' + str(loss))
        f.write('\n')

        f.write('val_loss')
        for loss in y_loss['val']:
            f.write(' '+str(loss))
        f.write('\n')

        f.write('train_error')
        for error in y_err['train']:
            f.write(' '+str(error))
        f.write('\n')

        f.write('val_loss')
        for error in y_err['val']:
            f.write(' '+str(error))

    return model


######################################################################
# Draw Curve
#---------------------------
x_epoch = []
fig = plt.figure()
ax0 = fig.add_subplot(121, title="loss")
ax1 = fig.add_subplot(122, title="top1err")
def draw_curve(current_epoch):
    x_epoch.append(current_epoch)
    ax0.plot(x_epoch, y_loss['train'], 'bo-', label='train')
    ax0.plot(x_epoch, y_loss['val'], 'ro-', label='val')
    ax1.plot(x_epoch, y_err['train'], 'bo-', label='train')
    ax1.plot(x_epoch, y_err['val'], 'ro-', label='val')
    if current_epoch == 0:
        ax0.legend()
        ax1.legend()
    fig.savefig( os.path.join('./model',name,name+'_train.jpg'))

######################################################################
# Save model
#---------------------------
def save_network(network, epoch_label):
    save_filename = 'net_%s.pth'% epoch_label
    save_path = os.path.join('./model',name,save_filename)
    torch.save(network.cpu().state_dict(), save_path)
    if torch.cuda.is_available:
        network.cuda(gpu_ids[0])


######################################################################
# Finetuning the convnet
# ----------------------
#
# Load a pretrainied model and reset final fully connected layer.


model = resnet50(len(class_names), opt.baseline)    
print('***************************net**********************************')
print(model)

if use_gpu:
    model = model.cuda()

criterion = nn.CrossEntropyLoss()
criterion2 = nn.CrossEntropyLoss()

ignored_params = list(map(id, model.fc1.parameters() )) + list(map(id, model.fc2.parameters() ))+list(map(id, model.fc1_2.parameters() ))+list(map(id, model.fc2_2.parameters() ))
base_params = filter(lambda p: id(p) not in ignored_params, model.parameters())

# Observe that all parameters are being optimized
optimizer_ft = optim.SGD([
             {'params': base_params, 'lr': 0.01},
             {'params': model.fc1.parameters(), 'lr': 0.1},
             {'params': model.fc2.parameters(), 'lr': 0.1},
             {'params': model.fc1_2.parameters(), 'lr': 0.1},
             {'params': model.fc2_2.parameters(), 'lr': 0.1},
         ], momentum=0.9, weight_decay=5e-4, nesterov=True)

# Decay LR by a factor of 0.1 every 40 epochs
# exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=40, gamma=0.1)
exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=opt.start_point, gamma=0.1)

######################################################################
# Train and evaluate
# ^^^^^^^^^^^^^^^^^^
#
# It should take around 15-25 min on CPU. On GPU though, it takes less than a
# minute.
#
dir_name = os.path.join('./model',name)
if not os.path.isdir(dir_name):
    os.mkdir(dir_name)

# save opts
with open('%s/opts.json'%dir_name,'w') as fp:
    json.dump(vars(opt), fp, indent=1)

model = train_model(model, criterion, optimizer_ft, exp_lr_scheduler, num_epochs=opt.end_point, baseline=opt.baseline)
