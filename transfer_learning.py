
# Spyros Kasapis, PhD Student
# University of Michigan, Department of Naval Architecture & Marine Engineering
# Friday, September 18, 2020

# Code does:
# Gets an unsupervised (FB DeepClustering) trained AlexNet and Transfer Learns its Classifier in a supervised manner
# Dataset used contains 5 Classes, 20 pictures each for training - 5 Classes, 5 pictures each for Validation

import argparse
import os
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data
import torchvision
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torch.backends.cudnn as cudnn
from util import AverageMeter, load_model, Logger
from my_evaldataset import My_ImageFolder

# Initialization
model_file = '/home/skasapis/Desktop/original-deepcluster-master/feats/checkpoint.pth.tar'
data_dir = '~/Desktop/MarkdownFiles/custom_clean/'

# Transforms
mean = np.array([0.5, 0.5, 0.5])
std = np.array([0.25, 0.25, 0.25])
data_transforms = {
    'train': transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ]),
    'val': transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ]),
}

image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x), data_transforms[x]) for x in ['train', 'val']}
dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=4, shuffle=True, num_workers=0) for x in ['train', 'val']}
dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}
class_names = image_datasets['train'].classes
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def main():
    
    with open("spyros_custom_labels_5.txt") as f:
        classes_wanted = eval(f.read())
    classes_list = [x for x in classes_wanted]      
    
    # Get Model
    model = load_model(model_file)

    in_feat = model.classifier[1].in_features
    out_feat = model.classifier[1].out_features

    # Freeze trained layers
    for param in model.parameters():
        param.requires_grad = False

    # Set up new layers
    model.classifier[1] = nn.Linear(in_feat,out_feat)
    model.classifier[4] = nn.Linear(out_feat,out_feat)
    model.top_layer = nn.Linear(out_feat,len(class_names))

    # Set Optimizer and Criterion
    optimizer = optim.SGD(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()

    # Train Model 
    model,losses,accs = train_model(model,criterion, optimizer)
    
    # Evaluate Model
    evaluate(model,losses,accs)


def train_model(model,criterion,optimizer,num_epochs=25):

    all_loss = []
    all_accuracy = []

    for epoch in range(num_epochs):

        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)
        model.train()

        running_loss = 0.0
        running_corrects = 0.0

        # Iterate over data.
        for inputs, labels in dataloaders['train']:
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            loss = criterion(outputs, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Statistics
            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data)

        epoch_loss = running_loss / dataset_sizes['train']
        epoch_acc = running_corrects.double() / dataset_sizes['train']

        print('{} Loss: {:.4f} Acc: {:.4f}'.format('train', epoch_loss, epoch_acc))

        all_loss.append(epoch_loss)
        all_accuracy.append(epoch_acc)
    
    return(model,all_loss,all_accuracy)


def evaluate(model,losses,accs):

    # Evaluation
    model.eval()
    all_labels = []
    all_preds = []
    for inputs, labels in dataloaders['val']:
        outputs = model(inputs)
        _, preds = torch.max(outputs, 1) # stoped here
        all_labels = np.concatenate((all_labels,labels.numpy()),axis=0)
        all_preds = np.concatenate((all_preds,preds.numpy()),axis=0)
    compare = all_labels - all_preds
    correct = len(np.where(compare == 0)[0])
    percentage_correct = correct/len(compare)*100
    print('The Evaluation Accuracy is: {}%'.format(percentage_correct))

    # Graphing
    plt.subplot(2, 1, 1)
    plt.plot(np.arange(len(losses)), np.array(losses))
    plt.ylabel('Train Loss')

    plt.subplot(2, 1, 2)
    plt.plot(np.arange(len(accs)), np.array(accs))
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')  

    plt.show() #savefig('~/Desktop/deepcluster-fully-unsupervised/results/TLLosses.png')

main()

