import time
import os
from glob import glob

import numpy as np
import torch
import torch.optim as optim
from torch import nn
from torch.autograd import Variable

import torchvision
import torchvision.models as models
import torch.utils.model_zoo as model_zoo
import torchvision.transforms as transforms
from torchvision import datasets

from itertools import accumulate
from functools import reduce

from utils.CLR import *

# New imports
from PIL import Image
from utils.dataloader import *
from utils.auc import *
from utils import new_transforms
import argparse
import random



model_urls = {
    'inception_v3': 'https://download.pytorch.org/models/inception_v3_google-1a9a5a14.pth'  
}

model_names = model_urls.keys()

input_sizes = {
    'inception' : (299,299)
}

models_to_test = ['inception_v3']


#Fix this directories
#####
data_dir = '/beegfs/jmw784/Capstone/LungTilesSorted/'

# train_subfolder = os.path.join(data_dir, 'train')

#####

classes = [0,1,2] # [d.split(train_subfolder, 1)[1] for d in \
          #  glob(os.path.join(train_subfolder, '**'))]

last_params = ['AuxLogits.fc.weight', 'AuxLogits.fc.bias', 'fc.weight', 'fc.bias']

batch_size = 30
epoch_multiplier = 8 #per class and times 1(shallow), 2(deep), 4(from_scratch)
use_gpu = torch.cuda.is_available()
use_clr = True

#Assume 50 examples per class and CLR authors' middle ground
clr_stepsize = (len(classes)*50//batch_size)*4

print("Shootout of model(s) %s with batch_size %d running on CUDA %s " % \
            (", ".join(models_to_test), batch_size, use_gpu) + \
            "with CLR %s for %d classes on data in %s." % \
            (use_clr, len(classes), data_dir))


###### Generic pretrained model loading


def diff_states(dict_canonical, dict_subset):
    names1, names2 = (list(dict_canonical.keys()), list(dict_subset.keys()))
    
    #Sanity check that param names overlap
    #Note that params are not necessarily in the same order
    #for every pretrained model
    not_in_1 = [n for n in names1 if n not in names2]
    not_in_2 = [n for n in names2 if n not in names1]
    assert len(not_in_1) == 0
    assert len(not_in_2) == 0
    for name, v1 in dict_canonical.items():

        if name in last_params:
            break

        v2 = dict_subset[name]
        assert hasattr(v2, 'size')
        if v1.size() != v2.size():
            print(name,v1.size(),v2.size())
            yield (name, v1)                

def load_model_merged(name, num_classes):
    model = models.__dict__[name](num_classes=num_classes)
    model_dict = model.state_dict()
    pretrained_state = model_zoo.load_url(model_urls[name])

    #Diff
    diff = [s for s in diff_states(model.state_dict(), pretrained_state)]
    print("Replacing the following state from initialized", name, ":", [d[0] for d in diff])
    for name, value in diff:
        pretrained_state[name] = value
        assert len([s for s in diff_states(model.state_dict(), pretrained_state)]) == 0

    # Remove last layer weights because different number of classes
    for name in last_params:
        del pretrained_state[name]
    
    #Update weights from pretrained state
    model_dict.update(pretrained_state)

    #Merge
    model.load_state_dict(model_dict)
    return model, diff


def filtered_params(net, param_list=None):
    def in_param_list(s):
        for p in param_list:
            if s.endswith(p):
                return True
        return False    
    #Caution: DataParallel prefixes '.module' to every parameter name
    params = net.named_parameters() if param_list is None \
    else (p for p in net.named_parameters() if \
          in_param_list(p[0]) and p[1].requires_grad)
    return params


#### Training and Evaluation

def get_data(resize):
    '''

    dsets = {x: datasets.ImageFolder(os.path.join(data_dir, x), data_transforms[x])
             for x in ['train', 'val']}
    '''

    augment = transforms.Compose([
        transforms.RandomSizedCrop(max(resize)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

    transform = transforms.Compose([
        #Higher scale-up for inception
        transforms.Scale(int(max(resize)/224*256)),
        transforms.CenterCrop(max(resize)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

    dsets = {}

    for dset_type in ['train', 'val']:
        if dset_type == 'train':
            dsets[dset_type] = TissueData(data_dir, dset_type, transform = augment , metadata=False) 
        else:
            dsets[dset_type] = TissueData(data_dir, dset_type, transform = transform , metadata=False) 


    dset_loaders = {x: torch.utils.data.DataLoader(dsets[x], batch_size=batch_size, shuffle=True) for x in ['train', 'val']}
    dset_sizes = {x: len(dsets[x]) for x in ['train', 'val']}
    dset_classes = dsets['train'].classes
    
    return dset_loaders['train'], dset_loaders['val']


def train(net, trainloader, epochs, param_list=None, CLR=False):
    #Todo: DRY
    def in_param_list(s):
        for p in param_list:
            if s.endswith(p):
                return True
        return False

    criterion = nn.CrossEntropyLoss()
    if use_gpu:
        criterion = criterion.cuda()
    
    #If finetuning model, turn off grad for other params and make sure to turn on others
    for p in net.named_parameters():
        p[1].requires_grad = (param_list is None) or in_param_list(p[0])

    params = (p for p in filtered_params(net, param_list))

    #Optimizer as in tutorial
    optimizer = optim.SGD((p[1] for p in params), lr=0.001, momentum=0.9)
    if CLR:
            
        global clr_stepsize
        clr_wrapper = CyclicLR(optimizer, step_size=clr_stepsize)
    
    losses = []
    for epoch in range(epochs):

        running_loss = 0.0
        for i, data in enumerate(trainloader, 0):
            # get the inputs
            inputs, labels = data
            if use_gpu:
                inputs, labels = Variable(inputs.cuda()), Variable(labels.cuda(async=True))
            else:
                inputs, labels = Variable(inputs), Variable(labels)

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = net(inputs)
            
            loss = None
            # for nets that have multiple outputs such as inception
            if isinstance(outputs, tuple):
                loss = sum((criterion(o,labels) for o in outputs))
            else:
                loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            if CLR:
                clr_wrapper.batch_step()
            
            # print statistics
            running_loss += loss.data[0]
            if i % 30 == 29:
                avg_loss = running_loss / 30
                losses.append(avg_loss)
                
                lrs = [p['lr'] for p in optimizer.param_groups]
                    
                print('[%d, %5d] loss: %.3f' %
                      (epoch + 1, i + 1, avg_loss), lrs)
                running_loss = 0.0

    print('Finished Training')
    return losses

#Get stats for training and evaluation in a structured way
#If param_list is None all relevant parameters are tuned,
#otherwise, only parameters that have been constructed for custom
#num_classes
def train_stats(m, trainloader, epochs, param_list=None, CLR=False):
    stats = {}
    params = filtered_params(m, param_list)    
    counts = 0,0
    for counts in enumerate(accumulate((reduce(lambda d1,d2: d1*d2, p[1].size()) for p in params)) ):
        pass
    stats['variables_optimized'] = counts[0] + 1
    stats['params_optimized'] = counts[1]
    
    before = time.time()
    losses = train(m, trainloader, epochs, param_list=param_list, CLR=CLR)
    stats['training_time'] = time.time() - before

    stats['training_loss'] = losses[-1] if len(losses) else float('nan')
    stats['training_losses'] = losses
    
    return stats

def evaluate_stats(net, testloader):
    stats = {}
    correct = 0
    total = 0
    
    before = time.time()
    for i, data in enumerate(testloader, 0):
        images, labels = data

        if use_gpu:
            images, labels = (images.cuda()), (labels.cuda(async=True))

        outputs = net(Variable(images))
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum()
    accuracy = correct / total
    stats['accuracy'] = accuracy
    stats['eval_time'] = time.time() - before
    
    print('Accuracy on test images: %f' % accuracy)
    return stats


def train_eval(net, trainloader, testloader, epochs, param_list=None, CLR=False):
    print("Training..." if not param_list else "Retraining...")
    stats_train = train_stats(net, trainloader, epochs, param_list=param_list, CLR=CLR)
    
    print("Evaluating...")
    net = net.eval()
    stats_eval = evaluate_stats(net, testloader)
    
    return {**stats_train, **stats_eval}


if __name__=='__main__':
    stats = []
    num_classes = len(classes)
    print(num_classes)   
 
    epochs = num_classes * epoch_multiplier * 1
    print("RETRAINING %d epochs" % epochs)

    for name in models_to_test:
        print("")
        print("Targeting %s with %d classes" % (name, num_classes))
        print("------------------------------------------")
        model_pretrained, diff = load_model_merged(name, num_classes)
        final_params = [d[0] for d in diff]

        resize = [s[1] for s in input_sizes.items() if s[0] in name][0]
        print("Resizing input images to max of", resize)
        
        trainloader, testloader = get_data(resize)

        if use_gpu:
            print("Transfering models to GPU(s)")
            model_pretrained = torch.nn.DataParallel(model_pretrained).cuda()

        pretrained_stats = train_eval(model_pretrained, 
                                      trainloader, testloader, epochs,
                                      final_params, CLR=use_clr)
        pretrained_stats['name'] = name
        pretrained_stats['retrained'] = True
        pretrained_stats['shallow_retrain'] = True
        stats.append(pretrained_stats)

        print("")





