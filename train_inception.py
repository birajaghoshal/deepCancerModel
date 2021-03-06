from comet_ml import Experiment

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils
import torch.nn.init as init
from torch.autograd import Variable

import os
import numpy as np
from PIL import Image
from utils.dataloader import *
from utils.auc import *
from utils import new_transforms
import argparse
import random
from utils.inception_custom import *

"""
Options for training
"""

parser = argparse.ArgumentParser()
parser.add_argument('--batchSize', type=int, default=32, help='input batch size')
parser.add_argument('--imgSize', type=int, default=299, help='the height / width of the image to network')
parser.add_argument('--nc', type=int, default=3, help='input image channels (+ concatenated info channels if metadata = True)')
parser.add_argument('--niter', type=int, default=25, help='number of epochs to train for')
parser.add_argument('--lr', type=float, default=0.001, help='learning rate, default=0.001')
parser.add_argument('--dropout', type=float, default=0.5, help='probability of dropout, default=0.5')
parser.add_argument('--beta1', type=float, default=0.5, help='beta1 for adam, default=0.5')
parser.add_argument('--cuda'  , action='store_true', help='enables cuda')
parser.add_argument('--ngpu'  , type=int, default=1, help='number of GPUs to use')
parser.add_argument('--model', default='', help="path to model (to continue training)")
parser.add_argument('--experiment', default=None, help='where to store samples and models')
parser.add_argument('--augment', action='store_true', help='whether to use data augmentation or not')
parser.add_argument('--optimizer',type=str, default='Adam',  help='optimizer: Adam, SGD or RMSprop; default: Adam')
parser.add_argument('--metadata', action='store_true', help='whether to use metadata (default is not)')
parser.add_argument('--init', type=str, default='normal', help='initialization method (normal, xavier, kaiming)')
parser.add_argument('--evalSize', type=int, default=50000, help='number of samples to obtain validation loss on')
parser.add_argument('--nonlinearity', type=str, default='relu', help='nonlinearity to use (selu, prelu, leaky, relu)')
parser.add_argument('--earlystop', action='store_true', help='trigger early stopping (boolean)')
parser.add_argument('--method', type=str, default='average', help='aggregation prediction method (max, average)')
parser.add_argument('--decay_lr', action='store_true', help='activate decay learning rate function')

parser.add_argument('--inception', action='store_true', help='train inception_v3 (shallow training)')

opt = parser.parse_args()
print(opt)

ngpu = int(opt.ngpu)
nc = int(opt.nc)
imgSize = int(opt.imgSize)

experiment = Experiment(api_key="qcf4MjyyOhZj7Xw7UuPvZluts", log_code=True)
hyper_params = vars(opt)
experiment.log_multiple_params(hyper_params)

"""
Save experiment 
"""

if opt.experiment is None:
    opt.experiment = 'samples'

os.system('mkdir experiments')
os.system('mkdir experiments/{0}'.format(opt.experiment))
os.system('mkdir experiments/{0}/images'.format(opt.experiment))

opt.manualSeed = random.randint(1, 10000) # fix seed
print("Random Seed: ", opt.manualSeed)
random.seed(opt.manualSeed)
torch.manual_seed(opt.manualSeed)

cudnn.benchmark = True

###############################################################################

"""
Load data
"""

root_dir = "/beegfs/jmw784/Capstone/LungTilesSorted/"

# Random data augmentation
augment = transforms.Compose([new_transforms.Resize((imgSize, imgSize)),
                              new_transforms.RandomVerticalFlip(),
                              transforms.RandomHorizontalFlip(),
                              new_transforms.RandomRotate(),
                              new_transforms.ColorJitter(0.25, 0.25, 0.25, 0.05),
                              transforms.ToTensor(),
                              transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

transform = transforms.Compose([new_transforms.Resize((imgSize,imgSize)),
                                transforms.ToTensor(),
                                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

data = {}
loaders = {}

print('Loading data ...')
for dset_type in ['train', 'valid']:
    if dset_type == 'train' and opt.augment:
        data[dset_type] = TissueData(root_dir, dset_type, transform = augment, metadata=opt.metadata)
    else:
        data[dset_type] = TissueData(root_dir, dset_type, transform = transform, metadata=opt.metadata)

    loaders[dset_type] = torch.utils.data.DataLoader(data[dset_type], batch_size=opt.batchSize, shuffle=True)
    print('Finished loading %s dataset: %s samples' % (dset_type, len(data[dset_type])))

class_to_idx = data['train'].class_to_idx
classes = data['train'].classes

print('Class encoding:')
print(class_to_idx)

###############################################################################

"""
Model initialization and definition
"""

# Custom weights initialization
if opt.init not in ['normal', 'xavier', 'kaiming']:
    print('Initialization method not found, defaulting to normal')

def init_model(model):

    if opt.inception:
        modules = (p for p in filtered_params(model, last_params))
    else:
        modules = model.modules()

    for m in modules:
        if isinstance(m,nn.Conv2d):
            if opt.init == 'xavier':
                m.weight.data = init.xavier_normal(m.weight.data)
            elif opt.init == 'kaiming':
                m.weight.data = init.kaiming_normal(m.weight.data)
            else:
                m.weight.data.normal_(-0.1, 0.1)
            
        elif isinstance(m,nn.BatchNorm2d):
            m.weight.data.normal_(-0.1, 0.1)

class BasicConv2d(nn.Module):

    def __init__(self, in_channels, out_channels, pool, **kwargs):
        super(BasicConv2d, self).__init__()

        self.pool = pool
        self.conv = nn.Conv2d(in_channels, out_channels, **kwargs)
        self.bn = nn.BatchNorm2d(out_channels, eps=0.001)

        if opt.nonlinearity == 'selu':
            self.relu = nn.SELU()
        elif opt.nonlinearity == 'prelu':
            self.relu = nn.PReLU()
        elif opt.nonlinearity == 'leaky':
            self.relu = nn.LeakyReLU()
        else:
            self.relu = nn.ReLU()

        self.dropout = nn.Dropout(p=opt.dropout)

    def forward(self, x):
        x = self.conv(x)

        if self.pool:
            x = F.max_pool2d(x, 2)
        
        x = self.relu(x)
        x = self.bn(x)
        x = self.dropout(x)
        return x

# Define model
class cancer_CNN(nn.Module):
    def __init__(self, nc, imgSize, ngpu):
        super(cancer_CNN, self).__init__()
        self.nc = nc
        self.imgSize = imgSize
        self.ngpu = ngpu
        self.conv1 = BasicConv2d(nc, 16, False, kernel_size=5, padding=1, stride=2, bias=True)
        self.conv2 = BasicConv2d(16, 32, False, kernel_size=3, bias=True)
        self.conv3 = BasicConv2d(32, 64, True, kernel_size=3, padding=1, bias=True)
        self.conv4 = BasicConv2d(64, 64, True, kernel_size=3, padding=1, bias=True)
        self.conv5 = BasicConv2d(64, 128, True, kernel_size=3, padding=1, bias=True)
        self.conv6 = BasicConv2d(128, 64, True, kernel_size=3, padding=1, bias=True)

        # Three classes
        self.linear = nn.Linear(5184, 3)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = self.conv6(x)
        x = x.view(x.size(0), -1)
        x = self.linear(x)
        return x

###############################################################################

# Create model objects

if opt.inception:
    #Inception objects
    models_to_test = 'inception_v3'
    print('Test print: len classes')
    print(len(classes))
    num_classes = len(classes)
    model_urls = {'inception_v3': 'https://download.pytorch.org/models/inception_v3_google-1a9a5a14.pth' }
    model_names = model_urls.keys()
    # input_sizes = {'inception' : (299,299)}
    last_params = ['AuxLogits.fc.weight', 'AuxLogits.fc.bias', 'fc.weight', 'fc.bias']

if opt.inception:
    print('Loading pre-trained inceotion...')
    model, diff = load_model_merged( models_to_test , num_classes, model_urls, last_params )
    print('Done! (loading pretrained inception)')
else:
    model = cancer_CNN(nc, imgSize, ngpu)

init_model(model)
model.train()

criterion = nn.CrossEntropyLoss()

# Load checkpoint models if needed
if opt.model != '': 
    model.load_state_dict(torch.load(opt.model))
print(model)

if opt.cuda:
    model.cuda()

# Set up 

if opt.inception:
    def in_param_list(s):
        for p in last_params:
            if s.endswith(p):
                return True
        return False

    for p in model.named_parameters():
        p[1].requires_grad = (last_params is None) or in_param_list(p[0])

    params = (p for p in filtered_params(model, last_params))
    parameters = (p[1] for p in params)
else:
    parameters = model.parameters()

if opt.optimizer == "Adam":
    optimizer = optim.Adam( parameters, lr=opt.lr, betas=(opt.beta1, 0.999))
elif opt.optimizer == "RMSprop":
    optimizer = optim.RMSprop(parameters, lr = opt.lr)
elif opt.optimizer == "SGD": 
    optimizer = optim.SGD(parameters, lr = opt.lr)
else: 
    raise ValueError('Optimizer not found. Accepted "Adam", "SGD" or "RMSprop"')

###############################################################################

"""
Evaluation functions
"""

def evaluate(dset_type, sample_size='full'):

    """
    Returns loss for a dataset (train, valid, or test)

    Note: sample_size will be rounded up to be a multiple of the batch_size
    of the dataloader.

    @param dset_type: 'train', 'valid', or 'test'
    @param sample_size: Number of samples to evaluate in the set,
                        'full' means the entire set
    """

    if sample_size == 'full':
        sample_size = len(data[dset_type])
    elif not isinstance(sample_size, int):
        raise ValueError("Amount should be 'full' or an integer")
    elif sample_size > len(data[dset_type]):
        raise ValueError("Amount cannot exceed size of dataset")    

    model.eval()
    loss = 0
    num_evaluated = 0

    for img, label in loaders[dset_type]:

        if opt.cuda:
            img = img.cuda()
            label = label.cuda()

        eval_input = Variable(img, volatile=True)
        eval_label = Variable(label, volatile=True)

        loss += criterion(model(eval_input), eval_label) * img.size(0)

        num_evaluated += img.size(0)

        if num_evaluated >= sample_size:
            return loss / num_evaluated

def get_tile_probability(tile_path):

    """
    Returns an array of probabilities for each class given a tile

    @param tile_path: Filepath to the tile
    @return: A ndarray of class probabilities for that tile
    """

    # Some tiles are empty with no path, return nan
    if tile_path == '':
        return np.full(3, np.nan)

    tile_path = root_dir + tile_path

    with open(tile_path, 'rb') as f:
        with Image.open(f) as img:
            img = img.convert('RGB')

    # Model expects a 4D tensor, unsqueeze first dimension
    img = transform(img).unsqueeze(0)

    if opt.cuda:
        img = img.cuda()

    # Turn output into probabilities with softmax
    var_img = Variable(img, volatile=True)
    output = F.softmax(model(var_img)).data.squeeze(0)

    return output.cpu().numpy()

# Load tile dictionary

with open('/beegfs/jmw784/Capstone/Lung_FileMappingDict.p', 'rb') as f:
    tile_dict = pickle.load(f)

def aggregate(file_list, method):

    """
    Given a list of files, return scores for each class according to the
    method and labels for those files.

    @param file_list: A list of file paths to do predictions on
    @param method: 'average' - returns the average probability score across
                               all tiles for that file
                   'max' - predicts each tile to be the class of the maximum
                           score, and returns the proportion of tiles for
                           each class

    @return: a ndarray of class probabilities for all files in the list
             a ndarray of the labels

    """

    model.eval()
    predictions = []
    true_labels = []

    for file in file_list:
        tile_paths, label = tile_dict[file]

        folder = classes[label]

        def add_folder(tile_path):
            if tile_path == '':
                return ''
            else:
                return folder + '/' + tile_path

        # Add the folder for the class name in front
        add_folder_v = np.vectorize(add_folder)
        tile_paths = add_folder_v(tile_paths)

        # Get the probability array for the file
        prob_v = np.vectorize(get_tile_probability, otypes=[np.ndarray])
        probabilities = prob_v(tile_paths)


        """
        imgSize = probabilities.shape()
        newShape = (imgSize[0], imgSize[1], 3)
        probabilities = np.reshape(np.stack(probabilities.flat), newShape)
        """

        if method == 'average':
            probabilities = np.stack(probabilities.flat)
            prediction = np.nanmean(probabilities, axis = 0)

        elif method == 'max':
            probabilities = np.stack(probabilities.flat)
            probabilities = probabilities[~np.isnan(probabilities).all(axis=1)]
            votes = np.nanargmax(probabilities, axis=1)
            out = np.array([ sum(votes == 0) , sum(votes == 1) , sum(votes == 2)])
            prediction = out / out.sum()

        else:
            raise ValueError('Method not valid')

        predictions.append(prediction)
        true_labels.append(label)

    return np.array(predictions), np.array(true_labels)

###############################################################################

def early_stop(val_history, t=3, required_progress=0.0001):

    """
    Stop the training if there is no non-trivial progress in k steps
    @param val_history: a list contains all the historical validation auc
    @param required_progress: the next auc should be higher than the previous by 
        at least required_progress amount to be non-trivial
    @param t: number of training steps 
    @return: a boolean indicates if the model should early stop
    """
    
    if (len(val_history) > t+1):
        differences = []
        for x in range(1, t+1):
            differences.append(val_history[-x]-val_history[-(x+1)])
        differences = [y < required_progress for y in differences]
        if sum(differences) == t: 
            return True
        else:
            return False
    else:
        return False

if opt.earlystop:
    validation_history = []
else:
    print("No early stopping implemented")
    
stop_training = False

###############################################################################

def adjust_learning_rate(optimizer, epoch):

    """Sets the learning rate to the initial LR decayed by 10 every 3 epochs
        Function copied from: https://github.com/pytorch/examples/blob/master/imagenet/main.py"""
    
    lr = opt.lr * (0.1 ** (epoch // 3)) # Original
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

###############################################################################

"""
Training loop
"""

best_AUC = 0.0

print('Starting training')

for epoch in range(opt.niter+1):
    data_iter = iter(loaders['train'])
    i = 0
    
    if opt.decay_lr:
        adjust_learning_rate(optimizer, epoch)
        print("Epoch %d :lr = %f" % (epoch, optimizer.state_dict()['param_groups'][0]['lr']))    

    while i < len(loaders['train']):
        model.train()
        img, label = data_iter.next()
        i += 1

        # Drop the last batch if it's not the same size as the batchsize
        if img.size(0) != opt.batchSize:
            break

        if opt.cuda:
            img = img.cuda()
            label = label.cuda()


        input_img = Variable(img)
        target_label = Variable(label)

        output_ = model(input_img)
        if isinstance( output_ , tuple):
                train_loss = sum((criterion(o,target_label) for o in output_))
        else:
                train_loss = criterion( output_ , target_label)

        # Zero gradients then backward pass
        optimizer.zero_grad()
        train_loss.backward()

        optimizer.step()

        print('[%d/%d][%d/%d] Training Loss: %f'
               % (epoch, opt.niter, i, len(loaders['train']), train_loss.data[0]))

    # Get validation AUC once per epoch
    val_predictions, val_labels = aggregate(data['valid'].filenames, method=opt.method)
    roc_auc = get_auc('experiments/{0}/images/{1}.jpg'.format(opt.experiment, epoch),
                      val_predictions, val_labels)

    for k, v in roc_auc.items():

        if k in [0, 1, 2]:
            k = classes[k]

        experiment.log_metric("{0} AUC".format(k), v)
        print('%s AUC: %0.4f' % (k, v))

    # Save model if best macro AUC
    if roc_auc['macro'] > best_AUC:
        torch.save(model.state_dict(), 'experiments/{0}/epoch_{1}.pth'.format(opt.experiment, epoch))
        best_AUC = roc_auc['macro']

    # Stop training if no progress on AUC is being made
    if opt.earlystop:
        validation_history.append(roc_auc['macro'])
        stop_training = early_stop(validation_history)

        if stop_training: 
            print("Early stop triggered")
            break

# Final evaluation
train_loss = evaluate('train')
val_loss = evaluate('valid')

print('Finished training, train loss: %f, valid loss: %f, best AUC: %0.4f'
    % (train_loss.data[0], val_loss.data[0], best_AUC))
