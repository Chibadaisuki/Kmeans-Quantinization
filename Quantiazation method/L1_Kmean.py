import torch
import random
import os
import sys
import numpy as np
from PIL import Image
import pandas as pd
import torchvision
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import torch.optim as optim
from sklearn.metrics import accuracy_score
from tqdm import tqdm
import time
# import matplotlib.pyplot as plt
import time
import threading

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"]="4"
cuda = torch.cuda.is_available()
device = torch.device("cuda" if cuda else "cpu")
num_workers = 4 if cuda else 8
print("Cuda = "+str(cuda)+" with num_workers = "+str(num_workers))
# sys.version
# print(cuda, sys.version)

def load_data():
    grad_path = './grad.pth'
    grad = torch.load(grad_path, map_location = torch.device('cpu'))
    weight_path = './cnn'
    weight = torch.load(weight_path, map_location = torch.device('cpu'))
    # weight = weight['model_state_dict']
    return grad, weight

# Make x in the shape [weight, dL/dx]
def clustering_loss(x, center):
    # grad = x[1]
    # value = x[0]
    return abs(x[1] * (x[0]-center))

def centering_loss(x_list, center):
    loss = 0
    for x in x_list:
        loss += clustering_loss(x, center)
    return loss

# x_list, list of x shaped [weight, dL/dx]
def find_center_mht(x_list):
    # Increasing order in terms of gradient
    x_list.sort(key = lambda x: x[1])
    grads = [i[1] for i in x_list]
    weights = [i[0] for i in x_list]

    total_grad = sum(grads)
    tmp_grad = 0 
    idx = 0
    for i in range(1, len(grads)):
        idx = i
        tmp_grad += grads[i]
        later_grad = total_grad - tmp_grad
        if tmp_grad > later_grad:
            break
    
    fore_idx = idx - 1
    late_idx = idx

    return (weights[fore_idx] + weights[late_idx])/2

def find_center_mean(x_list):
    weights = [i[0] for i in x_list]
    return sum(weights)/len(weights)

def init_kmean(cur_weight, k):
    length = len(cur_weight)
    for i in range(k):
        idxes = random.sample(range(length), k)
        centers = [cur_weight[i] for i in idxes]
    return centers

def kmean_iter(x, center, k):
    # clustering
    cluster_loss = 0
    x_idx = []
    for node in x:
        dists = []
        for center_node in center:
            dists.append(clustering_loss(node, center_node))
        min_dist = min(dists)
        cluster_loss += min_dist
        x_idx.append(dists.index(min_dist))
    # print("Clutering loss: ", cluster_loss)
    
    center_loss = 0
    # recompute centers
    for center_idx in range(k):

        # Find xes
        x_list = []
        for i in range(len(x_idx)):
            if x_idx[i] == center_idx:
                x_list.append(x[i])

        # find center
        # new_center = find_center_mht(x_list)
        new_center = find_center_mean(x_list)
        center[center_idx] = new_center

        # centering loss
        center_loss += centering_loss(x_list, new_center)
    # print("Center loss: ", center_loss)

    # print("Clustering loss: {}, Center loss: {}, total loss: {}".format(cluster_loss, center_loss, cluster_loss + center_loss))
    return center, x_idx, cluster_loss, center_loss
        
def restore_weight(x_idx, center):
    new_weight = []
    for idx in x_idx:
        new_weight.append(center[idx])
    return new_weight

def kmean(grad, weight, k, layer, n_iter):
    cur_grad = grad[layer]
    cur_weight = weight['model_state_dict'][layer]
    shapes = cur_weight.shape
    cur_grad = list(cur_grad.reshape(-1).numpy())
    cur_weight = list(cur_weight.reshape(-1).numpy())
    x = list(zip(cur_weight, cur_grad))
    center = init_kmean(cur_weight, k)
    last_loss = 99999999999
    flag = 0
    for i in range(n_iter):
        center, x_idx, cluster_loss, center_loss = kmean_iter(x, center, k)
        total_loss = cluster_loss + center_loss
        loss_step = last_loss - total_loss
        last_loss = total_loss

        # print("Clustering loss: {}, Center loss: {}, total loss: {}".format(cluster_loss, center_loss, cluster_loss + center_loss))
        # print(loss_step)
        if abs(loss_step) < 1e-4:
            flag += 1
        else:
            flag = 0
        
        if flag > 5:
            break
    changed_layer = restore_weight(x_idx, center)
    changed_layer = torch.tensor(changed_layer).reshape(shapes)
    new_weight = weight.copy()
    new_weight['model_state_dict'][layer] = changed_layer
    return new_weight
    
class ResBlock(nn.Module):
    def __init__(self, input_channel_size, output_channel_size, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(input_channel_size, output_channel_size, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(output_channel_size)
        self.relu1 = nn.ReLU()  
        self.conv2 = nn.Conv2d(output_channel_size, output_channel_size, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(output_channel_size)
        if stride == 1:
            self.shortcut = nn.Identity()
        else:
            self.shortcut = nn.Conv2d(input_channel_size, output_channel_size, kernel_size=1, stride=stride)
        self.relu2 = nn.ReLU()
    
    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu1(out)
        out = self.conv2(out)
        out = self.bn2(out)
        shortcut = self.shortcut(x)
        #print(out.shape)
        #print(shortcut.shape)
        out = self.relu2(out + shortcut)
        return out

class Res34(nn.Module):
    def __init__(self, in_features, num_classes,feat_dim = 512):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(in_features, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            ResBlock(64, 64),
            ResBlock(64, 64),
            ResBlock(64, 64),
            ResBlock(64, 128, stride=2),
            ResBlock(128, 128),
            ResBlock(128, 128),
            ResBlock(128, 128),
            ResBlock(128, 256, stride=2),
            ResBlock(256, 256),
            ResBlock(256, 256),
            ResBlock(256, 256),
            ResBlock(256, 256),
            ResBlock(256, 256),
            ResBlock(256, 512, stride=2),
            ResBlock(512, 512),
            ResBlock(512, 512),
            nn.AdaptiveAvgPool2d((1, 1)), # For each channel, collapses (averages) the entire feature map (height & width) to 1x1
            nn.Flatten(), # the above ends up with batch_size x 64 x 1 x 1, flatten to batch_size x 64
        )
        self.linear = nn.Linear(512, feat_dim)
        self.relu = nn.ReLU(inplace=True)
        self.linear_output = nn.Linear(512, num_classes)

    def forward(self, x, return_embedding=False):
        embedding = self.layers(x) 
        embedding_out = self.relu(self.linear(embedding))
        output = self.linear_output(embedding)
        if return_embedding:
            return embedding,output
        else:
            return output  


def evaluate_model(val_loader, model):
  # Set model to evaluating mode
  model.eval()
  num_correct = 0
  loss_value = 0
  with torch.no_grad():
    for batch_num, (inputs, target) in enumerate(val_loader):
      inputs = inputs.to(device)
      target = target.to(device)
      outputs = model(inputs)
      # Computing loss
      loss = criterion(outputs, target)
      num_correct += (torch.argmax(outputs, axis=1) == target).sum().item()
      loss_value += loss.item()

  acc = num_correct / len(eval_dataset)
  #loss_value /= len(eval_dataset)
  return acc, loss_value


val_data_path = "val_data"
test_transforms = torchvision.transforms.Compose([
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406],
                              std=[0.229, 0.224, 0.225]),
])

# Evaluating dataloader
eval_dataset = torchvision.datasets.ImageFolder(root=val_data_path, 
                                                 transform=test_transforms)
eval_dataloader = DataLoader(eval_dataset, batch_size = 256, shuffle=True, num_workers=32, pin_memory = True)

numEpochs = 1
in_features = 3 # RGB channels

learningRate = 1e-9
weightDecay = 5e-5

numEpochs = 1
in_features = 3 # RGB channels

learningRate = 1e-9
weightDecay = 5e-5

num_classes = len(eval_dataset.classes)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = Res34(in_features, num_classes)
# model.load_state_dict(base_model_state_dict)
model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=learningRate, weight_decay=weightDecay, momentum=0.9)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor = 0.5, patience=5)




k_list = [2**1, 2**2, 2**4, 2**8]
layer_list = ['layers.0.weight', 
'layers.3.conv1.weight', 
'layers.3.conv2.weight', 
'layers.4.conv1.weight',  
'layers.4.conv2.weight', 
'layers.5.conv1.weight', 
'layers.5.conv2.weight', 
'layers.6.conv1.weight', 
'layers.6.conv2.weight',
'layers.6.shortcut.weight', 
'layers.7.conv1.weight', 
'layers.7.conv2.weight',
'layers.8.conv1.weight',
'layers.8.conv2.weight',  
'layers.9.conv1.weight', 
'layers.9.conv2.weight']

global fh
fh = open("single_layer.csv", 'w')
fh.write('layer_name,k,acc\n')
grad, weight = load_data()
lock = threading.Lock()

def batch_processing(k_list, layer_list, grad, weight, model, eval_dataloader):
    global fh
    for k in k_list:
        for layer in layer_list:
            start_time = time.time()
            print("KMeaning layer {} with k {}".format(layer, k))
            if k == 2**4 or k == 2**8:
                iters = 75
            else:
                iters = 100
            new_weight = kmean(grad,weight,k, layer, iters)
            end_time = time.time()
            print("KMean time taken {}".format(end_time - start_time))

            model.load_state_dict(new_weight['model_state_dict'])
            print("Evaluating")
            start_time = time.time()
            val_acc, val_loss = evaluate_model(eval_dataloader, model)
            end_time = time.time()
            print("Time taken:"+str(end_time-start_time)+" sec, Validation loss: "+str(val_loss)+", Validation accuracy: "+str(val_acc*100)+"%")

            # if lock.acquire():
            #     try:
            fh.write("{},{},{}\n".format(layer,k,val_acc))
            fh.flush()
                # finally:
                #     lock.release()
    fh.close()
    return True

# def parallel_processing(k_list, layer_list, grad, weight, model, eval_dataloader):
#     threads = []
#     # print "Processing", fns[i * chunksize:(i + 1) * chunksize]
#     # batch_processing(k_list, layer_list, grad, weight, model, eval_dataloader)
#     i = 0
#     for j in range(9):
#         threads.append(
#             threading.Thread(
#                 target = batch_processing,
#                 args = (k_list,
#                         layer_list[4*j:4*j+3],
#                         grad,
#                         weight,
#                         model,
#                         eval_dataloader
#                 ),
#                 name = "thread-" + str(i)
#             )
#         )
#         i += 1
#     # start run threads
#     for thread in threads:
#         thread.start()
#     # wait threads finish
#     for thread in threads:
#         thread.join()
    
#     global fh
#     fh.close()

#     return True

    
batch_processing(k_list, layer_list, grad, weight, model, eval_dataloader)







