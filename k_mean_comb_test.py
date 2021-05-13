import os
import sys
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import accuracy_score
import time
from tqdm import tqdm
import math
from sklearn.cluster import KMeans
import mkl
mkl.set_num_threads(70)
np.random.seed(11785)
torch.manual_seed(11785)

cuda = torch.cuda.is_available()
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

def evaluate_model(val_loader, model):
  # Set model to evaluating mode
  model.eval()

  predictions = np.asarray([])
  actuals = np.asarray([])

  eval_loss = 0
  with torch.no_grad():
    for i, (inputs, target) in tqdm(enumerate(val_loader)):
      inputs = inputs.to(device)
      target = target.to(device)

      output = model(inputs)

      # Computing loss
      loss = criterion(output, target)
      eval_loss += loss.item()

      output = output.cpu().detach().numpy()
      actual = target.cpu().numpy()
      # print("target: ")
      # print(actual)
      # print("output: ")
      # print(output)
      output = np.argmax(output, axis=1)

      predictions = np.concatenate((predictions, output), axis=0)
      actuals = np.concatenate((actuals, actual), axis=0)
      
  predictions = predictions.astype(int)
  actuals = actuals.astype(int)
  acc = accuracy_score(actuals, predictions)
  eval_loss /= len(val_loader)
  return acc, eval_loss

def analy(model_state):
    weight_list = []
    shapes = {}
    arrays = {}
    new_model_state = {}
    for key in model_state:
        data = model_state[key]
        arrays[key] = data.numpy()
        shapes[key] = list(data.shape)
        if len(shapes[key]) == 0:
            weight_list.append(model_state[key].item())
            # print("single: ", model_state[key].item())

        elif len(shapes[key]) == 1:
            weight_list += list(arrays[key])
            # print("array: ", list(arrays[key]))
        elif len(shapes[key]) == 2:
            tmp_list = list(arrays[key].reshape(-1))
            weight_list += tmp_list
        else:
            print("You are fucked")

            
        
    # print("Lengths: ", len(weight_list))
    # print(weight_list)
    for key in model_state:
        shape = shapes[key]
        ori_data = model_state[key]
        new_data = np.zeros(shape, dtype=np.int64)
        
        if len(shape) == 0:
            idx = weight_list.index(new_data.item())
            new_data = torch.tensor(idx, dtype=torch.int64)
            new_model_state[key] = new_data
        elif len(shape) == 1:
            for i in range(shape[0]):
                idx = weight_list.index(ori_data[i].item())
                new_data[i] = idx
            new_model_state[key] = torch.from_numpy(new_data, dtype=torch.int64)
        elif len(shape) == 2:
            for i in range(shape[0]):
                for j in range(shape[1]):
                    idx = weight_list.index(ori_data[i, j].item())
                    new_data[i, j] = idx
            new_model_state[key] = torch.from_numpy(new_data, dtype = torch.int64)
        else:
            print("You are fucked")

        print("Finished key: ", key)
    return weight_list, new_model_state

def analy_all_model(model_state, index_list):
    weight_list = []
    shapes = {}
    arrays = {}
    new_model_state = {}
    counter = 0
    for key in model_state:
        print("Start key: ", key)
        data = model_state[key]
        arrays[key] = data.numpy()
        shapes[key] = list(data.shape)

        shape = shapes[key]
        ori_data = model_state[key]
        new_data = np.zeros(shape, dtype=np.int64)

        if len(shapes[key]) == 0:
            tmp_idx = 0
            # idx = len(weight_list) + tmp_idx
            idx = index_list[counter]
            counter += 1
            new_data = torch.tensor(idx).int()
            new_model_state[key] = new_data
            # weight_list.append(data.item())
        elif len(shapes[key]) == 1:
            tmp_list = list(arrays[key])
            # init_length = len(weight_list)
            for i in range(shapes[key][0]):
                new_data[i] = index_list[counter]
                counter += 1
                # weight_list.append(ori_data[i].item())
                # tmp_idx = tmp_list.index(ori_data[i].item())
                # new_data[i] = idx
            # new_data = new_data + init_length
            new_model_state[key] = torch.from_numpy(new_data).int()
            # weight_list += list(arrays[key])
            # print("array: ", list(arrays[key]))

        elif len(shapes[key]) == 2:
            tmp_list = list(arrays[key].reshape(-1))
            # init_length = len(weight_list)
            for i in range(shape[0]):
                for j in range(shape[1]):
                    # idx = i*shape[1] + j
                    # weight_list.append(ori_data[i, j].item())
                    new_data[i, j] = index_list[counter]
                    counter += 1
            # new_data = new_data + init_length
            new_model_state[key] = torch.from_numpy(new_data).int()
            # weight_list += tmp_list
        else:
            print("You are fucked")

        print("Finished: ", key)
    return weight_list, new_model_state

def analy_base_model(model_state):
    weight_list = []
    shapes = {}
    arrays = {}
    new_model_state = {}
    for key in model_state:
        print("Start key: ", key)
        data = model_state[key]
        arrays[key] = data.numpy()
        shapes[key] = list(data.shape)

        shape = shapes[key]
        ori_data = model_state[key]
        new_data = np.zeros(shape, dtype=np.int64)
        if len(shapes[key]) == 0:
            tmp_idx = 0
            idx = len(weight_list) + tmp_idx
            new_data = torch.tensor(idx).int()
            new_model_state[key] = new_data
            weight_list.append(data.item())
        elif len(shapes[key]) == 1:
            tmp_list = list(arrays[key])
            init_length = len(weight_list)
            for i in range(shapes[key][0]):
                new_data[i] = i
                weight_list.append(ori_data[i].item())
                # tmp_idx = tmp_list.index(ori_data[i].item())
                # new_data[i] = idx
            new_data = new_data + init_length
            new_model_state[key] = torch.from_numpy(new_data).int()
            # weight_list += list(arrays[key])
            # print("array: ", list(arrays[key]))

        elif len(shapes[key]) == 2:
            tmp_list = list(arrays[key].reshape(-1))
            init_length = len(weight_list)
            for i in range(shape[0]):
                for j in range(shape[1]):
                    idx = i*shape[1] + j
                    weight_list.append(ori_data[i, j].item())
                    new_data[i, j] = idx
            new_data = new_data + init_length
            new_model_state[key] = torch.from_numpy(new_data).int()
            # weight_list += tmp_list
        else:
            print("You are fucked")

        print("Finished: ", key)
    return weight_list, new_model_state


class MLP(nn.Module):
  # Define Model Element
  def __init__(self, sizeList):
    super().__init__()
    layers = []
    self.sizeList = sizeList
    for i in range(len(sizeList) - 2):
      linearLayer = nn.Linear(sizeList[i], sizeList[i+1])
      if (i % 2 == 0):
        batchNormalizationLayer = nn.BatchNorm1d(sizeList[i+1])
      if (i % 2 == 0):
        dropOut = nn.Dropout(0.5)
      activationLayer = nn.ReLU()
      layers.append(linearLayer)
      layers.append(activationLayer)
      if (i % 2 == 0):
        layers.append(batchNormalizationLayer)
      if (i % 2 == 0):
        layers.append(dropOut)
    outputLayer = nn.Linear(sizeList[-2], sizeList[-1])
    layers.append(outputLayer)
    self.net = nn.Sequential(*layers)
  def forward(self, x):
    return self.net(x)

class SpeechDataSet(Dataset):
  def __init__(self, X_path, Y_path, offset=1, context = 1):
    # Load the data and label from file path
    X = np.load(X_path, allow_pickle=True)
    Y = np.load(Y_path, allow_pickle=True)
    X = np.vstack(X)
    Y = np.hstack(Y)
    self.X = X
    self.Y = Y
    # Assigning length to self
    self.length = len(self.Y)
    # Add offset and context to self
    self.offset = offset
    self.context = context
    # Zero Padding the input data to account for the context
    self.X = np.pad(self.X, ((self.context, self.context), (0, 0)),
                                mode='constant', constant_values=0)
  def __len__(self):
    return self.length
  def __getitem__(self, index):
    # Compute the start index and the end index
    start_index = index + self.offset - self.context
    end_index = index + self.offset + self.context + 1
    # Extract data and label
    data = torch.flatten(torch.tensor(self.X[start_index:end_index, :]).float())
    label = torch.tensor(self.Y[index]).long()
    return data, label


# save_path = "state_dicts"
# model_name = "base_model"

# base = torch.load("benchmark", map_location = torch.device('cpu'))
# base_model_state_dict = base['model_state_dict']

# weight_list, new_model_state = analy_base_model(base_model_state_dict)


# torch.save({
#     'model_state_dict': new_model_state,
#     'weight_list': weight_list
# }, os.path.join(save_path,model_name))



# Model 1(original), Model 2(quantized)


def single_layer_kmean(root_codebook_size, weight_dict):
    length_list = []
    layer_weight_dict = {}
    layer_shape_dict = {}

    new_weight_dict = {}
    for key in weight_dict:
        if "weight" in key:
            length_list.append(len(weight_dict[key]))
        if "bias" in key:
            length_list[-1] += len(weight_dict[key])

    total = sum(length_list)
    ratio_list = [i / total for i in length_list]
    # print(ratio_list)
    codebook_size_list = [math.floor(root_codebook_size*i) for i in ratio_list]
    for key in weight_dict:
        if "weight" in key:
            layer_weight_dict[key] = weight_dict[key]
            layer_shape_dict[key] = [len(weight_dict[key])]
        else:
            tmp_key = key.replace("bias", "weight")
            layer_weight_dict[tmp_key] += weight_dict[key]
            layer_shape_dict[tmp_key].append(len(weight_dict[key]))

    for i, length in enumerate(codebook_size_list):
        if length < 2:
            print("Not able to perform kmean in this case")
            return None
        
        if  length > length_list[i]:  
            print("Codebook size {}, Weight size {}".format(length, length_list[i]))
            print("Codebook size too large for single kmean")
            return None
    
    for i, key in enumerate(layer_weight_dict):
        weight = layer_weight_dict[key]
        codebook_size = codebook_size_list[i]

        kmean = KMeans(n_clusters = codebook_size, init = "k-means++", n_init = 3, max_iter = 300, precompute_distances = True, verbose = 1, n_jobs = -1,algorithm = "full")
        kmean.fit(np.array(weight).reshape(-1, 1))

        labels = kmean.labels_
        centers = kmean.cluster_centers_
        centers = [i[0] for i in centers]
        
        new_weight = [centers[i] for i in labels]
        new_weight_dict[key] = new_weight.copy()

    # print(new_weight_dict)
    #  Restoring.
    tmp_dict = {}

    for key in new_weight_dict:
        data = new_weight_dict[key].copy()
        length1, length2 = layer_shape_dict[key][0], layer_shape_dict[key][1]
        data1 = data[:length1]
        data2 = data[length1:]

        bias_key = key.replace("weight", "bias")
        data1_shape = list(ori_model_weight["model_state_dict"][key].shape)
        data2_shape = list(ori_model_weight["model_state_dict"][bias_key].shape)

        data1 = np.array(data1).reshape(data1_shape)
        data2 = np.array(data2).reshape(data2_shape)

        data1 = torch.from_numpy(data1)
        data2 = torch.from_numpy(data2)

        tmp_dict[key] = data1
        tmp_dict[bias_key] = data2

    return tmp_dict
        

    
def analy_comb(root_codebook_size, layer_list):
    modified_model =  MLP([inputSize, 2048, 1950, 1750, 1600, 1400, 1300, 1200, 1000, 800, outputSize])
    modified_model.eval()
    modified_model.to(device)

    new_state_dict_single = modified_model.state_dict().copy()
    new_state_dict_all = modified_model.state_dict().copy()

    weight_name_list = []
    weight_dict = {}
    all_weight_dict = {}

    for layer in layer_list:
        weight_name_list += weight_bias_dict[layer]
    
    print("weight/bias to be quantized: ", weight_name_list)

    # Get weight lists.
    for key in weight_name_list:
        tmp_list = []
        data = ori_model_weight["model_state_dict"][key]

        shape = list(data.shape)
        array = data.numpy()

        if len(shape) == 1:
            tmp_list = list(array)
            weight_dict[key] = tmp_list.copy()
        elif len(shape) == 2:
            tmp_list = list(array.reshape(-1))
            weight_dict[key] = tmp_list.copy()
        else:
            print("You are fucked")
            exit(-1)
    
    single_weight_dict = single_layer_kmean(root_codebook_size, weight_dict)
    if single_weight_dict is not None:
        for key in single_weight_dict:
            new_state_dict_single[key] = single_weight_dict[key]
    else:
        print("Single layer kmean is not done.")

    return new_state_dict_single



root_codebook_size = 2**8

model_weights = {}
weight_bias_dict = {}
weight_bias_dict["l0"] = ["net.0.weight", "net.0.bias"]
weight_bias_dict["l1"] = ["net.4.weight", "net.4.bias"]
weight_bias_dict["l2"] = ["net.6.weight", "net.6.bias"]
weight_bias_dict["l3"] = ["net.10.weight", "net.10.bias"]
weight_bias_dict["l4"] = ["net.12.weight", "net.12.bias"]
weight_bias_dict["l5"] = ["net.16.weight", "net.16.bias"]
weight_bias_dict["l6"] = ["net.18.weight", "net.18.bias"]
weight_bias_dict["l7"] = ["net.22.weight", "net.22.bias"]
weight_bias_dict["l8"] = ["net.24.weight", "net.24.bias"]
weight_bias_dict["l9"] = ["net.28.weight", "net.28.bias"]


# Vals
context = 30
offset = context
inputSize = (2 * context + 1) * 40
outputSize = 71
ori_model = MLP([inputSize, 2048, 1950, 1750, 1600, 1400, 1300, 1200, 1000, 800, outputSize])
val_data = SpeechDataSet('dev.npy', 'dev_labels.npy', offset, context)
val_args = dict(shuffle=False, batch_size=2048, num_workers=8, drop_last=False) if cuda\
                    else dict(shuffle=False, batch_size=2048, drop_last=False, num_workers = 16)
val_loader = DataLoader(val_data, **val_args)


#Define Criterion/Loss Function and optimizer
criterion = nn.CrossEntropyLoss()
ori_model_weight =  torch.load("benchmark", map_location=device)
ori_model.load_state_dict(ori_model_weight["model_state_dict"])
ori_model.to(device)
ori_model.eval()

# acc, loss = evaluate_model(val_loader, ori_model)
# print("acc: {} loss: {}".format(acc, loss))

state = torch.load("test_8")['model_state_dict_single']
modified_model =  MLP([inputSize, 2048, 1950, 1750, 1600, 1400, 1300, 1200, 1000, 800, outputSize])
modified_model.eval()
model = ori_model.load_state_dict(state)
modified_model.to(device)
acc, loss = evaluate_model(val_loader, modified_model)
print("acc: {} loss: {}".format(acc, loss))

# state = analy_comb(root_codebook_size, ["l1"])

# torch.save({"model_state_dict":state}, "test")




