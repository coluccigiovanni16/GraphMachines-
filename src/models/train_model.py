import math
import time

import matplotlib.pyplot as plt
import numpy as np
import torch

from .predict_model import predict_regression


# train_regression algorithm,take in input the trainset and the testset(to check how the model evolves
def train_regression(net, data_set_train, data_set_test, optimizer, num_epochs, d_value, criterion):
    # # Training loop
    rmse_train = []
    rmse_test = []
    start = time.time()
    dummy_dict, deepthdict_batch_tensor, deepthdict_batch_label = data_set_train
    for epoch in range(num_epochs):
        leaf_index = len(deepthdict_batch_tensor) - 1
        res = net(deepthdict_batch_tensor[leaf_index].view(-1, d_value), leaf_index)
        for depth in reversed(range(0, len(deepthdict_batch_tensor) - 1)):
            res = net(torch.sparse.addmm(deepthdict_batch_tensor[depth], dummy_dict[depth], res).view(-1, d_value),
                      depth)
        losses = criterion(res, deepthdict_batch_label)
        optimizer.zero_grad()
        losses = losses + (net.fc1.weight.norm() + net.fc2.weight.norm()) + 0.20 * (
                net.fc1Root.weight.norm() + net.fc2Root.weight.norm())
        losses.backward()
        optimizer.step()
        if (epoch + 1) % (int(num_epochs / 10)) == 0:  # print every (num_epochs/10) epochs --> total 10 print
            print('TRAIN SET \nEpoch [%d/%d],  \nRMSE: %.5f \n '
                  % (epoch + 1, num_epochs, math.sqrt(losses)))
            # call the predict_regression algo to check the evolution of the network
            rmse_test.append(predict_regression(net, data_set_test, d_value, criterion, rmse=True))
            rmse_train.append(math.sqrt(losses))

    #             print('-------------------------------------------------------------------------------------\n\n')

    end = time.time()
    training_time = end - start
    print('Tempo di training ', training_time +" seconds")
    print('FINE TRAINING')
    print('\n+++++++++++++++++++++++++++++++++++++++\n\n')
    print('rmse_test', rmse_test)
    print('rmse_train', rmse_train)
    return rmse_train, rmse_test, training_time


def train_classification(net, data_set_train, data_set_test, optimizer, num_epochs, d_value, criterion):
    # # Training loop
    lossTrain = []
    lossTest = []
    start = time.time()
    y_true = []
    y_pred = []
    losses_a = []
    dummy_dict, deepthdict_batch_tensor, deepthdict_batch_label = data_set_train
    for epoch in range(num_epochs):
        leafIndex = len(deepthdict_batch_tensor) - 1
        res = net(deepthdict_batch_tensor[leafIndex].view(-1, d_value), leafIndex)
        for depth in reversed(range(0, len(deepthdict_batch_tensor) - 1)):
            res = net(torch.sparse.addmm(deepthdict_batch_tensor[depth], dummy_dict[depth], res).view(-1, d_value),
                      depth)
        losses = criterion(res, deepthdict_batch_label)
        optimizer.zero_grad()
        losses.backward()
        optimizer.step()
        plt.plot(losses.item())
        losses_a.append(losses)
        if (epoch + 1) % (int(num_epochs / 1)) == 0:  # print every (num_epochs/10) epochs --> total 10 print
            print('TRAIN SET \nEpoch [%d/%d],  \nLOSS: %.5f \n '
                  % (epoch + 1, num_epochs, losses))
            lossTest.append(predict_regression(net, data_set_test, d_value, criterion, rmse=True))
            lossTrain.append(losses)
            pred = np.round(res.detach().numpy())
            y_true.extend(deepthdict_batch_label.tolist())
            y_pred.extend(pred.reshape(-1).tolist())

    plt.plot(losses_a)
    plt.show()
    end = time.time()
    training_time = end - start
    print('Tempo di training ', training_time)
    print('FINE TRAINING')
    print('\n+++++++++++++++++++++++++++++++++++++++\n\n')
    print('LOSSTest', lossTest)
    print('LOSSTrain', lossTrain)
    return lossTrain, lossTest, training_time
