import math
import time

import torch

from .predict_model import test


def train(net, dataSetTrain, dataSetTest, optimizer, num_epochs, DValue, criterion):
    # # Training loop
    RMSETrain = []
    RMSETest = []
    start = time.time()
    losses = 0
    dummyDict, deepthdictBatchTensor, deepthdictBatchLabel = dataSetTrain
    for epoch in range(num_epochs):
        leafIndex = len(deepthdictBatchTensor) - 1
        res = net(deepthdictBatchTensor[leafIndex].view(-1, DValue), leafIndex)
        for depth in reversed(range(0, len(deepthdictBatchTensor) - 1)):
            res = net(torch.sparse.addmm(deepthdictBatchTensor[depth], dummyDict[depth], res).view(-1, DValue), depth)
        losses = criterion(res, deepthdictBatchLabel)
        optimizer.zero_grad()
        losses = losses + (net.fc1.weight.norm() + net.fc2.weight.norm()) + 0.20 * (
                net.fc1Root.weight.norm() + net.fc2Root.weight.norm())
        losses.backward()
        optimizer.step()
        if (epoch + 1) % (int(num_epochs / 10)) == 0:  # print every (num_epochs/10) epochs --> total 10 print
            print('TRAIN SET \nEpoch [%d/%d],  \nRMSE: %.5f \n '
                  % (epoch + 1, num_epochs, math.sqrt(losses)))
            RMSETest.append(test(net, dataSetTest, DValue, criterion, rmse=True))
            RMSETrain.append(math.sqrt(losses))

    #             print('-------------------------------------------------------------------------------------\n\n')

    end = time.time()
    training_time = end - start
    print('Tempo di training ', training_time)
    print('FINE TRAINING')
    print('\n+++++++++++++++++++++++++++++++++++++++\n\n')
    print('RMSETest', RMSETest)
    print('rmse_train', RMSETrain)
    return RMSETrain, RMSETest, training_time
