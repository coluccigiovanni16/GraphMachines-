import time, math, torch, re, argparse,re
import torch.nn as nn
import matplotlib.pyplot as plt
import networkx as nx
from src.scott import *


def train(net, dataSetTrain, dataSetTest, optimizer, num_epochs, DValue, criterion):
    # # Training loop
    m = nn.Softmax(dim=1)
    RMSETrain = []
    RMSETest = []
    start = time.time()
    losses = 0
    y_true = []
    y_pred = []
    lossesA = []
    dummyDict, deepthdictBatchTensor, deepthdictBatchLabel = dataSetTrain
    for epoch in range(num_epochs):
        leafIndex = len(deepthdictBatchTensor) - 1
        res = net(deepthdictBatchTensor[leafIndex].view(-1, DValue), leafIndex)
        for depth in reversed(range(0, len(deepthdictBatchTensor) - 1)):
            res = net(torch.sparse.addmm(deepthdictBatchTensor[depth], dummyDict[depth], res).view(-1, DValue), depth)
        losses = criterion(res, deepthdictBatchLabel)
        optimizer.zero_grad()
        losses.backward()
        optimizer.step()
        plt.plot(losses.item())
        lossesA.append(losses)
        if (epoch + 1) % (int(num_epochs / 1)) == 0:  # print every (num_epochs/10) epochs --> total 10 print
            print('TRAIN SET \nEpoch [%d/%d],  \nLOSS: %.5f \n '
                  % (epoch + 1, num_epochs, losses))
            RMSETest.append(test(net, dataSetTest, DValue, criterion, rmse=True))
            RMSETrain.append((losses))
            pred = np.round(res.detach().numpy())
            target = deepthdictBatchLabel.float()
            y_true.extend(deepthdictBatchLabel.tolist())
            y_pred.extend(pred.reshape(-1).tolist())

    plt.plot(lossesA)
    plt.show()
    end = time.time()
    training_time = end - start
    print('Tempo di training ', training_time)
    print('FINE TRAINING')
    print('\n+++++++++++++++++++++++++++++++++++++++\n\n')
    print('LOSSTest', RMSETest)
    print('LOSSTrain', RMSETrain)
    return RMSETrain, RMSETest, training_time