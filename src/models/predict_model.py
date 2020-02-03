import torch
import torch.nn as nn


def test(net, dataSetTrain, DValue, criterion, rmse=False):
    m = nn.Softmax(dim=1)
    avg_error = 0
    with torch.no_grad():
        lossesMean = 0
        result = 0
        dummyDict, deepthdictBatchTensor, deepthdictBatchLabel = dataSetTrain
        leafIndex = len(deepthdictBatchTensor) - 1
        result = net(deepthdictBatchTensor[leafIndex].view(-1, DValue), leafIndex)
        for depth in reversed(range(0, len(deepthdictBatchTensor) - 1)):
            result = net(torch.addmm(deepthdictBatchTensor[depth], dummyDict[depth], result).view(-1, DValue), depth)
        print(result)
        print(m(result))
        print(deepthdictBatchLabel)
        lossesMean = criterion(result, deepthdictBatchLabel)
    print('VALIDATION SET ')
    print('LOSS: ', (lossesMean))

    if (rmse):
        return (lossesMean)
    else:
        for i in range(len(deepthdictBatchLabel)):
            deepthdictBatchLabel[i] = deepthdictBatchLabel[i].item()
        return result.view(-1).tolist(), deepthdictBatchLabel.cpu(), avg_error
