import math

import torch
import torch.nn as nn


def test(net, data_set_train, d_value, criterion, rmse=False):
    avg_error = 0
    with torch.no_grad():
        # lossesMean = 0
        # criterion = nn.MSELoss()
        # result = 0
        dummy_dict, deepthdict_batch_tensor, deepthdict_batch_label = data_set_train
        leaf_index = len(deepthdict_batch_tensor) - 1
        result = net(deepthdict_batch_tensor[leaf_index].view(-1, d_value), leaf_index)
        for depth in reversed(range(0, len(deepthdict_batch_tensor) - 1)):
            result = net(torch.addmm(deepthdict_batch_tensor[depth], dummy_dict[depth], result).view(-1, d_value),
                         depth)
        lossesMean = criterion(result, deepthdict_batch_label)
    for t in (result.sub(deepthdict_batch_label)).tolist():
        avg_error += abs(t[0])
    avg_error = avg_error / len(deepthdict_batch_label)
    print('VALIDATION SET ')
    print('RMSE: ', math.sqrt(lossesMean))
    print('Average Error: ', avg_error)
    if (rmse):
        return math.sqrt(lossesMean)
    else:
        for i in range(len(deepthdict_batch_label)):
            deepthdict_batch_label[i] = deepthdict_batch_label[i].item()
        return result.view(-1).tolist(), deepthdict_batch_label.cpu(), avg_error
