import math

import torch


# method used to predict_regression the value using the testset in input, can return the list of the predicted values,
# the true values and the average error or simply the rmse value (if invoked in the train_regression method)
def predict_regression(net, data_set_train, d_value, criterion, rmse=False):
    avg_error = 0
    with torch.no_grad():
        dummy_dict, deepthdict_batch_tensor, deepthdict_batch_label = data_set_train
        leaf_index = len(deepthdict_batch_tensor) - 1
        result = net(deepthdict_batch_tensor[leaf_index].view(-1, d_value), leaf_index)
        for depth in reversed(range(0, len(deepthdict_batch_tensor) - 1)):
            result = net(torch.addmm(deepthdict_batch_tensor[depth], dummy_dict[depth], result).view(-1, d_value),
                         depth)
        losses_mean = criterion(result, deepthdict_batch_label)
    for t in (result.sub(deepthdict_batch_label)).tolist():
        avg_error += abs(t[0])
    avg_error = avg_error / len(deepthdict_batch_label)
    print('VALIDATION SET ')
    print('RMSE: ', math.sqrt(losses_mean))
    print('Average Error: ', avg_error)
    print("-" * 100)
    if rmse:
        return math.sqrt(losses_mean)
    else:
        for i in range(len(deepthdict_batch_label)):
            deepthdict_batch_label[i] = deepthdict_batch_label[i].item()
        return result.view(-1).tolist(), deepthdict_batch_label.cpu(), avg_error


def predict_classification(net, data_set_train, d_value, criterion, loss=False):
    avg_error = 0
    with torch.no_grad():
        dummy_dict, deepthdict_batch_tensor, deepthdict_batch_label = data_set_train
        leaf_index = len(deepthdict_batch_tensor) - 1
        result = net(deepthdict_batch_tensor[leaf_index].view(-1, d_value), leaf_index)
        for depth in reversed(range(0, len(deepthdict_batch_tensor) - 1)):
            result = net(torch.addmm(deepthdict_batch_tensor[depth], dummy_dict[depth], result).view(-1, d_value),
                         depth)
        losses_mean = criterion(result, deepthdict_batch_label)
    print('VALIDATION SET ')
    print('LOSS: ', losses_mean)

    if (loss):
        return (losses_mean)
    else:
        for i in range(len(deepthdict_batch_label)):
            deepthdict_batch_label[i] = deepthdict_batch_label[i].item()
        return result.view(-1).tolist(), deepthdict_batch_label.cpu(), avg_error
