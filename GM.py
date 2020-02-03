import argparse
import os

abspath = os.path.abspath(__file__)
dname = os.path.dirname(abspath)
print(dname)
os.chdir(dname)
import torch.nn as nn
import torch

from src.Net.FNN_GM_Net import RegressionGm
from src.data.loadDataset import load_true_value, dag_dict, dictOfFNameList, getDvalue, create_graph_tensor, \
    dataset_loader
from src.models.train_model import train

#
# i=0
# num_epochs = 100
# bias = 1
# rootDir = '/home/elbarto91/provapipEnv/graphmachines/data/processed/Alkane/'
# trainFile = "trainset_" + str(i) + ".ds"
# testFile = "testset_" + str(i) + ".ds"
# device = torch.device("cpu")

parser = argparse.ArgumentParser()
parser.add_argument('-d', "--device", default='cpu', help="device to use(GPU or CPU(defualt))")
parser.add_argument('-e', "--num_epochs", default=10000, help="number of epochs")
parser.add_argument('-hln', "--hidden_layer_size", default=4, help="number of nodes for the hidden layer", )
parser.add_argument('-lr', "--learning_rate", default=0.001, help="learning rate for the optimizer")
parser.add_argument('-r', "--report", default=False, help="save result in a report file")
parser.add_argument('-root', "--rootDir", default='', help="directory of file")
parser.add_argument('-trf', "--trainFile", default='', help="train")
parser.add_argument('-tef', "--testFile", default='', help="test")
parser.add_argument('-b', "--bias", default=1, help="bias value")
args = parser.parse_args()
num_epochs = int(args.num_epochs)
hidden_layer = int(args.hidden_layer_size)
save = bool(args.report)
learning_rate = float(args.learning_rate)
rootDir = args.rootDir
trainFile = args.trainFile
testFile = args.testFile
bias = int(args.bias)
device = torch.device(args.device)
print(args)

i = 0
num_epochs = 100
bias = 1
rootDir = '/home/elbarto91/provapipEnv/graphmachines/data/processed/Alkane/'
trainFile = "trainset_" + str(i) + ".ds"
testFile = "testset_" + str(i) + ".ds"
device = torch.device("cpu")

datasetFilenameTrain, labelTrain = load_true_value(rootDir, trainFile)
graphTrain, sorOrderedListTrain, depthNodesTrain, centerNodeTrain = dag_dict(
    dictOfFNameList(rootDir, datasetFilenameTrain))
DTrain, maxMTrain = getDvalue(graphTrain)

datasetFilenameTest, labelTest = load_true_value(rootDir, testFile)
graphTest, sorOrderedListTest, depthNodesTest, centerNodeTest = dag_dict(
    dictOfFNameList(rootDir, datasetFilenameTest))
DTest, maxMTest = getDvalue(graphTest)

DValue = max(DTrain, DTest)
maxMValue = max(maxMTrain, maxMTest)

graphTensorTrain = create_graph_tensor(graphTrain, bias, maxMValue, DValue)
dataSetTrain = dataset_loader(depthNodesTrain, centerNodeTrain, sorOrderedListTrain, graphTensorTrain, labelTrain,
                              DValue, device)

graphTensorTest = create_graph_tensor(graphTest, bias, maxMValue, DValue)
dataSetTest = dataset_loader(depthNodesTest, centerNodeTest, sorOrderedListTest, graphTensorTest, labelTest, DValue,
                             device)

input_size = DValue  # The image size = 28 x 28 = 784
hidden_size = 4  # The number of nodes at the hidden layer
output_size = 1  # The number of output classes. In this case, from 0 to 9

net = RegressionGm(input_size, hidden_size, output_size).to(device)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(net.parameters())
# optimizer = torch.optim.RMSprop(net.parameters())

RMSETrain, RMSETest, training_time = train(net, dataSetTrain, dataSetTest, optimizer, num_epochs, DValue, criterion)
# predicted, true, avg_error = test(net, dataSetTest, d_value,criterion)
# plotRmse(true, predicted, RMSETrain, RMSETest)
# fileName = './REPORT/' + testFile + 'ReportGenerale.txt'
# fileNameRMSE = './REPORT/' + testFile + 'ReportRMSE.txt'
# report_stamp(fileName, fileNameRMSE, avg_error, RMSETrain, RMSETest, num_epochs, true, predicted, optimizer,
#              graphTrain,
#              graphTest, net, criterion, training_time)
