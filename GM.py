import argparse
import os

from src.models.predict_model import test
from src.visualization.visualize import report_stamp, plot_rmse

# imposto come path root quella del progetto
os.chdir(os.path.dirname(os.path.abspath(__file__)))
import torch.nn as nn
import torch

from src.Net.FNN_GM_Net import RegressionGm
from src.data.loadDataset import load_true_value, dag_dict, dictOfFNameList, getDvalue, create_graph_tensor, \
    dataset_loader
from src.models.train_model import train

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
parser.add_argument('-rn', "--reportName", default='', help="base name for the report's folder ")
args = parser.parse_args()
num_epochs = int(args.num_epochs)
hidden_layer = int(args.hidden_layer_size)
save = bool(args.report)
learning_rate = float(args.learning_rate)
rootDir = args.rootDir
trainFile = args.trainFile
testFile = args.testFile
reportName = args.reportName
bias = int(args.bias)
device = torch.device(args.device)
print(args)

datasetFilenameTrain, labelTrain = load_true_value(rootDir, trainFile)
graphTrain, sorOrderedListTrain, depthNodesTrain, centerNodeTrain = dag_dict(
    dictOfFNameList(rootDir, datasetFilenameTrain))
DTrain, maxMTrain = getDvalue(graphTrain)

datasetFilenameTest, labelTest = load_true_value(rootDir, testFile)
graphTest, sorOrderedListTest, depthNodesTest, centerNodeTest = dag_dict(
    dictOfFNameList(rootDir, datasetFilenameTest))
DTest, maxMTest = getDvalue(graphTest)

d_value = max(DTrain, DTest)
maxMValue = max(maxMTrain, maxMTest)

graphTensorTrain = create_graph_tensor(graphTrain, bias, maxMValue, d_value)
dataSetTrain = dataset_loader(depthNodesTrain, centerNodeTrain, sorOrderedListTrain, graphTensorTrain, labelTrain,
                              d_value, device)

graphTensorTest = create_graph_tensor(graphTest, bias, maxMValue, d_value)
dataSetTest = dataset_loader(depthNodesTest, centerNodeTest, sorOrderedListTest, graphTensorTest, labelTest, d_value,
                             device)

input_size = d_value  # The features size (our case is 12)
hidden_size = hidden_layer  # The number of nodes at the hidden layer
output_size = 1  # The number of output classes. In this case 1

net = RegressionGm(input_size, hidden_size, output_size).to(device)

criterion = nn.MSELoss()
optimizer = torch.optim.Adam(net.parameters())

RMSETrain, RMSETest, training_time = train(net, dataSetTrain, dataSetTest, optimizer, num_epochs, d_value, criterion)
predicted, true, avg_error = test(net, dataSetTest, d_value, criterion)

reportFolder = './reports/REPORT-' + reportName

plot_rmse(true, predicted, RMSETrain, RMSETest, testFile, reportFolder)  # funziona

report_stamp(reportFolder, testFile, avg_error, RMSETrain, RMSETest, num_epochs, true, predicted, optimizer,
             graphTrain,
             graphTest, net, criterion, training_time)
