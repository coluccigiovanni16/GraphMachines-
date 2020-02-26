import argparse
import os

import torch
import torch.nn as nn

from src.Net.FNN_GM_Net import RegressionGm, save_model, laod_model
from src.data.load_dataset import load_true_value_regression, dag_creator, dict_of_file_name_list, get_d_value, \
    create_graph_tensor, dataset_loader
from src.models.predict_model import predict_regression
from src.models.train_model import train_regression
from src.visualization.visualize import report_stamp, plot_rmse

# set the path of the project as root path
os.chdir(os.path.dirname(os.path.abspath(__file__)))

parser = argparse.ArgumentParser()
parser.add_argument('-d', "--device", default='cpu', help="device to use(GPU or CPU(defualt))")
parser.add_argument('-e', "--num_epochs", default=10000, help="number of epochs,default=10000")
parser.add_argument('-hln', "--hidden_layer_size", default=4,
                    help="number of nodes for the hidden layer, default = 4", )
parser.add_argument('-lr', "--learning_rate", default=0.001, help="learning rate for the optimizer, default = 0.001")
parser.add_argument('-r', "--report", default=False, help="save result in a report file")
parser.add_argument('-rdd', "--rootDirDataset", default='', help="directory of dataset files")
parser.add_argument('-trf', "--trainFile", default=False, help="dataset containing the name on the trainset files")
parser.add_argument('-tef', "--testFile", default=False, help="dataset containing the name on the testset files")
parser.add_argument('-s', "--save", default=False, help="True if you want to save the model, default = False")
parser.add_argument('-l', "--load", default=False, help="True if you want to load the model, default = False")
parser.add_argument('-b', "--bias", default=1, help="bias value, default = 1")
parser.add_argument('-rn', "--reportName", default=False, help="base name for the report's folder ")
parser.add_argument('-mp', "--modelPath", default=False, help="model's path")
args = parser.parse_args()
num_epochs = int(args.num_epochs)
hidden_layer = int(args.hidden_layer_size)
save = bool(args.save)
load = bool(args.load)
report = bool(args.report)
learning_rate = float(args.learning_rate)
rootDir = args.rootDirDataset
modelPath = args.modelPath
trainFile = args.trainFile
testFile = args.testFile
reportName = args.reportName
bias = int(args.bias)
device = torch.device(args.device)
print(args)

# TRAIN PART
if trainFile and testFile:
    # load the true value of the trainset
    datasetFilenameTrain, labelTrain = load_true_value_regression(rootDir, trainFile)
    # datastructure used behind to construct the matrix (one for every depth)
    graphTrain, sorOrderedListTrain, depthNodesTrain, centerNodeTrain = dag_creator(
        dict_of_file_name_list(rootDir, datasetFilenameTrain))

    # d and m value used to constarct the fature array of every node(total size of array and size for the son's value)
    DTrain, maxMTrain = get_d_value(graphTrain)

    datasetFilenameTest, labelTest = load_true_value_regression(rootDir, testFile)
    graphTest, sorOrderedListTest, depthNodesTest, centerNodeTest = dag_creator(
        dict_of_file_name_list(rootDir, datasetFilenameTest))
    DTest, maxMTest = get_d_value(graphTest)

    d_value = max(DTrain, DTest)
    maxMValue = max(maxMTrain, maxMTest)

    graphTensorTrain = create_graph_tensor(graphTrain, bias, maxMValue, d_value)
    dataSetTrain = dataset_loader(depthNodesTrain, centerNodeTrain, sorOrderedListTrain, graphTensorTrain, labelTrain,
                                  d_value, device)

    graphTensorTest = create_graph_tensor(graphTest, bias, maxMValue, d_value)
    dataSetTest = dataset_loader(depthNodesTest, centerNodeTest, sorOrderedListTest, graphTensorTest, labelTest,
                                 d_value, device)

    # layer size of the neural network(FNN)
    input_size = d_value  # The features size (our case is 12)
    hidden_size = hidden_layer  # The number of nodes at the hidden layer
    output_size = 1  # The number of output classes. In this case 1

    net = RegressionGm(input_size, hidden_size, output_size).to(device)

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate)

    # laod a model from disk
    if load:
        net = laod_model(net, modelPath)
        print("Model correctly laoded")

    # train_regression the NN
    RMSETrain, RMSETest, training_time = train_regression(net, dataSetTrain, dataSetTest, optimizer, num_epochs,
                                                          d_value, criterion)
    # predict_regression using NN and testset
    predicted, true, avg_error = predict_regression(net, dataSetTest, d_value, criterion)

    reportFolder = './reports/REPORT-' + reportName

    # PLOT + PRINT
    plot_rmse(true, predicted, RMSETrain, RMSETest, testFile, reportFolder)  # funziona
    if report:
        report_stamp(reportFolder, testFile, avg_error, RMSETrain, RMSETest, num_epochs, true, predicted, optimizer,
                     graphTrain,
                     graphTest, net, criterion, training_time)

    # save the model trained
    if save:
        save_model(net, reportName,
                   "/model_" + testFile + "-Dvalue" + str(d_value) + "-maxMValue" + str(maxMValue) + "-Saved.pth")
        print("Model correctly saved")

# PREDICT ONLY PART
elif testFile and load and modelPath and not trainFile:
    # load the d and M value using the name of the model
    d_value = int(modelPath.split("-Dvalue")[1].split("-maxMValue")[0])
    maxMValue = int(modelPath.split("-maxMValue")[1].split("-Saved.pth")[0])

    input_size = d_value  # The features size (our case is 12)
    hidden_size = hidden_layer  # The number of nodes at the hidden layer
    output_size = 1  # The number of output classes. In this case 1
    model = RegressionGm(input_size, hidden_size, output_size).to(device)
    criterion = nn.MSELoss()
    net = laod_model(model, modelPath)
    print("Model correctly loaded")

    datasetFilenameTest, labelTest = load_true_value_regression(rootDir, testFile)
    graphTest, sorOrderedListTest, depthNodesTest, centerNodeTest = dag_creator(
        dict_of_file_name_list(rootDir, datasetFilenameTest))
    graphTensorTest = create_graph_tensor(graphTest, bias, maxMValue, d_value)
    dataSetTest = dataset_loader(depthNodesTest, centerNodeTest, sorOrderedListTest, graphTensorTest, labelTest,
                                 d_value,
                                 device)
    predicted, true, avg_error = predict_regression(net, dataSetTest, d_value, criterion)
    reportFolder = './reports/REPORT-' + reportName
    plot_rmse(true, predicted, [], [], testFile, reportFolder)
