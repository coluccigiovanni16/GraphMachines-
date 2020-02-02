def report_stamp(file_name, file_name_rmse, avg_error, rmse_train, rmse_test, num_epochs, true, predicted, optimizer,
                 graph_set_train, graph_set_test, net, criterion, training_time):
    file = open(file_name, 'w+')
    file.write(
        '\n+++++++++++++++++++++++++++++++++++REPORT+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++\n')
    file.write('\nLISTA TEST FILE  \n')
    file.write(str(graph_set_train.keys()))
    file.write(
        '----------------------------------------------------------------------------------------------------------\n')
    file.write('\nLISTA VALIDATION FILE  \n')
    file.write(str(graph_set_test.keys()))
    file.write(
        '----------------------------------------------------------------------------------------------------------\n')
    file.write("\nOur model: \n\n")
    file.write(str(net))
    file.write('\n')
    file.write("Printing children\n------------------------------")
    file.write(str(list(net.children())))
    file.write("\n\nPrinting Modules\n------------------------------")
    file.write(str(list(net.modules())))
    file.write("\nThe state dict keys: \n\n")
    file.write(str(net.state_dict().keys()))
    file.write('Models state_dict:')
    for param_tensor in net.state_dict():
        file.write(param_tensor)
        file.write("\t")
        file.write(str(net.state_dict()[param_tensor].size()))
    file.write(str(net.state_dict()))
    file.write(
        '\n---------------plt.close(fig)    # close the figure window-------------------------------------------------------------------------------------------\n')
    file.write("\n Optimizer's state_dict:")
    file.write('\n')
    file.write("CRITERION = ")
    file.write(str(criterion))
    file.write('\n')
    file.write("OPTIMIZER = ")
    file.write(str(optimizer))
    file.write('\n')
    file.write('Tempo di training ')
    file.write(str(training_time))
    file.write('\n\n\n')
    for var_name in optimizer.state_dict():
        file.write(var_name)
        file.write("\t")
        file.write(str(optimizer.state_dict()[var_name]))
    file.write(
        '\n++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++\n')
    file.write('Average Error Test Dataset= ')
    file.write(str(avg_error))
    file.write('\nRMSE Train Dataset = ')
    file.write(str(rmse_train))
    file.write('\nRMSE Test Dataset = ')
    file.write(str(rmse_test))
    file.write('\nNumber of epochs ')
    file.write(str(num_epochs))
    file.write(
        '\n++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++\n')
    for i in range(len(true)):  # close the figure window
        file.write('pred =')
        file.write(str(predicted[i]))
        file.write('\n')
        file.write('true = ')
        file.write(str(true[i]))
        file.write('\n')
        file.write('differenza = ')
        file.write(str(abs(predicted[i] - true[i])))
        file.write('\n\n\n')
    file.close()
    rmse_stamp(file_name_rmse, avg_error, rmse_train[-1], rmse_test[-1], num_epochs, optimizer, criterion,
               training_time)


def rmse_stamp(file_name_rmse, avg_error, rmse_train, rmse_test, num_epochs, optimizer, criterion, training_time):
    file = open(file_name_rmse, 'w+')
    file.write(
        '\n' + '+' * 500 + '\n')
    file.write('Average Error Test Dataset= ')
    file.write(str(avg_error))
    file.write('\nRMSE Train Dataset = ')
    file.write(str(rmse_train))
    file.write('\nRMSE Test Dataset = ')
    file.write(str(rmse_test))
    file.write('\nNumber of epochs ')
    file.write(str(num_epochs))
    file.write(
        '\n' + '+' * 500 + '\n')
    file.write("CRITERION = ")
    file.write(str(criterion))
    file.write('\n')
    file.write("OPTIMIZER = ")
    file.write(str(optimizer))
    file.write('\n')
    file.write('Tempo di training ')
    file.write(str(training_time))
    file.write(
        '\n' + '+' * 500 + '\n')
    file.close()
