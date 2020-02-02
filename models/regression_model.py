import torch.nn as nn


class RFNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(RFNN, self).__init__()  # Inherited from the parent class nn.Module
        self.fc1 = nn.Linear(input_size, hidden_size,
                             bias=False)  # 1st Full-Connected Layer:  (input data) ->  (hidden node)
        # nn.init.kaiming_normal_(self.fc1.weight)
        self.fc2 = nn.Linear(hidden_size, output_size,
                             bias=False)  # 2nd Full-Connected Layer:  (hidden node) ->  (output class)
        # nn.init.kaiming_normal_(self.fc2.weight)
        #         self.LRelu = nn.LeakyReLU(0.25)
        self.LRelu = nn.SELU()
        self.fc1Root = nn.Linear(input_size, hidden_size,
                                 bias=False)  # 1st Full-Connected Layer:  (input data) ->  (hidden node)
        # nn.init.kaiming_normal_(self.fc1Root.weight)
        self.fc2Root = nn.Linear(hidden_size, output_size,
                                 bias=False)  # 2nd Full-Connected Layer:  (hidden node) ->  (output class)
        # nn.init.kaiming_normal_(self.fc2Root.weight)
        #         self.LReluRoot = nn.LeakyReLU(0.25)
        self.LReluRoot = nn.SELU()

    def forward(self, batch_tensor, deep):
        if deep == 0:
            out = self.fc1Root(batch_tensor)
            out = self.LReluRoot(out)
            out = self.fc2Root(out)
            return out
        else:
            out = self.fc1(batch_tensor)
            out = self.LRelu(out)
            out = self.fc2(out)
            return out
