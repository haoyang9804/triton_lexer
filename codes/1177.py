import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton_python_backend_utils as pb_utils


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.dropout1 = nn.Dropout2d(0.25)
        self.dropout2 = nn.Dropout2d(0.5)
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        output = F.log_softmax(x, dim=1)
        return output


class TritonPythonModel:
    def initialize(self, args):
        torch.manual_seed(0)
        self.model = Net()
        self.model.eval()

    def execute(self, requests):
        responses = []
        for request in requests:
            input_tensor = pb_utils.get_input_tensor_by_name(request, "IN")

            input_data_ro = input_tensor.as_numpy()
            input_data = np.array(input_data_ro)
            result = self.model(torch.tensor(input_data))

            out_tensor = pb_utils.Tensor("OUT", result.detach().numpy())
            responses.append(pb_utils.InferenceResponse([out_tensor]))
        return responses
