import torch
import torch.nn as nn

class AdvancedNeuralModel(nn.Module):
    def __init__(self, input_features, hidden_units, output_classes):
        super(AdvancedNeuralModel, self).__init__()
        self.input_layer = nn.Linear(input_features, hidden_units)
        self.hidden_layer = nn.Linear(hidden_units, hidden_units)
        self.output_layer = nn.Linear(hidden_units, output_classes)
        self.activation_function = nn.ReLU()

    def forward(self, input_tensor):
        x = self.input_layer(input_tensor)
        x = self.activation_function(x)
        x = self.hidden_layer(x)
        x = self.activation_function(x)
        x = self.output_layer(x)
        return x
