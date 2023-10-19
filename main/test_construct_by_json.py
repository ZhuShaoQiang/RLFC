# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
from torchsummary import summary

class CustomNet(nn.Module):
    def __init__(self, in_shape, structure):
        super(CustomNet, self).__init__()

        layers = []
        input_channels = in_shape[0]

        for layer in structure:
            if isinstance(layer, list):
                layer_name, layer_params = layer[0], layer[1]
                if layer_name == "conv2d":
                    conv2d_layer = nn.Conv2d(
                        in_channels=input_channels,
                        out_channels=layer_params["out_channels"],
                        kernel_size=layer_params["kernel_size"],
                        stride=layer_params["stride"]
                    )
                    layers.append(conv2d_layer)
                    layers.append(nn.ReLU())
                    input_channels = layer_params["out_channels"]
                elif layer_name == "activation":
                    layers.append(nn.ReLU())
                elif layer_name == "flatten":
                    layers.append(nn.Flatten())
                else:
                    raise ValueError("Invalid layer name.")
            elif isinstance(layer, int):
                ## 这个地方之前必有flatten，
                self.nn = nn.Sequential(*layers)
                __tmp = self._get_conv_output_shape(in_shape)
                dense_layer = nn.Linear(__tmp[-1], layer)
                layers.append(dense_layer)
                input_channels = layer

        self.nn = nn.Sequential(*layers)

    def _get_conv_output_shape(self, input_shape: tuple):
        """
        计算卷积层的输出形状
        input_shape: 如(3, 96, 96)
        """
        zeros = torch.zeros((1,)+input_shape)
        return self.nn(zeros).shape

    def forward(self, x):
        return self.nn(x)

# Structure parameter
structure = [
    ["conv2d", {
        "in_channels": 3,
        "out_channels": 32,
        "kernel_size": 8,
        "stride": 4
    }],
    ["activation", {}],
    ["conv2d", {
        "in_channels": 32,
        "out_channels": 64,
        "kernel_size": 4,
        "stride": 2
    }],
    ["activation", {}],
    ["conv2d", {
        "in_channels": 64,
        "out_channels": 64,
        "kernel_size": 3,
        "stride": 1
    }],
    ["activation", {}],
    ["flatten", {}],
    1024
]

# Create the network
net = CustomNet((3, 96, 96), structure).to("cuda")
print(net)


# Test the network
input_tensor = torch.randn(1, 3, 96, 96).to("cuda")
output = net(input_tensor)
print(output.shape)
summary(net, input_size=(3, 96, 96))