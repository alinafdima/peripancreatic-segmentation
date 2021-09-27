# ------------------------------------------------------
# File: network.py
# Author: Alina Dima <alina.dima@tum.de>
#
# Adapted from https://github.com/nikhilroxtomar/Polyp-Segmentation-using-UNET-in-TensorFlow-2.0
# Source https://github.com/nikhilroxtomar/Polyp-Segmentation-using-UNET-in-TensorFlow-2.0/blob/master/model.py
#
# ------------------------------------------------------

from tensorflow.keras.layers import Input, Activation, Concatenate, BatchNormalization
from tensorflow.keras.layers import Conv3D, MaxPool3D, UpSampling3D
from tensorflow.keras.models import Model


def convolutional_block(x, num_filters):
    x = Conv3D(num_filters, (3, 3, 3), padding="same")(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)

    x = Conv3D(num_filters, (3, 3, 3), padding="same")(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)

    return x


def build_model(input_size, output_channels=None):
    if output_channels is None:
        output_channels = [16, 32, 48, 64]
    else:
        assert type(output_channels) is list, \
            f'The output channels should be a list, got a {type(output_channels)} instead'
    print(f'Creating a U-Net with {len(output_channels)} levels..')

    inputs = Input(input_size)
    fw_nodes = []
    x = inputs
    for C in output_channels:
        x = convolutional_block(x, C)
        fw_nodes.append(x)
        x = MaxPool3D((2, 2, 2))(x)

    x = convolutional_block(x, output_channels[-1])
    output_channels.reverse()
    fw_nodes.reverse()

    for C, node in zip(output_channels, fw_nodes):
        x = UpSampling3D((2, 2, 2))(x)
        x = Concatenate()([x, node])
        x = convolutional_block(x, C)

    x = Conv3D(1, (1, 1, 1), padding="same")(x)
    x = Activation("sigmoid")(x)

    return Model(inputs, x)
