import torch

# Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
#
# NVIDIA CORPORATION & AFFILIATES and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION & AFFILIATES is strictly prohibited.


def position(position, features, layers, activation):
    """Use the position as input (i.e. no conditioning)
    Args:
        position   : [N, ..., d] tensor of coordinates
        features   : [N, ..., f] tensor of features
        layers     : nn.ModuleList of layers
        activation : activation function
    """

    h = position

    for i, l in enumerate(layers):
        h = activation(l(h))

    return h


def feature(position, features, layers, activation):
    """Use the features as input.
    Args:
        position   : [N, ..., d] tensor of coordinates
        features   : [N, ..., f] tensor of features
        layers     : nn.ModuleList of layers
        activation : activation function
    """

    h = features

    for i, l in enumerate(layers):
        h = activation(l(h))

    return h


def concat(position, features, layers, activation):
    """Concatenates the input onto the features, and then feeds into the input of the neural network.
    Args:
        position   : [N, ..., d] tensor of coordinates
        features   : [N, ..., f] tensor of features
        layers     : nn.ModuleList of layers
        activation : activation function
    """

    h = torch.cat([position, features], dim=-1)

    for i, l in enumerate(layers):
        h = activation(l(h))
    return h


def film_linear(position, features, layers, activation, with_batch=True):
    """Applies film conditioning (multiply only) on the network.
    Args:
        position   : [N, ..., d] tensor of coordinates
        features   : [N, ..., f] tensor of features
        layers     : nn.ModuleList of layers
        activation : activation function
    """
    feature_shape = features.shape[0]
    feature_dim = features.shape[-1]
    num_hidden = len(layers)
    # Maybe add assertion here... but if it errors, your feature_dim size is wrong
    if with_batch:
        features = features.reshape(feature_shape, 1, num_hidden, feature_dim // num_hidden)
    else:
        features = features.reshape(feature_shape, num_hidden, feature_dim // num_hidden)

    h = position

    for i, l in enumerate(layers):
        # Maybe also add another assertion here
        h = activation(l(h) * features[..., i, :])
    return h


def film_translate(position, features, layers, activation, with_batch=True):
    """Applies film conditioning (add only) on the network.
    Args:
        position   : [N, ..., d] tensor of coordinates
        features   : [N, ..., f] tensor of features
        layers     : nn.ModuleList of layers
        activation : activation function
    """
    feature_shape = features.shape[0]  # features.shape[:-1]
    feature_dim = features.shape[-1]
    num_hidden = len(layers)
    # Maybe add assertion here... but if it errors, your feature_dim size is wrong

    if with_batch:
        features = features.reshape(feature_shape, 1, num_hidden, feature_dim // num_hidden)
    else:
        features = features.reshape(feature_shape, num_hidden, feature_dim // num_hidden)

    h = position

    for i, l in enumerate(layers):
        # Maybe also add another assertion here
        h = activation(l(h) + features[..., i, :])
    return h


def film(position, features, layers, activation, with_batch=True):
    """Applies film conditioning (add only) on the network.
    Args:
        position   : [N, ..., d] tensor of coordinates
        features   : [N, ..., f] tensor of features
        layers     : nn.ModuleList of layers
        activation : activation function
    """
    feature_shape = features.shape[0]
    feature_dim = features.shape[-1]
    num_hidden = len(layers)
    # Maybe add assertion here... but if it errors, your feature_dim size is wrong
    
    if with_batch:
        features = features.reshape(
        feature_shape, 1, 2, num_hidden, feature_dim // (num_hidden * 2)
    )
    else:

        features = features.reshape(
            feature_shape, 2, num_hidden, feature_dim // (num_hidden * 2)
        )

    h = position

    for i, l in enumerate(layers):
        # Maybe also add another assertion here
        h = activation(l(h) * features[..., 0, i, :] + features[..., 1, i, :])
    return h