from pathlib import Path

import einops
import numpy as np
import torch
import torch.nn as nn

from coral.utils.data.dynamics_dataset import KEY_TO_INDEX
from coral.utils.data.graph_dataset import KEY_TO_INDEX as GRAPH_KEY_TO_INDEX


def get_reconstructions(inr, coords, modulations, z_mean, z_std, dataset_name=None):
    n_samples = coords.shape[0]
    T = modulations.shape[-1]

    if type(inr) == dict:
        c = len(list(inr.keys()))
        modulations = einops.rearrange(modulations, "b (l c) t -> b l c t", c=c)     

    if isinstance(inr, nn.Module):
        modulations = einops.rearrange(modulations, "b ... t -> (b t) ...")
        coords = einops.rearrange(coords, "b ... t -> (b t) ...")
        z_m = z_mean.repeat(n_samples*T, 1, 1).squeeze().cuda()
        z_s = z_std.repeat(n_samples*T, 1, 1).squeeze().cuda()
        with torch.no_grad():
            predictions = inr.modulated_forward(coords, modulations * z_s + z_m)
            predictions = einops.rearrange(predictions, "(b t) ... -> b ... t", t=T)
        return predictions
    elif type(inr) == dict:
        c = len(list(inr.keys()))
        # create predictions with shape ( (b t) dx dy c)
        predictions = torch.zeros(*coords.shape[:-2], c, T).cuda()
        for to_encode in inr.keys():
            idx = KEY_TO_INDEX[dataset_name][to_encode]
            inr_model = inr[to_encode]
            z_m = z_mean[to_encode].repeat(n_samples, 1, 1).squeeze().cuda()
            z_s = z_std[to_encode].repeat(n_samples, 1, 1).squeeze().cuda()

            with torch.no_grad():
                for t in range(T):
                    pred = inr_model.modulated_forward(
                        coords[..., t], modulations[..., idx, t] * z_s + z_m
                    )
                    predictions[..., idx, t] = pred.squeeze()

        return predictions #einops.rearrange(predictions, "(b t) ... -> b ... t", t=T)


def get_graph_reconstructions(inr, coords, modulations, batch, z_mean, z_std, dataset_name=None):
    n_samples = coords.shape[0]

    c = len(list(inr.keys()))
    modulations = einops.rearrange(modulations, "b (c l) -> b l c", c=c)     

    if type(inr) == dict:
        # create predictions with shape ( (b t) dx dy c)
        predictions = torch.zeros(*coords.shape[:-1], c).cuda()
        #print('pred totottoot', predictions.shape)
        for to_encode in inr.keys():
            idx = GRAPH_KEY_TO_INDEX[dataset_name][to_encode]
            inr_model = inr[to_encode]
            z_m = z_mean[to_encode].repeat(n_samples, 1, 1).squeeze().cuda()
            z_s = z_std[to_encode].repeat(n_samples, 1, 1).squeeze().cuda()

            #print("mod toto", modulations[batch].shape)
            #print('z_m', z_m.shape, z_s.shape)

            with torch.no_grad():
                pred = inr_model.modulated_forward(
                    coords, modulations[batch, :, idx] * z_s + z_m
                )
                predictions[..., idx] = pred.squeeze()

        return predictions